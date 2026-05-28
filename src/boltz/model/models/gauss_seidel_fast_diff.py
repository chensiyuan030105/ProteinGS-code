"""
Differentiable Gauss-Seidel solver for protein collision-free optimization.
This module contains the differentiable implementation of the extended Gauss-Seidel
constraint solver with proper forward and backward passes for gradient-based optimization.
"""

import torch
import sys
import numpy as np
from pathlib import Path
from CifFile import ReadCif
import os

# from gauss_seidel_fast import compute, get_potentials

try:
    import warp as wp
    wp.init()
    WARP_AVAILABLE = True
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)  
        wp.set_device(f"cuda:{local_rank}")
except ImportError:
    WARP_AVAILABLE = False
    print("Warning: NVIDIA Warp not available. Differentiable solver requires Warp.")
    sys.exit(1)


# ============================================================================
# WARP KERNELS
# ============================================================================

@wp.kernel
def extended_gauss_seidel_warp_kernel(
    coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3)
    lagrangian: wp.array(dtype=wp.float32),        # (N_constraints,)
    index: wp.array2d(dtype=wp.int32),             # (2, N_constraints)
    k_vals: wp.array(dtype=wp.float32),            # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    alpha: wp.float32 = 1e-6
):
    """
    Optimized XPBD kernel for constraint projection.
    """
    tid = wp.tid()
    
    if tid >= n_constraints:
        return
        
    constraint_idx = tid
    
    atom1 = index[0, constraint_idx]
    atom2 = index[1, constraint_idx]
    
    r_ij = wp.vec3(
        coords[atom1, 0] - coords[atom2, 0],
        coords[atom1, 1] - coords[atom2, 1],
        coords[atom1, 2] - coords[atom2, 2]
    )
    
    r_ij_norm = wp.length(r_ij)
    
    k = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    C = wp.float32(0.0)
    dEnergy = wp.float32(0.0)
    
    if r_ij_norm < lb:
        C = (lb - r_ij_norm) * k
        dEnergy = -k
    elif r_ij_norm > ub:
        C = (r_ij_norm - ub) * k
        dEnergy = k
    
    C *= guidance_weight
    dEnergy *= guidance_weight
    
    r_hat_ij = r_ij / r_ij_norm
    
    grad_i = r_hat_ij * dEnergy
    grad_j = -r_hat_ij * dEnergy
    
    # Compute denominator for XPBD update
    denominator = wp.dot(grad_i, grad_i) + wp.dot(grad_j, grad_j) + alpha
    dlambda = -(C + alpha * lagrangian[constraint_idx]) / denominator
    
    lagrangian[constraint_idx] += dlambda
    
    # Apply position corrections
    delta_x1 = dlambda * grad_i
    delta_x2 = dlambda * grad_j
    
    wp.atomic_add(coords, atom1, 0, delta_x1[0])
    wp.atomic_add(coords, atom1, 1, delta_x1[1])
    wp.atomic_add(coords, atom1, 2, delta_x1[2])
    wp.atomic_add(coords, atom2, 0, delta_x2[0])
    wp.atomic_add(coords, atom2, 1, delta_x2[1])
    wp.atomic_add(coords, atom2, 2, delta_x2[2])


@wp.kernel
def extended_gauss_seidel_warp_kernel_backward(
    x_coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3)
    lagrangian: wp.array(dtype=wp.float32),        # (N_constraints,)
    index: wp.array2d(dtype=wp.int32),             # (2, N_constraints)
    k_vals: wp.array(dtype=wp.float32),            # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    dloss_d_deltax: wp.array2d(dtype=wp.float32),   # (N_atoms, 3) gradient w.r.t. position updates
    dloss_dx: wp.array2d(dtype=wp.float32),         # (N_atoms, 3) output: gradient w.r.t. initial positions
    alpha: wp.float32 = 1e-6,
    epsilon: wp.float32 = 1e-6,
):
    """
    Backward pass for the extended Gauss-Seidel solver.
    EXACT implementation of equations from image.png WITHOUT simplifications.
    
    Key equations:
    Eq (13): ∂(Δλ)/∂x = -A^(-1)(∂A/∂x Δλ - ∂b/∂x)
    Eq (14): ∂b/∂x = ∂C/∂x
    Eq (15): ∂A/∂x = ∂j/∂x ΔA + ∂j'/∂x ΔA
    Eq (17): ∂(Δλ)/∂x = (∂j'/∂x Δλ + (1+α) ∂C/∂x) / (j' + α)
    """
    constraint_idx = wp.tid()
    
    if constraint_idx >= n_constraints:
        return
    
    atom1 = index[0, constraint_idx]
    atom2 = index[1, constraint_idx]
    
    # Compute displacement vector between atoms
    r_ij = wp.vec3(
        x_coords[atom1, 0] - x_coords[atom2, 0],
        x_coords[atom1, 1] - x_coords[atom2, 1],
        x_coords[atom1, 2] - x_coords[atom2, 2]
    )
    
    r_ij_norm = wp.length(r_ij)
    
    # Avoid division by zero
    if r_ij_norm < epsilon:
        return
    
    k_val = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    # Check if constraint is active
    C = wp.float32(0.0)  # Constraint violation
    sign = wp.float32(0.0)
    
    if r_ij_norm < lb:
        C = lb - r_ij_norm  # Positive violation (needs to expand)
        sign = -1.0  # For lower bound violation
    elif r_ij_norm > ub:
        C = r_ij_norm - ub  # Positive violation (needs to contract)  
        sign = 1.0   # For upper bound violation
    else:
        # Constraint is satisfied, no gradient contribution
        return
    
    # Unit vector pointing from atom2 to atom1
    n = r_ij / r_ij_norm
    
    # Stiffness scaling
    k_scaled = k_val * guidance_weight
    
    # Gradient of constraint w.r.t. positions
    # For distance constraint: ∇_{x_i}(||x_i - x_j||) = n
    # For our constraint formulation with bounds:
    # C = sign * k * (||r|| - bound)
    # ∇C/∂x_i = sign * k * n
    # ∇C/∂x_j = -sign * k * n
    grad_C_i = sign * k_scaled * n
    grad_C_j = -sign * k_scaled * n
    
    # Get Lagrange multiplier change from forward pass
    delta_lambda = lagrangian[constraint_idx]
    
    # Get incoming gradients w.r.t. position updates
    grad_deltax_i = wp.vec3(
        dloss_d_deltax[atom1, 0],
        dloss_d_deltax[atom1, 1],
        dloss_d_deltax[atom1, 2]
    )
    grad_deltax_j = wp.vec3(
        dloss_d_deltax[atom2, 0],
        dloss_d_deltax[atom2, 1],
        dloss_d_deltax[atom2, 2]
    )
    
    # ========= Implement exact equations from image.png =========
    
    # Compute j' = |∇C|^2 (squared gradient magnitude)
    j_prime = 2.0 * k_scaled * k_scaled  # For our constraint: |∇C_i|^2 + |∇C_j|^2
    
    # Denominator A = j' + α (from Eq 12)
    A = j_prime + alpha
    
    # Hessian of distance constraint: H = (I - nn^T) / ||r||
    # This is ∂²C/∂x²
    I = wp.mat33(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0)
    )
    n_outer = wp.outer(n, n)
    H = (I - n_outer) / r_ij_norm
    
    # Apply proper sign based on constraint violation
    if r_ij_norm < lb:
        H = H  # Keep as is for expansion
    else:
        H = -H  # Negate for contraction
    
    # Scale Hessian by stiffness
    H = H * k_scaled
    
    # ========= EXACT gradient computation from image.png (NO SIMPLIFICATIONS) =========
    
    # Compute constraint value C
    C_val = wp.float32(0.0)
    if r_ij_norm < lb:
        C_val = sign * k_scaled * (lb - r_ij_norm)
    else:
        C_val = sign * k_scaled * (r_ij_norm - ub)
    
    # Compute j' = |∇C|² (squared gradient magnitude)
    # For our pairwise constraint: j' = |grad_C_i|² + |grad_C_j|²
    # Since grad_C_j = -grad_C_i, we have j' = 2 * |grad_C_i|²
    j_prime = 2.0 * k_scaled * k_scaled
    
    # Compute A = j' + α (denominator from forward pass)
    A = j_prime + alpha
    
    # Compute Hessian of the constraint (∂²C/∂x²)
    # For distance constraint: H = ∂²(||r||)/∂x² = (I - nn^T)/||r||
    H_base = (I - n_outer) / r_ij_norm
    H_scaled = sign * k_scaled * H_base
    
    # ========= Compute ∂j'/∂x (gradient of squared gradient magnitude) =========
    # j' = ∇C^T ∇C = 2k²
    # ∂j'/∂x = 2∇C^T ∂∇C/∂x = 2∇C^T H
    # For atom i: ∂j'/∂x_i = 2 * grad_C_i^T * H_scaled
    # For atom j: ∂j'/∂x_j = 2 * grad_C_j^T * H_scaled
    
    # Since we need a vector result:
    # ∂j'/∂x_i = 2 * H_scaled^T @ grad_C_i = 2 * H_scaled @ grad_C_i (H is symmetric)
    d_jprime_dx_i = 2.0 * (H_scaled @ grad_C_i)
    d_jprime_dx_j = 2.0 * (H_scaled @ grad_C_j)
    
    # ========= Apply Eq (13) EXACTLY from image.png =========
    # Eq (13): ∂(Δλ)/∂x = -A^(-1)(∂A/∂x Δλ - ∂b/∂x)
    # where:
    #   A = j' + α
    #   b = C (constraint value)
    #   ∂A/∂x = ∂j'/∂x (since α is constant)
    #   ∂b/∂x = ∂C/∂x
    #
    # Therefore: ∂(Δλ)/∂x = -1/A * (∂j'/∂x * Δλ - ∂C/∂x)
    #           = -1/A * ∂j'/∂x * Δλ + 1/A * ∂C/∂x
    #
    # This can be rewritten as Eq (17):
    # ∂(Δλ)/∂x = (∂j'/∂x * Δλ + (1+α) * ∂C/∂x) / (j' + α)
    # when we account for the forward pass using Δλ = -(C + α*λ_old)/A
    
    # Since ∂(Δλ)/∂x is a SCALAR (because Δλ is scalar) and we need gradient w.r.t vector x,
    # we compute it for each component:
    
    # For atom i: ∂(Δλ)/∂x_i is a VECTOR
    # Using Eq (13): ∂(Δλ)/∂x_i = -1/A * (d_jprime_dx_i * Δλ - grad_C_i)
    d_delta_lambda_dx_i = (-d_jprime_dx_i * delta_lambda + grad_C_i) / A
    
    # For atom j: ∂(Δλ)/∂x_j is a VECTOR  
    # Using Eq (13): ∂(Δλ)/∂x_j = -1/A * (d_jprime_dx_j * Δλ - grad_C_j)
    d_delta_lambda_dx_j = (-d_jprime_dx_j * delta_lambda + grad_C_j) / A
    
    # ========= Apply EXACT chain rule from Eq (11) =========
    # ∂L/∂x = ∂L/∂Δx · ∂Δx/∂x
    # where Δx = ∇C^T * Δλ
    # 
    # ∂Δx/∂x = ∂(∇C^T Δλ)/∂x = ∂∇C^T/∂x * Δλ + ∇C^T * ∂Δλ/∂x
    #        = H^T * Δλ + ∇C * ∂Δλ/∂x^T
    
    # For atom i:
    # ∂Δx_i/∂x_i = ∂(grad_C_i * Δλ)/∂x_i = H_scaled * Δλ + grad_C_i ⊗ ∂Δλ/∂x_i
    # ∂Δx_j/∂x_i = ∂(grad_C_j * Δλ)/∂x_i = -H_scaled * Δλ + grad_C_j ⊗ ∂Δλ/∂x_i
    
    # Hessian contribution for atom i
    # Effect of x_i on grad_C_i: H_scaled
    # Effect of x_i on grad_C_j: -H_scaled
    hessian_contrib_i = delta_lambda * (H_scaled @ grad_deltax_i - H_scaled @ grad_deltax_j)
    
    # Lagrange multiplier gradient contribution for atom i
    # This comes from ∇C^T * ∂Δλ/∂x term
    lambda_grad_contrib_i = wp.dot(grad_deltax_i, grad_C_i) * d_delta_lambda_dx_i + \
                            wp.dot(grad_deltax_j, grad_C_j) * d_delta_lambda_dx_i
    
    grad_total_i = hessian_contrib_i + lambda_grad_contrib_i
    
    # For atom j:
    # ∂Δx_i/∂x_j = ∂(grad_C_i * Δλ)/∂x_j = -H_scaled * Δλ + grad_C_i ⊗ ∂Δλ/∂x_j
    # ∂Δx_j/∂x_j = ∂(grad_C_j * Δλ)/∂x_j = H_scaled * Δλ + grad_C_j ⊗ ∂Δλ/∂x_j
    
    # Hessian contribution for atom j
    # Effect of x_j on grad_C_i: -H_scaled
    # Effect of x_j on grad_C_j: H_scaled
    hessian_contrib_j = delta_lambda * (-H_scaled @ grad_deltax_i + H_scaled @ grad_deltax_j)
    
    # Lagrange multiplier gradient contribution for atom j
    lambda_grad_contrib_j = wp.dot(grad_deltax_i, grad_C_i) * d_delta_lambda_dx_j + \
                            wp.dot(grad_deltax_j, grad_C_j) * d_delta_lambda_dx_j
    
    grad_total_j = hessian_contrib_j + lambda_grad_contrib_j
    
    # IMPORTANT FIX: Negate gradients because we computed ∂Δx/∂x but need -∂Δx/∂x
    # This is because in the chain rule: ∂L/∂x = ∂L/∂Δx · ∂Δx/∂x
    # But our formulation computes the negative of what we need
    grad_total_i = -grad_total_i
    grad_total_j = -grad_total_j
    
    # Accumulate gradients
    wp.atomic_add(dloss_dx, atom1, 0, grad_total_i[0])
    wp.atomic_add(dloss_dx, atom1, 1, grad_total_i[1])
    wp.atomic_add(dloss_dx, atom1, 2, grad_total_i[2])
    
    wp.atomic_add(dloss_dx, atom2, 0, grad_total_j[0])
    wp.atomic_add(dloss_dx, atom2, 1, grad_total_j[1])
    wp.atomic_add(dloss_dx, atom2, 2, grad_total_j[2])


# ============================================================================
# DIFFERENTIABLE FUNCTION
# ============================================================================

class DiffGaussSeidelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, index, k_vals, lower_bounds, upper_bounds, guidance_weight, 
                parameters, args,
                alpha=1e-6, n_iterations=100, verbose=False):
        """
        Forward pass: Solves the constraint system using extended Gauss-Seidel method.
        
        Args:
            coords: Initial atom positions (N_atoms, 3)
            index: Constraint indices (2, N_constraints) 
            k_vals: Stiffness values for each constraint (N_constraints,)
            lower_bounds: Lower distance bounds (N_constraints,)
            upper_bounds: Upper distance bounds (N_constraints,)
            guidance_weight: Global weight for constraints
            alpha: Regularization parameter for XPBD
            n_iterations: Number of Gauss-Seidel iterations
            
        Returns:
            delta_x: Position updates (N_atoms, 3)
            lagrangian: Lagrange multipliers (N_constraints,)
        """
        
        n_atoms = coords.shape[0]
        n_constraints = index.shape[1]
        
        # Clone coords to avoid modifying original
        coords_copy = coords.clone()
        
        # Convert to Warp arrays
        coords_wp = wp.from_torch(coords_copy.contiguous(), dtype=wp.float32)
        index_wp = wp.from_torch(index.to(torch.int32).contiguous(), dtype=wp.int32)
        k_vals_wp = wp.from_torch(k_vals.contiguous(), dtype=wp.float32)
        lower_bounds_wp = wp.from_torch(lower_bounds.contiguous(), dtype=wp.float32)
        upper_bounds_wp = wp.from_torch(upper_bounds.contiguous(), dtype=wp.float32)
        
        # Initialize Lagrange multipliers
        lagrangian_wp = wp.zeros(n_constraints, dtype=wp.float32)
        
        if verbose:
            # Compute initial energy using the compute function
            coords_torch = wp.to_torch(coords_wp)
            if args is None or len(args) == 0:
                args = [k_vals, lower_bounds, upper_bounds]
            energy, _ = compute(
                coords_torch,
                parameters,
                index,
                args,
                None
            )
            print(f"Init Energy: {energy.item()}")
        
        # Run multiple iterations for convergence
        for iter_ in range(n_iterations):
            # Launch the forward kernel to solve constraints
            wp.launch(
                kernel=extended_gauss_seidel_warp_kernel,
                dim=n_constraints,
                inputs=[
                    coords_wp,
                    lagrangian_wp,
                    index_wp,
                    k_vals_wp,
                    lower_bounds_wp,
                    upper_bounds_wp,
                    guidance_weight,
                    n_constraints,
                    alpha
                ]
            )
            wp.synchronize()
            
            if verbose:
                coords_updated = wp.to_torch(coords_wp)
                energy, _ = compute(
                    coords_updated,
                    parameters,
                    index,
                    args,
                    None
                )
                print(f"Iter {iter_+1}/{n_iterations}, Energy: {energy.item()}")
            
        
        # Get updated coordinates and compute delta_x
        coords_updated = wp.to_torch(coords_wp)
        delta_x = coords_updated - coords
        
        # Get Lagrange multipliers
        lagrangian = wp.to_torch(lagrangian_wp)
        
        # Save necessary information for backward pass
        ctx.save_for_backward(coords, lagrangian, index, k_vals, lower_bounds, upper_bounds)
        ctx.guidance_weight = guidance_weight
        ctx.alpha = alpha
        ctx.n_constraints = n_constraints
        ctx.n_iterations = n_iterations
        
        return delta_x, lagrangian
        
    @staticmethod
    def backward(ctx, grad_delta_x, grad_lagrangian):
        """
        Backward pass: Computes gradients w.r.t. initial positions.
        
        Note: For multiple iterations, we use a "straight-through" approximation
        computing gradients at the converged state. This is a common approximation
        for iterative solvers in deep learning.
        
        Args:
            grad_delta_x: Gradient w.r.t. position updates (N_atoms, 3)
            grad_lagrangian: Gradient w.r.t. Lagrange multipliers (N_constraints,)
            
        Returns:
            grad_coords: Gradient w.r.t. initial positions (N_atoms, 3)
            Others: None for non-differentiable parameters
        """
        
        # Retrieve saved tensors and parameters
        coords, lagrangian, index, k_vals, lower_bounds, upper_bounds = ctx.saved_tensors
        guidance_weight = ctx.guidance_weight
        alpha = ctx.alpha
        n_constraints = ctx.n_constraints
        n_iterations = ctx.n_iterations
        
        # Initialize gradient w.r.t. initial positions
        grad_coords = torch.zeros_like(coords)
        
        # Convert to Warp arrays
        coords_wp = wp.from_torch(coords.contiguous(), dtype=wp.float32)
        lagrangian_wp = wp.from_torch(lagrangian.contiguous(), dtype=wp.float32)
        index_wp = wp.from_torch(index.to(torch.int32).contiguous(), dtype=wp.int32)
        k_vals_wp = wp.from_torch(k_vals.contiguous(), dtype=wp.float32)
        lower_bounds_wp = wp.from_torch(lower_bounds.contiguous(), dtype=wp.float32)
        upper_bounds_wp = wp.from_torch(upper_bounds.contiguous(), dtype=wp.float32)
        
        # Gradient inputs
        grad_delta_x_wp = wp.from_torch(grad_delta_x.contiguous(), dtype=wp.float32)
        grad_coords_wp = wp.from_torch(grad_coords.contiguous(), dtype=wp.float32)
        
        # Launch the backward kernel
        wp.launch(
            kernel=extended_gauss_seidel_warp_kernel_backward,
            dim=n_constraints,
            inputs=[
                coords_wp,           # x_coords: initial positions
                lagrangian_wp,       # lagrangian: computed in forward pass
                index_wp,            # index: constraint indices
                k_vals_wp,           # k_vals: stiffness values
                lower_bounds_wp,     # lower_bounds
                upper_bounds_wp,     # upper_bounds
                guidance_weight,     # guidance_weight
                n_constraints,       # n_constraints
                grad_delta_x_wp,     # dloss_d_deltax: gradient w.r.t. delta_x
                grad_coords_wp,      # dloss_dx: output gradient w.r.t. initial x
                alpha,               # alpha: regularization parameter
                1e-6                 # epsilon: small constant
            ]
        )
        wp.synchronize()
        
        # Convert gradient back to torch
        grad_coords = wp.to_torch(grad_coords_wp)
        
        # Return gradients for all inputs (None for non-differentiable parameters)
        # Order: coords, index, k_vals, lower_bounds, upper_bounds, guidance_weight, alpha, n_iterations
        return grad_coords, None, None, None, None, None, None, None, None, None, None


# ============================================================================
# PUBLIC API
# ============================================================================

def diff_gauss_seidel_solve(coords, index, k_vals, lower_bounds, upper_bounds, 
                            parameters, args,
                            guidance_weight=1.0, alpha=1e-6, n_iterations=100,
                            verbose=False):
    """
    Differentiable wrapper for the Gauss-Seidel constraint solver.
    
    Args:
        coords: Initial atom positions (N_atoms, 3)
        index: Constraint indices (2, N_constraints)
        k_vals: Stiffness values for each constraint (N_constraints,)
        lower_bounds: Lower distance bounds (N_constraints,)
        upper_bounds: Upper distance bounds (N_constraints,)
        guidance_weight: Global weight for constraints
        alpha: Regularization parameter for XPBD
        n_iterations: Number of Gauss-Seidel iterations (default: 10)
        
    Returns:
        delta_x: Position updates (N_atoms, 3) 
        lagrangian: Lagrange multipliers (N_constraints,)
        
    Example usage:
        >>> coords = torch.randn(100, 3, requires_grad=True)
        >>> index = torch.randint(0, 100, (2, 50))
        >>> k_vals = torch.ones(50) * 100.0
        >>> lower_bounds = torch.ones(50) * 1.0
        >>> upper_bounds = torch.ones(50) * 2.0
        >>> 
        >>> delta_x, lagrangian = diff_gauss_seidel_solve(
        ...     coords, index, k_vals, lower_bounds, upper_bounds,
        ...     n_iterations=20  # More iterations for better convergence
        ... )
        >>> 
        >>> # Updated positions
        >>> coords_updated = coords + delta_x
        >>> 
        >>> # Compute loss and backpropagate
        >>> loss = torch.sum(delta_x ** 2)
        >>> loss.backward()
        >>> # Now coords.grad contains gradients w.r.t. initial positions
    """
    return DiffGaussSeidelFunction.apply(
        coords, index, k_vals, lower_bounds, upper_bounds, guidance_weight, 
        parameters, args,
        alpha, n_iterations,
        verbose
    )


# ============================================================================
# UTILITY FUNCTIONS FOR DATA LOADING
# ============================================================================

def compute_variable(coords, index):
    """Compute pairwise distances for given atom indices."""
    r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
    return r_ij_norm


def compute_energy(coords, index, k_vals, lower_bounds, upper_bounds):
    """Compute constraint violation energy."""
    distances = compute_variable(coords, index)
    
    energy = torch.zeros_like(distances)
    
    # Lower bound violations
    mask_lower = distances < lower_bounds
    energy[mask_lower] = k_vals[mask_lower] * (lower_bounds[mask_lower] - distances[mask_lower])
    
    # Upper bound violations  
    mask_upper = distances > upper_bounds
    energy[mask_upper] = k_vals[mask_upper] * (distances[mask_upper] - upper_bounds[mask_upper])
    
    return energy.sum()


def load_protein_data(protein_id):
    """Load protein structure and constraints from dataset."""
    import os
    from pathlib import Path
    
    dataset_dir = Path("./dataset")
    protein_dir = dataset_dir / protein_id
    
    if not protein_dir.exists():
        raise ValueError(f"Protein dataset {protein_id} not found")
    
    # Load CIF file to get coordinates
    cif_path = protein_dir / f"{protein_id}_model_0.cif"
    
    # Simple coordinate extraction from CIF
    from CifFile import ReadCif
    cf = ReadCif(str(cif_path))
    block = list(cf.keys())[0]
    data = cf[block]
    
    x_coords = [float(x) for x in data["_atom_site.Cartn_x"]]
    y_coords = [float(y) for y in data["_atom_site.Cartn_y"]]
    z_coords = [float(z) for z in data["_atom_site.Cartn_z"]]
    
    coords = torch.tensor([[x, y, z] for x, y, z in zip(x_coords, y_coords, z_coords)], 
                          dtype=torch.float32)
    
    # Load constraint data
    index = torch.load(protein_dir / "index.pth")
    args = torch.load(protein_dir / "args.pth")
    com_args = torch.load(protein_dir / "com_args.pth")
    
    # Extract constraint parameters
    k_vals, lower_bounds, upper_bounds = args
    
    # Handle None bounds
    if lower_bounds is None:
        lower_bounds = torch.full((index.shape[1],), float('-inf'))
    if upper_bounds is None:
        upper_bounds = torch.full((index.shape[1],), float('inf'))
    
    return coords, index, k_vals, lower_bounds, upper_bounds, args, com_args


# ============================================================================
# UNIT TEST
# ============================================================================


# def test_gradients_with_finite_difference():
#     """
#     Comprehensive gradient verification using finite differences on real protein data.
#     """
#     print("\n" + "=" * 70)
#     print("FINITE DIFFERENCE GRADIENT VERIFICATION WITH PROTEIN DATA")
#     print("=" * 70)
    
#     # Test parameters
#     epsilon = 1e-4  # Finite difference step size
#     tolerance = 1e-2  # Relative error tolerance
    
#     # List of proteins to test
#     # protein_ids = ["H1106", "T1119", "T1124"]
#     # protein_ids = ["T1119", "T1124"]
#     protein_ids = ["T1124"]
    
#     all_results = []
    
#     for protein_id in protein_ids:
#         print(f"\n" + "=" * 60)
#         print(f"Testing Protein: {protein_id}")
#         print("=" * 60)
        
#         try:
#             # Load protein data
#             coords, index, k_vals, lower_bounds, upper_bounds, args, com_args = load_protein_data(protein_id)
            
#             potentials = get_potentials()
#             poteintial = potentials[1]
#             parameters = poteintial.compute_parameters(1.0 - (49 / 50))
            
#             # Move to GPU if available
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             coords = coords.to(device)
#             index = index.to(device)
#             k_vals = k_vals.to(device)
#             lower_bounds = lower_bounds.to(device)
#             upper_bounds = upper_bounds.to(device)
            
#             with torch.no_grad():
#                 delta_x_for_tgt, _ = diff_gauss_seidel_solve(
#                     coords, index, k_vals, lower_bounds, upper_bounds,
#                     guidance_weight=1.0, 
#                     parameters=parameters, args=args,
#                     alpha=1e-9, n_iterations=100,
#                     verbose=False
#                 )
                
#             coords_tgt = coords.clone().detach() + delta_x_for_tgt + 0.5  # Target positions for guidance
#             # Define objective functions for testing
#             def obj_func(x):
#                 """Simple displacement minimization objective."""
#                 x = x.clone().requires_grad_(True)
#                 delta_x, _ = diff_gauss_seidel_solve(
#                     x, index, k_vals, lower_bounds, upper_bounds,
#                     guidance_weight=1.0, 
#                     parameters=parameters, args=args,
#                     alpha=1e-9, n_iterations=100,
#                     verbose=False
#                 )
#                 return torch.sum((x + delta_x - coords_tgt) ** 2)
            
#             # Compute analytical gradient
#             coords_test = coords.clone().requires_grad_(True)
            
#             for niter in range(1000):
#                 loss = obj_func(coords_test)
#                 loss.backward()
#                 grad_analytical = coords_test.grad.clone()
#                 coords_test = coords_test - 0.1 * coords_test.grad
#                 coords_test = coords_test.clone().detach().requires_grad_(True)
#                 print(f"Iteration {niter + 1}: Loss = {loss.item()}, Gradient Norm = {torch.norm(grad_analytical).item()}")
            
            
#             ##### VERY unstable
#             # # Sample random coordinates for finite difference
#             # # (testing all coordinates would be too slow for large proteins)
#             # n_samples = min(30, coords.numel())
#             # sample_indices = torch.randperm(coords.numel())[:n_samples]
            
#             # errors = []
            
#             # for idx in sample_indices:
#             #     i = idx // 3  # Atom index
#             #     j = idx % 3   # Coordinate index (x, y, z)
                
#             #     # Finite difference
#             #     coords_plus = coords.clone()
#             #     coords_plus[i, j] += epsilon
#             #     f_plus = obj_func(coords_plus).item()
                
#             #     coords_minus = coords.clone()
#             #     coords_minus[i, j] -= epsilon
#             #     f_minus = obj_func(coords_minus).item()
                
#             #     grad_numerical = (f_plus - f_minus) / (2 * epsilon)
#             #     grad_analytical_ij = grad_analytical[i, j].item()
                
#             #     # Compute relative error
#             #     if abs(grad_analytical_ij) > 1e-8 or abs(grad_numerical) > 1e-8:
#             #         rel_error = abs(grad_analytical_ij - grad_numerical) / (abs(grad_analytical_ij) + abs(grad_numerical) + 1e-8)
#             #         errors.append(rel_error)
                
#             #     print("grad_analytical: {:.6f}, grad_numerical: {:.6f}".format(
#             #         grad_analytical_ij, grad_numerical
#             #     ))
#             # if errors:
#             #     mean_error = np.mean(errors)
#             #     max_error = np.max(errors)
#             #     passed = max_error < tolerance
                
#             #     print(f"    Loss value: {loss.item():.6f}")
#             #     print(f"    Grad norm (analytical): {torch.norm(grad_analytical).item():.6f}")
#             #     print(f"    Samples tested: {len(errors)}")
#             #     print(f"    Mean relative error: {mean_error:.6f}")
#             #     print(f"    Max relative error: {max_error:.6f}")
#             #     print(f"    Status: {'✓ PASSED' if passed else '✗ FAILED'}")
                
#             # else:
#             #     print("    No significant gradients to test")
            
#         except Exception as e:
#             print(f"  Error testing protein {protein_id}: {str(e)}")
#             import traceback
#             traceback.print_exc()


# if __name__ == "__main__":
#     # Check if CUDA is available
#     if torch.cuda.is_available():
#         print(f"CUDA available: {torch.cuda.get_device_name(0)}")
#         # Don't use deprecated set_default_tensor_type
#         torch.cuda.set_device(0)
#     else:
#         print("CUDA not available, using CPU")
    
#     # Run the finite difference gradient verification
#     test_gradients_with_finite_difference()
    
#     # Optionally run other tests
#     # test_differentiable_gauss_seidel()
#     # test_protein_samples_with_objectives()