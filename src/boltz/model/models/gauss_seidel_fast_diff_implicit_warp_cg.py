"""
Differentiable Gauss-Seidel solver with Warp's native CG solver for memory efficiency.
This implementation uses warp.optim.linear.cg for the backward pass, providing
better memory efficiency compared to PyTorch sparse operations.
"""

import torch
import sys
import numpy as np
from pathlib import Path
from CifFile import ReadCif

from gauss_seidel_fast import compute, get_potentials
try:
    import warp as wp
    import warp.sparse as wps
    import warp.optim as wpo
    import warp.optim.linear as wpl
    wp.init()
    WARP_AVAILABLE = True
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
        # C = (lb - r_ij_norm) * k
        # dEnergy = -k
        # NOTE: C does not have to be positive! as we are using CTC in the objective
        # C = (r_ij_norm - lb) * k
        C = (lb - r_ij_norm) * k
        dEnergy = -k
    elif r_ij_norm > ub:
        C = (r_ij_norm - ub) * k
        dEnergy = k
    
    C *= guidance_weight
    dEnergy *= guidance_weight
    
    r_hat_ij = r_ij / r_ij_norm
    
    grad_i = r_hat_ij * dEnergy
    grad_j = -grad_i
    
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
def build_system_matrix_kernel(
    x_coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3) - converged positions
    index: wp.array2d(dtype=wp.int32),               # (2, N_constraints)
    k_vals: wp.array(dtype=wp.float32),              # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),        # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),        # (N_constraints,)
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    # Sparse matrix outputs (COO format)
    row_indices: wp.array(dtype=wp.int32),           # Row indices for sparse matrix
    col_indices: wp.array(dtype=wp.int32),           # Column indices for sparse matrix  
    values: wp.array(dtype=wp.float32),              # Values for sparse matrix
    
    alpha: wp.float32 = 1e-6,
):
    """
    Build the system matrix (H + ΔH + I) for implicit differentiation.
    From image_4.png: H = α⁻¹∇C^T∇C and ΔH = α⁻¹∇C : ∇²C
    
    This kernel populates a sparse matrix in COO format.
    Each constraint contributes a 6x6 block (2 atoms x 3 dimensions).
    """
    constraint_idx = wp.tid()
    
    if constraint_idx >= n_constraints:
        return
    
    atom1 = index[0, constraint_idx]
    atom2 = index[1, constraint_idx]
    
    # Compute displacement vector at converged state
    r_ij = wp.vec3(
        x_coords[atom1, 0] - x_coords[atom2, 0],
        x_coords[atom1, 1] - x_coords[atom2, 1],
        x_coords[atom1, 2] - x_coords[atom2, 2]
    )
    
    r_ij_norm = wp.length(r_ij)
    
    # # Avoid division by zero
    # if r_ij_norm < epsilon:
    #     return
    
    k_val = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    # Check if constraint is active at converged state
    C = wp.float32(0.0)
    grad_scalar = wp.float32(0.0)
    # grad_scalar = wp.float32(1.0)
    
    if r_ij_norm < lb:
        # C = (r_ij_norm - lb) * k_val
        C = (lb - r_ij_norm) * k_val
        grad_scalar = wp.float32(-1.0)
    elif r_ij_norm > ub:
        C = (r_ij_norm - ub) * k_val
        grad_scalar = wp.float32(1.0)
    
    C *= guidance_weight
        
    # Unit vector
    n = r_ij / r_ij_norm # (r_ij_norm + epsilon)
    
    # Gradient of constraint at converged state
    k_scaled = k_val * guidance_weight
    grad_C_i = k_scaled * n * grad_scalar
    grad_C_j = -grad_C_i # Equal and opposite for pairwise constraint
    
    alpha_inv = 1.0 / alpha
    # alpha_inv = 1e3 ### TODO Worth to investigate why, a too large alpha_inv is very unstable
    # alpha_inv = 1.0 / 1e-1
    
    # ===== Compute H = α⁻¹∇C^T∇C (outer product term) =====
    # This contributes a rank-1 update to the system matrix
    
    # ===== Compute ΔH = α⁻¹∇C : ∇²C (Hessian term) =====
    # For distance constraint: ∇²C = k * sign * (I - nn^T) / ||r||
    
    # Compute Hessian matrix
    I = wp.mat33(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0)
    )
    n_outer_n = wp.outer(n, n)
    hessian_base = (I - n_outer_n) / r_ij_norm # * grad_scalar # (r_ij_norm + epsilon)  # Add epsilon to avoid div by zero
    hessian_C = k_scaled * hessian_base
    
    # The system has a 2x2 block structure for atoms i and j
    # Each block is 3x3 (for x,y,z dimensions)
    
    # Base index for this constraint's contribution to sparse matrix
    # Each constraint contributes up to 36 entries (6x6 block)
    base_idx = constraint_idx * 36
    
    # Fill the sparse matrix entries for this constraint
    # We need to add contributions to:
    # - (atom1, atom1) block: H_ii + ΔH_ii
    # - (atom1, atom2) block: H_ij + ΔH_ij  
    # - (atom2, atom1) block: H_ji + ΔH_ji
    # - (atom2, atom2) block: H_jj + ΔH_jj
    
    entry_idx = 0
    
    # project delta_h_val
    sC = grad_scalar * C
    # coeff = wp.max(sC, wp.float32(0.0))
    coeff = sC
    
    # Block (i,i): Effect on atom1 from atom1
    for d1 in range(3):
        for d2 in range(3):
            row = atom1 * 3 + d1
            col = atom1 * 3 + d2
            
            # H contribution: α⁻¹ * grad_C_i[d1] * grad_C_i[d2]
            h_val = alpha_inv * grad_C_i[d1] * grad_C_i[d2]
            
            # ΔH contribution: α⁻¹ * hessian_C[d1,d2]
            delta_h_val = alpha_inv * hessian_C[d1, d2] * coeff
            
            # Total contribution
            total_val = h_val + delta_h_val
            
            # Store in sparse matrix
            idx = base_idx + entry_idx
            row_indices[idx] = row
            col_indices[idx] = col
            values[idx] = total_val
            
            entry_idx += 1
    
    # Block (i,j): Effect on atom1 from atom2
    for d1 in range(3):
        for d2 in range(3):
            row = atom1 * 3 + d1
            col = atom2 * 3 + d2
            
            # H contribution: α⁻¹ * grad_C_i[d1] * grad_C_j[d2]
            h_val = alpha_inv * grad_C_i[d1] * grad_C_j[d2]
            
            # ΔH contribution: -α⁻¹ * hessian_C[d1,d2] (negative due to coupling)
            delta_h_val = -alpha_inv * hessian_C[d1, d2] * coeff
            
            # Total contribution
            total_val = h_val + delta_h_val
            
            # Store in sparse matrix
            idx = base_idx + entry_idx
            row_indices[idx] = row
            col_indices[idx] = col
            values[idx] = total_val
            
            entry_idx += 1
    
    # Block (j,i): Effect on atom2 from atom1
    for d1 in range(3):
        for d2 in range(3):
            row = atom2 * 3 + d1
            col = atom1 * 3 + d2
            
            # H contribution: α⁻¹ * grad_C_j[d1] * grad_C_i[d2]
            h_val = alpha_inv * grad_C_j[d1] * grad_C_i[d2]
            
            # ΔH contribution: -α⁻¹ * hessian_C[d1,d2]
            delta_h_val = -alpha_inv * hessian_C[d1, d2] * coeff
            
            # Total contribution
            total_val = h_val + delta_h_val
            
            # Store in sparse matrix
            idx = base_idx + entry_idx
            row_indices[idx] = row
            col_indices[idx] = col
            values[idx] = total_val
            
            entry_idx += 1
    
    # Block (j,j): Effect on atom2 from atom2
    for d1 in range(3):
        for d2 in range(3):
            row = atom2 * 3 + d1
            col = atom2 * 3 + d2
            
            # H contribution: α⁻¹ * grad_C_j[d1] * grad_C_j[d2]
            h_val = alpha_inv * grad_C_j[d1] * grad_C_j[d2]
            
            # ΔH contribution: α⁻¹ * hessian_C[d1,d2]
            delta_h_val = alpha_inv * hessian_C[d1, d2] * coeff
            
            # Total contribution
            total_val = h_val + delta_h_val
            
            # Store in sparse matrix
            idx = base_idx + entry_idx
            row_indices[idx] = row
            col_indices[idx] = col
            values[idx] = total_val
            
            entry_idx += 1


# ============================================================================
# DIFFERENTIABLE FUNCTION
# ============================================================================

class DiffGaussSeidelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, index, k_vals, lower_bounds, upper_bounds, guidance_weight, 
                parameters, args,
                alpha=1e-6, n_iterations=100, 
                verbose=False):
        """
        Forward pass: Solves the constraint system using extended Gauss-Seidel method.
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
            
            if verbose or iter_ == n_iterations - 1:
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
        # delta_x = coords_updated - coords
        
        # Get Lagrange multipliers
        lagrangian = wp.to_torch(lagrangian_wp)
        
        # Save necessary information for backward pass
        # Save both initial and converged positions for implicit differentiation
        ctx.save_for_backward(coords, coords_updated, lagrangian, index, k_vals, lower_bounds, upper_bounds)
        ctx.guidance_weight = guidance_weight
        ctx.alpha = alpha
        ctx.n_constraints = n_constraints
        ctx.n_atoms = n_atoms
        ctx.n_iterations = n_iterations
        
        return coords_updated, lagrangian
        
    @staticmethod
    def backward(ctx, grad_coords_updated, grad_lagrangian):
        """
        Backward pass using iterative scheme from image_4.png equations (15)-(16).
        
        Outer iteration: z_{k+1} = z_k - H^{-1}[(H + ΔH + I)z_k - (∂L/∂x)^T]
        Inner solver: Solve Hz = b using Warp's CG solver
        """
        
        # Retrieve saved tensors and parameters
        coords_initial, coords_converged, lagrangian, index, k_vals, lower_bounds, upper_bounds = ctx.saved_tensors
        guidance_weight = ctx.guidance_weight
        alpha = ctx.alpha
        n_constraints = ctx.n_constraints
        n_atoms = ctx.n_atoms
        
        device = coords_converged.device
        
        # Dimension of the linear system (3 coords per atom)
        n_dim = n_atoms * 3
        
        # ===== Build system matrices H and ΔH separately =====
        # We need H for CG solver and (H + ΔH) for the iteration
        max_nnz = n_constraints * 36
        
        # Arrays for full H + ΔH matrix
        row_indices_full_wp = wp.zeros(max_nnz, dtype=wp.int32)
        col_indices_full_wp = wp.zeros(max_nnz, dtype=wp.int32)
        values_full_wp = wp.zeros(max_nnz, dtype=wp.float32)
        
        # Convert tensors to Warp
        coords_converged_wp = wp.from_torch(coords_converged.contiguous(), dtype=wp.float32)
        index_wp = wp.from_torch(index.to(torch.int32).contiguous(), dtype=wp.int32)
        k_vals_wp = wp.from_torch(k_vals.contiguous(), dtype=wp.float32)
        lower_bounds_wp = wp.from_torch(lower_bounds.contiguous(), dtype=wp.float32)
        upper_bounds_wp = wp.from_torch(upper_bounds.contiguous(), dtype=wp.float32)
        
        
        # Build H + ΔH matrix (full system matrix)
        wp.launch(
            kernel=build_system_matrix_kernel,
            dim=n_constraints,
            inputs=[
                coords_converged_wp,
                index_wp,
                k_vals_wp,
                lower_bounds_wp,
                upper_bounds_wp,
                guidance_weight,
                n_constraints,
                row_indices_full_wp,
                col_indices_full_wp,
                values_full_wp,
                
                alpha,
            ]
        )
        wp.synchronize()
        
        
        # ===== Solve (H + ΔH + I)z = b using Warp's CG solver =====
        
        # Count actual non-zero entries from the kernel output
        nnz_full = n_constraints * 36  # Each constraint contributes 36 entries
        # K = H + ΔH  (CSR = BSR with 1x1 blocks)
        K = wps.bsr_from_triplets(
            rows_of_blocks=n_dim,
            cols_of_blocks=n_dim,
            rows=row_indices_full_wp[:nnz_full],
            columns=col_indices_full_wp[:nnz_full],
            values=values_full_wp[:nnz_full],
        )  # BsrMatrix on device
        
        # A = K + I  -> add identity on the same device using a diagonal BSR
        ones_diag_torch = torch.ones(n_dim, device=device, dtype=torch.float32)
        I = wps.bsr_diag(diag=wp.from_torch(ones_diag_torch, dtype=wp.float32))  # CSR identity
        A = wps.bsr_axpy(K, y=I, alpha=1.0, beta=10.0)  # A = K + I

        # ---- Right-hand side and solution buffer ----
        b_wp = wp.from_torch(grad_coords_updated.reshape(-1).contiguous(), dtype=wp.float32)
        x_wp = wp.zeros(n_dim, dtype=wp.float32, device=coords_converged_wp.device)  # initial guess x0 = 0

        # ---- Optional preconditioner ----
        # Use ctx.preconditioner_type if provided; e.g., "diag_abs", "diag", or "id".
        ptype = getattr(ctx, "preconditioner_type", "diag_abs")
        M = wp.optim.linear.preconditioner(A, ptype=ptype) if ptype is not None else None  # LinearOperator
        # (Supported types per docs: "diag", "diag_abs", "id".) :contentReference[oaicite:1]{index=1}

        # ---- Solve with built-in Conjugate Gradient ----
        tol          = float(getattr(ctx, "cg_tol", 1e-4))
        maxiter      = int(getattr(ctx, "cg_maxiter", min(n_dim, 10000)))
        check_every  = int(getattr(ctx, "cg_check_every", 10))
        use_cg_graph = bool(getattr(ctx, "cg_use_cuda_graph", True))

        # Optionally, attach a host-side callback for logging every `check_every` iters
        callback = None
        if True:
            def _cb(it, r_norm, a_tol):
                # r_norm and a_tol are scalars per docs
                print(f"[CG] iter={it}  ||r||={r_norm:.3e}  atol={a_tol:.3e}")
            callback = _cb

        # Runs in-place: writes solution into x_wp; returns (iters, resid, atol_used)
        wp.optim.linear.cg(
            A, b_wp, x_wp,
            tol=tol,
            maxiter=maxiter,
            M=M,
            callback=callback,
            check_every=check_every,
            use_cuda_graph=use_cg_graph,
        )  # Built-in SPD solver for Ax=b. :contentReference[oaicite:2]{index=2}

        # ---- Return grads ----
        grad_coords = wp.to_torch(x_wp).reshape(n_atoms, 3)

        # Mirror your original return signature: grad for coords_initial + the rest None
        return grad_coords, None, None, None, None, None, None, None, None, None, None, None


# ============================================================================
# PUBLIC API
# ============================================================================

def diff_gauss_seidel_solve(coords, index, k_vals, lower_bounds, upper_bounds, 
                            parameters, args,
                            guidance_weight=1.0, alpha=1e-6, n_iterations=100,
                            verbose=False):
    """
    Differentiable wrapper for the Gauss-Seidel constraint solver with CG-based backward pass.
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

# def compute_variable(coords, index):
#     """Compute pairwise distances for given atom indices."""
#     r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
#     r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
#     return r_ij_norm


# def compute_energy(coords, index, k_vals, lower_bounds, upper_bounds):
#     """Compute constraint violation energy."""
#     distances = compute_variable(coords, index)
    
#     energy = torch.zeros_like(distances)
    
#     # Lower bound violations
#     mask_lower = distances < lower_bounds
#     energy[mask_lower] = k_vals[mask_lower] * (lower_bounds[mask_lower] - distances[mask_lower])
    
#     # Upper bound violations  
#     mask_upper = distances > upper_bounds
#     energy[mask_upper] = k_vals[mask_upper] * (distances[mask_upper] - upper_bounds[mask_upper])
    
#     return energy.sum()


# def load_protein_data(protein_id):
#     """Load protein structure and constraints from dataset."""
#     import os
#     from pathlib import Path
    
#     dataset_dir = Path("./dataset")
#     protein_dir = dataset_dir / protein_id
    
#     if not protein_dir.exists():
#         raise ValueError(f"Protein dataset {protein_id} not found")
    
#     # Load CIF file to get coordinates
#     cif_path = protein_dir / f"{protein_id}_model_0.cif"
    
#     # Simple coordinate extraction from CIF
#     from CifFile import ReadCif
#     cf = ReadCif(str(cif_path))
#     block = list(cf.keys())[0]
#     data = cf[block]
    
#     x_coords = [float(x) for x in data["_atom_site.Cartn_x"]]
#     y_coords = [float(y) for y in data["_atom_site.Cartn_y"]]
#     z_coords = [float(z) for z in data["_atom_site.Cartn_z"]]
    
#     coords = torch.tensor([[x, y, z] for x, y, z in zip(x_coords, y_coords, z_coords)], 
#                           dtype=torch.float32)
    
#     # Load constraint data
#     index = torch.load(protein_dir / "index.pth")
#     args = torch.load(protein_dir / "args.pth")
#     com_args = torch.load(protein_dir / "com_args.pth")
    
#     # Extract constraint parameters
#     k_vals, lower_bounds, upper_bounds = args
    
#     # Handle None bounds
#     if lower_bounds is None:
#         lower_bounds = torch.full((index.shape[1],), float('-inf'))
#     if upper_bounds is None:
#         upper_bounds = torch.full((index.shape[1],), float('inf'))
    
#     return coords, index, k_vals, lower_bounds, upper_bounds, args, com_args


# # ============================================================================
# # UNIT TEST
# # ============================================================================

# def test_gradients_with_finite_difference():
#     """
#     Comprehensive gradient verification using finite differences on real protein data.
#     """
#     print("\n" + "=" * 70)
#     print("CG-BASED GRADIENT VERIFICATION WITH PROTEIN DATA")
#     print("=" * 70)
    
#     # Test parameters
#     epsilon = 1e-4  # Finite difference step size
#     tolerance = 1e-2  # Relative error tolerance
    
#     protein_ids = ["T1124"]
    
#     for protein_id in protein_ids:
#         print(f"\n" + "=" * 60)
#         print(f"Testing Protein: {protein_id}")
#         print("=" * 60)
        
#         try:
#             # Load protein data
#             coords, index, k_vals, lower_bounds, upper_bounds, args, com_args = load_protein_data(protein_id)
            
#             potentials = get_potentials()
#             potential = potentials[1]
#             parameters = potential.compute_parameters(1.0 - (49 / 50))
            
#             # Move to GPU if available
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             coords = coords.to(device)
#             index = index.to(device)
#             k_vals = k_vals.to(device)
#             lower_bounds = lower_bounds.to(device)
#             upper_bounds = upper_bounds.to(device)
            
#             with torch.no_grad():
#                 coords_updated, _ = diff_gauss_seidel_solve(
#                     coords, index, k_vals, lower_bounds, upper_bounds,
#                     guidance_weight=1.0, 
#                     parameters=parameters, args=args,
#                     alpha=1e-6, n_iterations=100,  # Use larger alpha for stability
#                     verbose=True
#                 )
                
#             coords_tgt = coords_updated.clone().detach() + 0.5
            
#             def obj_func(x):
#                 """Simple displacement minimization objective."""
#                 x = x.clone().requires_grad_(True)
#                 x_updated, _ = diff_gauss_seidel_solve(
#                     x, index, k_vals, lower_bounds, upper_bounds,
#                     guidance_weight=1.0, 
#                     parameters=parameters, args=args,
#                     alpha=1e-6, n_iterations=100,  # Match alpha with forward pass
#                     verbose=False
#                 )
#                 return torch.sum((x_updated - coords_tgt) ** 2)
            
#             coords_test = coords.clone().requires_grad_(True)
            
#             for niter in range(1000):
                    
#                 loss = obj_func(coords_test)
#                 loss.backward()
#                 grad_analytical = coords_test.grad.clone()
#                 coords_test = coords_test - 0.3 * coords_test.grad
#                 coords_test = coords_test.clone().detach().requires_grad_(True)
#                 print(f"Iteration {niter + 1}: Loss = {loss.item()}, Gradient Norm = {torch.norm(grad_analytical).item()}")
                
#                 if torch.norm(grad_analytical).item() < 1e-3:
#                     print("Converged!")
#                     break
            
#             # # Compute analytical gradient
#             # coords_test = coords.clone().requires_grad_(True)
#             # loss = obj_func(coords_test)
#             # loss.backward()
#             # grad_analytical = coords_test.grad.clone()
            
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
                    
#             #     print("grad_analytical: {:.6f}, grad_numerical: {:.6f}, rel_error: {:.6f}".format(
#             #         grad_analytical_ij, grad_numerical, rel_error if 'rel_error' in locals() else 0.0
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
#         torch.cuda.set_device(0)
#     else:
#         print("CUDA not available, using CPU")
    
#     # Run the gradient verification
#     test_gradients_with_finite_difference()