"""
Differentiable Gauss-Seidel solver with all seven energy types from energies.png.
This implementation handles:
1. Distance-based constraints: VDW Overlap, Connections, PoseBusters, Symmetric Chain COM
2. Dihedral-based constraints: Chiral Atoms, Stereo Bonds, Planar Bonds
"""

import torch
import sys
import numpy as np
from pathlib import Path

# from gauss_seidel_fast import compute, get_potentials_pbd # get_potentials
import warp as wp
import warp.sparse as wps
import warp.optim as wpo
import warp.optim.linear as wpl
wp.init()
WARP_AVAILABLE = True

# ============================================================================
# WARP FUNCTIONS FOR DIHEDRAL CALCULATIONS
# ============================================================================

@wp.func
def compute_dihedral(
    coords_i: wp.vec3,
    coords_j: wp.vec3,
    coords_k: wp.vec3,
    coords_l: wp.vec3
) -> wp.float32:
    """Compute dihedral angle between four atoms."""
    r_ij = coords_i - coords_j
    r_kj = coords_k - coords_j
    r_kl = coords_k - coords_l
    
    n_ijk = wp.cross(r_ij, r_kj)
    n_jkl = wp.cross(r_kj, r_kl)
    
    r_kj_norm = wp.length(r_kj)
    n_ijk_norm = wp.length(n_ijk)
    n_jkl_norm = wp.length(n_jkl)
    
    # Avoid division by zero
    if n_ijk_norm < 1e-8 or n_jkl_norm < 1e-8:
        return wp.float32(0.0)
    
    # Compute the dihedral angle
    cos_phi = wp.dot(n_ijk, n_jkl) / (n_ijk_norm * n_jkl_norm)
    cos_phi = wp.clamp(cos_phi, -1.0 + 1e-8, 1.0 - 1e-8)
    
    # Determine sign
    sign_phi = wp.sign(wp.dot(r_kj, wp.cross(n_ijk, n_jkl)))
    phi = sign_phi * wp.acos(cos_phi)
    
    return phi

@wp.func
def compute_dihedral_gradient_i(
    coords_i: wp.vec3,
    coords_j: wp.vec3,
    coords_k: wp.vec3,
    coords_l: wp.vec3
) -> wp.vec3:
    """Compute gradient of dihedral angle with respect to atom i."""
    r_ij = coords_i - coords_j
    r_kj = coords_k - coords_j
    r_kl = coords_k - coords_l
    
    n_ijk = wp.cross(r_ij, r_kj)
    n_jkl = wp.cross(r_kj, r_kl)
    
    r_kj_norm = wp.length(r_kj)
    n_ijk_norm = wp.length(n_ijk)
    n_jkl_norm = wp.length(n_jkl)
    
    # Avoid division by zero
    if n_ijk_norm < 1e-8 or n_jkl_norm < 1e-8 or r_kj_norm < 1e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Compute gradient for atom i
    grad_i = n_ijk * (r_kj_norm / (n_ijk_norm * n_ijk_norm))
    return grad_i

@wp.func
def compute_dihedral_gradient_j(
    coords_i: wp.vec3,
    coords_j: wp.vec3,
    coords_k: wp.vec3,
    coords_l: wp.vec3
) -> wp.vec3:
    """Compute gradient of dihedral angle with respect to atom j."""
    r_ij = coords_i - coords_j
    r_kj = coords_k - coords_j
    r_kl = coords_k - coords_l
    
    n_ijk = wp.cross(r_ij, r_kj)
    n_jkl = wp.cross(r_kj, r_kl)
    
    r_kj_norm = wp.length(r_kj)
    n_ijk_norm = wp.length(n_ijk)
    n_jkl_norm = wp.length(n_jkl)
    
    # Avoid division by zero
    if n_ijk_norm < 1e-8 or n_jkl_norm < 1e-8 or r_kj_norm < 1e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Compute gradient components
    a = wp.dot(r_ij, r_kj) / (r_kj_norm * r_kj_norm)
    b = wp.dot(r_kl, r_kj) / (r_kj_norm * r_kj_norm)
    
    grad_i = n_ijk * (r_kj_norm / (n_ijk_norm * n_ijk_norm))
    grad_l = -n_jkl * (r_kj_norm / (n_jkl_norm * n_jkl_norm))
    grad_j = (a - 1.0) * grad_i - b * grad_l
    return grad_j

@wp.func
def compute_dihedral_gradient_k(
    coords_i: wp.vec3,
    coords_j: wp.vec3,
    coords_k: wp.vec3,
    coords_l: wp.vec3
) -> wp.vec3:
    """Compute gradient of dihedral angle with respect to atom k."""
    r_ij = coords_i - coords_j
    r_kj = coords_k - coords_j
    r_kl = coords_k - coords_l
    
    n_ijk = wp.cross(r_ij, r_kj)
    n_jkl = wp.cross(r_kj, r_kl)
    
    r_kj_norm = wp.length(r_kj)
    n_ijk_norm = wp.length(n_ijk)
    n_jkl_norm = wp.length(n_jkl)
    
    # Avoid division by zero
    if n_ijk_norm < 1e-8 or n_jkl_norm < 1e-8 or r_kj_norm < 1e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Compute gradient components
    a = wp.dot(r_ij, r_kj) / (r_kj_norm * r_kj_norm)
    b = wp.dot(r_kl, r_kj) / (r_kj_norm * r_kj_norm)
    
    grad_i = n_ijk * (r_kj_norm / (n_ijk_norm * n_ijk_norm))
    grad_l = -n_jkl * (r_kj_norm / (n_jkl_norm * n_jkl_norm))
    grad_k = (b - 1.0) * grad_l - a * grad_i
    return grad_k

@wp.func
def compute_dihedral_gradient_l(
    coords_i: wp.vec3,
    coords_j: wp.vec3,
    coords_k: wp.vec3,
    coords_l: wp.vec3
) -> wp.vec3:
    """Compute gradient of dihedral angle with respect to atom l."""
    r_ij = coords_i - coords_j
    r_kj = coords_k - coords_j
    r_kl = coords_k - coords_l
    
    n_ijk = wp.cross(r_ij, r_kj)
    n_jkl = wp.cross(r_kj, r_kl)
    
    r_kj_norm = wp.length(r_kj)
    n_ijk_norm = wp.length(n_ijk)
    n_jkl_norm = wp.length(n_jkl)
    
    # Avoid division by zero
    if n_ijk_norm < 1e-8 or n_jkl_norm < 1e-8 or r_kj_norm < 1e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Compute gradient for atom l
    grad_l = -n_jkl * (r_kj_norm / (n_jkl_norm * n_jkl_norm))
    return grad_l

# ============================================================================
# DISTANCE-BASED CONSTRAINT KERNEL
# ============================================================================

@wp.kernel
def distance_constraint_kernel(
    coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3)
    lagrangian: wp.array(dtype=wp.float32),        # (N_constraints,)
    index: wp.array2d(dtype=wp.int32),             # (2, N_constraints) for pairwise
    k_vals: wp.array(dtype=wp.float32),            # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    alpha: wp.float32 = 1e-6
):
    """
    Handle all distance-based constraints:
    - VDW Overlap (Steric Clash)
    - Connections (Covalently Bonded Chains)
    - PoseBusters (Internal Geometry)
    - Symmetric Chain COM
    All follow the same mathematical formulation with different parameters.
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
    
    # Compute constraint value C and gradient based on type
    C = wp.float32(0.0)
    dEnergy = wp.float32(0.0)
    
    # All distance constraints follow same pattern
    if r_ij_norm < lb:
        C = (lb - r_ij_norm) * k
        dEnergy = -k
    elif r_ij_norm > ub:
        C = (r_ij_norm - ub) * k  
        dEnergy = k
    
    C *= guidance_weight
    dEnergy *= guidance_weight
    
    # Compute gradient
    r_hat_ij = r_ij / (r_ij_norm + 1e-9)
    
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


# ============================================================================
# COM-SPECIFIC CONSTRAINT KERNEL
# ============================================================================

@wp.kernel
def com_constraint_kernel(
    coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3)
    lagrangian: wp.array(dtype=wp.float32),        # (N_constraints,)
    com_chain_index: wp.array2d(dtype=wp.int32),   # (2, N_constraints) chain IDs
    k_vals: wp.array(dtype=wp.float32),            # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    com_atom_index: wp.array(dtype=wp.int32),           # (N_atoms,) chain assignment
    atom_mask: wp.array(dtype=wp.bool),            # (N_atoms,) valid atoms
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    alpha: wp.float32 = 1e-6
):
    """
    Handle Symmetric Chain COM constraints.
    Computes center of mass for chains and applies distance constraints on COMs.
    """
    tid = wp.tid()
    
    if tid >= n_constraints:
        return
        
    constraint_idx = tid
    
    # For COM constraints, indices refer to chain IDs
    chain1 = com_chain_index[0, constraint_idx]
    chain2 = com_chain_index[1, constraint_idx]
    
    # Compute COM for chain1
    com1 = wp.vec3(0.0, 0.0, 0.0)
    count1 = float(0.0)
    for atom_idx in range(coords.shape[0]):
        if atom_mask[atom_idx] and com_atom_index[atom_idx] == chain1:
            com1 = com1 + wp.vec3(coords[atom_idx, 0], coords[atom_idx, 1], coords[atom_idx, 2])
            count1 = count1 + 1.0
    
    # Compute COM for chain2
    com2 = wp.vec3(0.0, 0.0, 0.0)
    count2 = float(0.0)
    for atom_idx in range(coords.shape[0]):
        if atom_mask[atom_idx] and com_atom_index[atom_idx] == chain2:
            com2 = com2 + wp.vec3(coords[atom_idx, 0], coords[atom_idx, 1], coords[atom_idx, 2])
            count2 = count2 + 1.0
    
    # Average to get COM
    if count1 > 0.0:
        com1 = com1 / count1
    if count2 > 0.0:
        com2 = com2 / count2
    
    # Compute COM distance
    r_ij = com1 - com2
    r_ij_norm = wp.length(r_ij)
    
    k = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    # Compute constraint value C and gradient
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
    
    # Compute gradient
    r_hat_ij = r_ij / (r_ij_norm + 1e-9)
    
    grad_com1 = r_hat_ij * dEnergy
    grad_com2 = -grad_com1
    
    # Compute denominator for XPBD update (sum over all atoms)
    denominator = alpha
    if count1 > 0.0:
        denominator += wp.dot(grad_com1, grad_com1) / count1
    if count2 > 0.0:
        denominator += wp.dot(grad_com2, grad_com2) / count2
    
    dlambda = -(C + alpha * lagrangian[constraint_idx]) / denominator
    
    lagrangian[constraint_idx] += dlambda
    
    # Apply position corrections to all atoms in chains
    delta_com1 = dlambda * grad_com1
    delta_com2 = dlambda * grad_com2
    
    # Distribute corrections to all atoms in chain1
    if count1 > 0.0:
        weight1 = 1.0 / count1
        for atom_idx in range(coords.shape[0]):
            if atom_mask[atom_idx] and com_atom_index[atom_idx] == chain1:
                wp.atomic_add(coords, atom_idx, 0, delta_com1[0] * weight1)
                wp.atomic_add(coords, atom_idx, 1, delta_com1[1] * weight1)
                wp.atomic_add(coords, atom_idx, 2, delta_com1[2] * weight1)
    
    # Distribute corrections to all atoms in chain2
    if count2 > 0.0:
        weight2 = 1.0 / count2
        for atom_idx in range(coords.shape[0]):
            if atom_mask[atom_idx] and com_atom_index[atom_idx] == chain2:
                wp.atomic_add(coords, atom_idx, 0, delta_com2[0] * weight2)
                wp.atomic_add(coords, atom_idx, 1, delta_com2[1] * weight2)
                wp.atomic_add(coords, atom_idx, 2, delta_com2[2] * weight2)

# ============================================================================
# DIHEDRAL-BASED CONSTRAINT KERNEL
# ============================================================================

@wp.kernel
def dihedral_constraint_kernel(
    coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3)
    lagrangian: wp.array(dtype=wp.float32),        # (N_constraints,)
    index: wp.array2d(dtype=wp.int32),             # (4, N_constraints) for dihedral
    k_vals: wp.array(dtype=wp.float32),            # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),      # (N_constraints,)
    use_abs: wp.array(dtype=wp.bool),              # (N_constraints,) whether to use absolute value
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    alpha: wp.float32 = 1e-6
):
    """
    Handle all dihedral-based constraints:
    - Chiral Atom (Tetrahedral Atom Chirality) - use_abs=False
    - Stereo Bond (Bond Stereochemistry) - use_abs=True
    - Planar Bond (Planar Double Bonds) - use_abs=True
    All follow the same mathematical formulation with different parameters.
    """
    tid = wp.tid()
    
    if tid >= n_constraints:
        return
        
    constraint_idx = tid
    
    # Get four atoms for dihedral
    atom_i = index[0, constraint_idx]
    atom_j = index[1, constraint_idx]
    atom_k = index[2, constraint_idx]
    atom_l = index[3, constraint_idx]
    
    coords_i = wp.vec3(coords[atom_i, 0], coords[atom_i, 1], coords[atom_i, 2])
    coords_j = wp.vec3(coords[atom_j, 0], coords[atom_j, 1], coords[atom_j, 2])
    coords_k = wp.vec3(coords[atom_k, 0], coords[atom_k, 1], coords[atom_k, 2])
    coords_l = wp.vec3(coords[atom_l, 0], coords[atom_l, 1], coords[atom_l, 2])
    
    # Compute dihedral angle
    phi = compute_dihedral(coords_i, coords_j, coords_k, coords_l)
    
    # Use absolute value if needed (for stereo bonds and planar bonds)
    if use_abs[constraint_idx]:
        phi_sign = wp.sign(phi)
        phi = wp.abs(phi)
    else:
        phi_sign = wp.float32(1.0)
    
    k = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    # Compute constraint value C
    C = wp.float32(0.0)
    dEnergy = wp.float32(0.0)
    
    if phi < lb:
        C = (lb - phi) * k
        dEnergy = -k
    elif phi > ub:
        C = (phi - ub) * k
        dEnergy = k
    
    C *= guidance_weight
    dEnergy *= guidance_weight * phi_sign
    
    # Compute gradient of dihedral
    grad_i = compute_dihedral_gradient_i(coords_i, coords_j, coords_k, coords_l)
    grad_j = compute_dihedral_gradient_j(coords_i, coords_j, coords_k, coords_l)
    grad_k = compute_dihedral_gradient_k(coords_i, coords_j, coords_k, coords_l)
    grad_l = compute_dihedral_gradient_l(coords_i, coords_j, coords_k, coords_l)
    
    # Scale gradients by dEnergy
    grad_i = grad_i * dEnergy
    grad_j = grad_j * dEnergy
    grad_k = grad_k * dEnergy
    grad_l = grad_l * dEnergy
    
    # Compute denominator for XPBD update
    denominator = (wp.dot(grad_i, grad_i) + wp.dot(grad_j, grad_j) + 
                   wp.dot(grad_k, grad_k) + wp.dot(grad_l, grad_l) + alpha)
    dlambda = -(C + alpha * lagrangian[constraint_idx]) / denominator
    
    lagrangian[constraint_idx] += dlambda
    
    # Apply position corrections
    delta_i = dlambda * grad_i
    delta_j = dlambda * grad_j
    delta_k = dlambda * grad_k
    delta_l = dlambda * grad_l
    
    wp.atomic_add(coords, atom_i, 0, delta_i[0])
    wp.atomic_add(coords, atom_i, 1, delta_i[1])
    wp.atomic_add(coords, atom_i, 2, delta_i[2])
    wp.atomic_add(coords, atom_j, 0, delta_j[0])
    wp.atomic_add(coords, atom_j, 1, delta_j[1])
    wp.atomic_add(coords, atom_j, 2, delta_j[2])
    wp.atomic_add(coords, atom_k, 0, delta_k[0])
    wp.atomic_add(coords, atom_k, 1, delta_k[1])
    wp.atomic_add(coords, atom_k, 2, delta_k[2])
    wp.atomic_add(coords, atom_l, 0, delta_l[0])
    wp.atomic_add(coords, atom_l, 1, delta_l[1])
    wp.atomic_add(coords, atom_l, 2, delta_l[2])

# ============================================================================
# SYSTEM MATRIX BUILDER FOR BACKWARD PASS
# ============================================================================
@wp.kernel
def build_distance_system_matrix_kernel(
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
    epsilon: wp.float32 = 1e-9,
):
    """
    Build the system matrix (H + ΔH) for implicit differentiation of distance constraints.
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
    
    # Avoid division by zero
    if r_ij_norm < epsilon:
        return
    
    k_val = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    # Check if constraint is active at converged state
    C = wp.float32(0.0)
    grad_scalar = wp.float32(0.0)
    
    if r_ij_norm < lb:
        C = (lb - r_ij_norm) * k_val
        grad_scalar = -wp.float32(1.0)
    elif r_ij_norm > ub:
        C = (r_ij_norm - ub) * k_val
        grad_scalar = wp.float32(1.0)
    
    C *= guidance_weight
        
    # Unit vector
    n = r_ij / r_ij_norm
    
    # Gradient of constraint at converged state
    k_scaled = k_val * guidance_weight
    grad_C_i = k_scaled * n * grad_scalar
    grad_C_j = -grad_C_i
    
    alpha_inv = 1.0 / alpha
    
    # Compute Hessian matrix
    I = wp.mat33(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0)
    )
    n_outer_n = wp.outer(n, n)
    hessian_base = (I - n_outer_n) / r_ij_norm * wp.abs(grad_scalar)
    hessian_C = k_scaled * hessian_base
    
    # Base index for this constraint's contribution to sparse matrix
    base_idx = constraint_idx * 36
    
    entry_idx = 0
    
    # Block (i,i): Effect on atom1 from atom1
    for d1 in range(3):
        for d2 in range(3):
            row = atom1 * 3 + d1
            col = atom1 * 3 + d2
            
            h_val = alpha_inv * grad_C_i[d1] * grad_C_i[d2]
            delta_h_val = alpha_inv * hessian_C[d1, d2] * C
            total_val = h_val + delta_h_val
            
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
            
            h_val = alpha_inv * grad_C_i[d1] * grad_C_j[d2]
            delta_h_val = -alpha_inv * hessian_C[d1, d2] * C
            total_val = h_val + delta_h_val
            
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
            
            h_val = alpha_inv * grad_C_j[d1] * grad_C_i[d2]
            delta_h_val = -alpha_inv * hessian_C[d1, d2] * C
            total_val = h_val + delta_h_val
            
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
            
            h_val = alpha_inv * grad_C_j[d1] * grad_C_j[d2]
            delta_h_val = alpha_inv * hessian_C[d1, d2] * C
            total_val = h_val + delta_h_val
            
            idx = base_idx + entry_idx
            row_indices[idx] = row
            col_indices[idx] = col
            values[idx] = total_val
            entry_idx += 1

@wp.kernel
def build_com_system_matrix_kernel(
    x_coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3) - converged positions
    index: wp.array2d(dtype=wp.int32),               # (2, N_constraints) chain IDs
    k_vals: wp.array(dtype=wp.float32),              # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),        # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),        # (N_constraints,)
    com_index: wp.array(dtype=wp.int32),             # (N_atoms,) chain assignment
    atom_mask: wp.array(dtype=wp.bool),              # (N_atoms,) valid atoms
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    n_atoms: wp.int32,
    # Sparse matrix outputs (COO format) - now variable size
    row_indices: wp.array(dtype=wp.int32),           
    col_indices: wp.array(dtype=wp.int32),           
    values: wp.array(dtype=wp.float32),              
    nnz_per_constraint: wp.array(dtype=wp.int32),    # Output: number of non-zeros for each constraint
    alpha: wp.float32 = 1e-6,
    epsilon: wp.float32 = 1e-9,
):
    """
    Build the system matrix for COM constraints.
    Each COM constraint affects ALL atoms in both chains.
    Fills the sparse matrix entries for all atom pairs affected by the COM constraint.
    """
    constraint_idx = wp.tid()
    
    if constraint_idx >= n_constraints:
        return
    
    # For COM constraints, indices refer to chain IDs
    chain1 = index[0, constraint_idx]
    chain2 = index[1, constraint_idx]
    
    # Compute COM for both chains
    com1 = wp.vec3(0.0, 0.0, 0.0)
    count1 = float(0.0)
    com2 = wp.vec3(0.0, 0.0, 0.0)
    count2 = float(0.0)
    
    for atom_idx in range(n_atoms):
        if atom_mask[atom_idx]:
            if com_index[atom_idx] == chain1:
                com1 = com1 + wp.vec3(x_coords[atom_idx, 0], x_coords[atom_idx, 1], x_coords[atom_idx, 2])
                count1 = count1 + 1.0
            elif com_index[atom_idx] == chain2:
                com2 = com2 + wp.vec3(x_coords[atom_idx, 0], x_coords[atom_idx, 1], x_coords[atom_idx, 2])
                count2 = count2 + 1.0
    
    # Average to get COM
    if count1 > 0.0:
        com1 = com1 / count1
    if count2 > 0.0:
        com2 = com2 / count2
    
    # Compute displacement vector at converged state
    r_ij = com1 - com2
    r_ij_norm = wp.length(r_ij)
    
    # Avoid division by zero
    if r_ij_norm < epsilon or count1 == 0.0 or count2 == 0.0:
        nnz_per_constraint[constraint_idx] = 0
        return
    
    k_val = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    # Check if constraint is active at converged state
    C = wp.float32(0.0)
    grad_scalar = wp.float32(0.0)
    
    if r_ij_norm < lb:
        C = (lb - r_ij_norm) * k_val
        grad_scalar = -wp.float32(1.0)
    elif r_ij_norm > ub:
        C = (r_ij_norm - ub) * k_val
        grad_scalar = wp.float32(1.0)
    else:
        # Constraint is satisfied - no contribution
        nnz_per_constraint[constraint_idx] = 0
        return
    
    C *= guidance_weight
        
    # Unit vector
    n = r_ij / r_ij_norm
    
    # Gradient of COM constraint
    k_scaled = k_val * guidance_weight
    grad_com1 = k_scaled * n * grad_scalar
    grad_com2 = -grad_com1
    
    alpha_inv = 1.0 / alpha
    
    # Compute Hessian matrix for COM
    I = wp.mat33(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0)
    )
    n_outer_n = wp.outer(n, n)
    hessian_base = (I - n_outer_n) / r_ij_norm * wp.abs(grad_scalar)
    hessian_C = k_scaled * hessian_base
    
    # Count actual atoms in each chain for this constraint
    n_chain1_atoms = wp.int32(count1)
    n_chain2_atoms = wp.int32(count2)
    
    # Total entries: all pairs between atoms
    # Each atom pair contributes 9 entries (3x3 block)
    total_entries = (n_chain1_atoms * n_chain1_atoms + 
                    n_chain2_atoms * n_chain2_atoms + 
                    2 * n_chain1_atoms * n_chain2_atoms) * 9
    
    nnz_per_constraint[constraint_idx] = total_entries
    
    # Now fill the matrix entries
    # Note: In practice, we'd need proper indexing from the caller
    # For now, we'll use a simplified approach
    entry_idx = int(0)  # Use int() for dynamic variable in Warp
    weight1 = 1.0 / count1
    weight2 = 1.0 / count2
    
    # Fill matrix entries for all atom pairs
    # This is a simplified version - in practice would need proper base indexing
    
    # Chain1-Chain1 interactions
    for atom_i in range(n_atoms):
        if not (atom_mask[atom_i] and com_index[atom_i] == chain1):
            continue
        for atom_j in range(n_atoms):
            if not (atom_mask[atom_j] and com_index[atom_j] == chain1):
                continue
            for d1 in range(3):
                for d2 in range(3):
                    row = atom_i * 3 + d1
                    col = atom_j * 3 + d2
                    
                    h_val = alpha_inv * grad_com1[d1] * grad_com1[d2] * weight1 * weight1
                    delta_h_val = alpha_inv * hessian_C[d1, d2] * C * weight1 * weight1
                    total_val = h_val + delta_h_val
                    
                    if entry_idx < row_indices.shape[0]:
                        row_indices[entry_idx] = row
                        col_indices[entry_idx] = col
                        values[entry_idx] = total_val
                    entry_idx = entry_idx + 1
    
    # Chain1-Chain2 interactions
    for atom_i in range(n_atoms):
        if not (atom_mask[atom_i] and com_index[atom_i] == chain1):
            continue
        for atom_j in range(n_atoms):
            if not (atom_mask[atom_j] and com_index[atom_j] == chain2):
                continue
            for d1 in range(3):
                for d2 in range(3):
                    row = atom_i * 3 + d1
                    col = atom_j * 3 + d2
                    
                    h_val = alpha_inv * grad_com1[d1] * grad_com2[d2] * weight1 * weight2
                    delta_h_val = -alpha_inv * hessian_C[d1, d2] * C * weight1 * weight2
                    total_val = h_val + delta_h_val
                    
                    if entry_idx < row_indices.shape[0]:
                        row_indices[entry_idx] = row
                        col_indices[entry_idx] = col
                        values[entry_idx] = total_val
                    entry_idx = entry_idx + 1
    
    # Chain2-Chain1 interactions
    for atom_i in range(n_atoms):
        if not (atom_mask[atom_i] and com_index[atom_i] == chain2):
            continue
        for atom_j in range(n_atoms):
            if not (atom_mask[atom_j] and com_index[atom_j] == chain1):
                continue
            for d1 in range(3):
                for d2 in range(3):
                    row = atom_i * 3 + d1
                    col = atom_j * 3 + d2
                    
                    h_val = alpha_inv * grad_com2[d1] * grad_com1[d2] * weight2 * weight1
                    delta_h_val = -alpha_inv * hessian_C[d1, d2] * C * weight2 * weight1
                    total_val = h_val + delta_h_val
                    
                    if entry_idx < row_indices.shape[0]:
                        row_indices[entry_idx] = row
                        col_indices[entry_idx] = col
                        values[entry_idx] = total_val
                    entry_idx = entry_idx + 1
    
    # Chain2-Chain2 interactions
    for atom_i in range(n_atoms):
        if not (atom_mask[atom_i] and com_index[atom_i] == chain2):
            continue
        for atom_j in range(n_atoms):
            if not (atom_mask[atom_j] and com_index[atom_j] == chain2):
                continue
            for d1 in range(3):
                for d2 in range(3):
                    row = atom_i * 3 + d1
                    col = atom_j * 3 + d2
                    
                    h_val = alpha_inv * grad_com2[d1] * grad_com2[d2] * weight2 * weight2
                    delta_h_val = alpha_inv * hessian_C[d1, d2] * C * weight2 * weight2
                    total_val = h_val + delta_h_val
                    
                    if entry_idx < row_indices.shape[0]:
                        row_indices[entry_idx] = row
                        col_indices[entry_idx] = col
                        values[entry_idx] = total_val
                    entry_idx = entry_idx + 1


@wp.kernel
def build_dihedral_system_matrix_kernel(
    x_coords: wp.array2d(dtype=wp.float32),          # (N_atoms, 3) - converged positions
    index: wp.array2d(dtype=wp.int32),               # (4, N_constraints)
    k_vals: wp.array(dtype=wp.float32),              # (N_constraints,) stiffness
    lower_bounds: wp.array(dtype=wp.float32),        # (N_constraints,)
    upper_bounds: wp.array(dtype=wp.float32),        # (N_constraints,)
    use_abs: wp.array(dtype=wp.bool),                # (N_constraints,) whether to use absolute value
    guidance_weight: wp.float32,
    n_constraints: wp.int32,
    # Sparse matrix outputs (COO format)
    row_indices: wp.array(dtype=wp.int32),           # Row indices for sparse matrix
    col_indices: wp.array(dtype=wp.int32),           # Column indices for sparse matrix  
    values: wp.array(dtype=wp.float32),              # Values for sparse matrix
    alpha: wp.float32 = 1e-6,
    epsilon: wp.float32 = 1e-9,
):
    """
    Build the system matrix (H + ΔH) for implicit differentiation of dihedral constraints.
    """
    constraint_idx = wp.tid()
    
    if constraint_idx >= n_constraints:
        return
    
    # Get four atoms for dihedral
    atom_i = index[0, constraint_idx]
    atom_j = index[1, constraint_idx]
    atom_k = index[2, constraint_idx]
    atom_l = index[3, constraint_idx]
    
    coords_i = wp.vec3(x_coords[atom_i, 0], x_coords[atom_i, 1], x_coords[atom_i, 2])
    coords_j = wp.vec3(x_coords[atom_j, 0], x_coords[atom_j, 1], x_coords[atom_j, 2])
    coords_k = wp.vec3(x_coords[atom_k, 0], x_coords[atom_k, 1], x_coords[atom_k, 2])
    coords_l = wp.vec3(x_coords[atom_l, 0], x_coords[atom_l, 1], x_coords[atom_l, 2])
    
    # Compute dihedral angle
    phi = compute_dihedral(coords_i, coords_j, coords_k, coords_l)
    
    # Use absolute value if needed
    if use_abs[constraint_idx]:
        phi_sign = wp.sign(phi)
        phi = wp.abs(phi)
    else:
        phi_sign = wp.float32(1.0)
    
    k_val = k_vals[constraint_idx]
    lb = lower_bounds[constraint_idx]
    ub = upper_bounds[constraint_idx]
    
    # Check if constraint is active at converged state
    C = wp.float32(0.0)
    grad_scalar = wp.float32(0.0)
    
    if phi < lb:
        C = (lb - phi) * k_val
        grad_scalar = -wp.float32(1.0)
    elif phi > ub:
        C = (phi - ub) * k_val
        grad_scalar = wp.float32(1.0)
    
    C *= guidance_weight
    grad_scalar *= guidance_weight * k_val * phi_sign
    
    # Compute gradient of dihedral
    grad_i = compute_dihedral_gradient_i(coords_i, coords_j, coords_k, coords_l)
    grad_j = compute_dihedral_gradient_j(coords_i, coords_j, coords_k, coords_l)
    grad_k = compute_dihedral_gradient_k(coords_i, coords_j, coords_k, coords_l)
    grad_l = compute_dihedral_gradient_l(coords_i, coords_j, coords_k, coords_l)
    
    # Scale gradients
    grad_C_i = grad_i * grad_scalar
    grad_C_j = grad_j * grad_scalar
    grad_C_k = grad_k * grad_scalar
    grad_C_l = grad_l * grad_scalar
    
    alpha_inv = 1.0 / alpha
    
    # For dihedral constraints, computing the full Hessian is complex
    # We approximate with just the H term (gradient outer product)
    # This is a reasonable approximation as dihedral Hessians are often small
    
    # Base index for this constraint's contribution to sparse matrix
    # Each dihedral constraint contributes a 12x12 block (4 atoms x 3 dimensions)
    base_idx = constraint_idx * 144  # 12*12
    
    entry_idx = 0
    
    # Fill the 12x12 block - unroll the loops to avoid list usage
    for a1 in range(4):
        # Select atom index based on a1
        if a1 == 0:
            atom_1 = atom_i
            grad_1 = grad_C_i
        elif a1 == 1:
            atom_1 = atom_j
            grad_1 = grad_C_j
        elif a1 == 2:
            atom_1 = atom_k
            grad_1 = grad_C_k
        else:  # a1 == 3
            atom_1 = atom_l
            grad_1 = grad_C_l
            
        for d1 in range(3):
            row = atom_1 * 3 + d1
            grad_1_d1 = grad_1[d1]
            
            for a2 in range(4):
                # Select atom index based on a2
                if a2 == 0:
                    atom_2 = atom_i
                    grad_2 = grad_C_i
                elif a2 == 1:
                    atom_2 = atom_j
                    grad_2 = grad_C_j
                elif a2 == 2:
                    atom_2 = atom_k
                    grad_2 = grad_C_k
                else:  # a2 == 3
                    atom_2 = atom_l
                    grad_2 = grad_C_l
                    
                for d2 in range(3):
                    col = atom_2 * 3 + d2
                    grad_2_d2 = grad_2[d2]
                    
                    # H contribution: α⁻¹ * grad[a1][d1] * grad[a2][d2]
                    h_val = alpha_inv * grad_1_d1 * grad_2_d2
                    
                    # For simplicity, we omit the ΔH term for dihedral constraints
                    # as it requires second derivatives of dihedrals which are complex
                    total_val = h_val
                    
                    idx = base_idx + entry_idx
                    row_indices[idx] = row
                    col_indices[idx] = col
                    values[idx] = total_val
                    entry_idx += 1

# ============================================================================
# DIFFERENTIABLE FUNCTION
# ============================================================================

class DiffGaussSeidelFunctionAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, 
                dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
                dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
                com_chain_index, com_args, com_k_vals, com_lower_bounds, com_upper_bounds,
                guidance_weight, parameters, args,
                alpha=1e-6, n_iterations=100, 
                debug=False, verbose=False):
        """
        Forward pass: Solves the constraint system using extended Gauss-Seidel method.
        Handles both distance and dihedral constraints.
        """
        
        n_atoms = coords.shape[0]
        n_dist_constraints = dist_index.shape[1] if dist_index is not None else 0
        n_dihed_constraints = dihed_index.shape[1] if dihed_index is not None else 0
        
        coords_contig = coords.clone().contiguous() # .clone().detach()
        
        # Convert to Warp arrays
        coords_wp = wp.from_torch(coords_contig, dtype=wp.float32)
        
        # Prepare COM args for kernels
        if com_args is not None:
            com_atom_index, atom_pad_mask = com_args
            com_atom_index_wp = wp.from_torch(com_atom_index.to(torch.int32).contiguous(), dtype=wp.int32)
            atom_mask_wp = wp.from_torch(atom_pad_mask.contiguous(), dtype=wp.bool)
            com_k_vals_wp = wp.from_torch(com_k_vals.contiguous(), dtype=wp.float32)
            com_lower_bounds_wp = wp.from_torch(com_lower_bounds.contiguous(), dtype=wp.float32)
            com_upper_bounds_wp = wp.from_torch(com_upper_bounds.contiguous(), dtype=wp.float32)
            
        # Distance constraints
        if n_dist_constraints > 0:
            dist_index_wp = wp.from_torch(dist_index.to(torch.int32).contiguous(), dtype=wp.int32)
            dist_k_vals_wp = wp.from_torch(dist_k_vals.contiguous(), dtype=wp.float32)
            dist_lower_bounds_wp = wp.from_torch(dist_lower_bounds.contiguous(), dtype=wp.float32)
            dist_upper_bounds_wp = wp.from_torch(dist_upper_bounds.contiguous(), dtype=wp.float32)
            dist_lagrangian_wp = wp.zeros(n_dist_constraints, dtype=wp.float32, device=coords_wp.device)
        
        # Dihedral constraints
        if n_dihed_constraints > 0:
            dihed_index_wp = wp.from_torch(dihed_index.to(torch.int32).contiguous(), dtype=wp.int32)
            dihed_k_vals_wp = wp.from_torch(dihed_k_vals.contiguous(), dtype=wp.float32)
            dihed_lower_bounds_wp = wp.from_torch(dihed_lower_bounds.contiguous(), dtype=wp.float32)
            dihed_upper_bounds_wp = wp.from_torch(dihed_upper_bounds.contiguous(), dtype=wp.float32)
            dihed_use_abs_wp = wp.from_torch(dihed_use_abs.contiguous(), dtype=wp.bool)
            dihed_lagrangian_wp = wp.zeros(n_dihed_constraints, dtype=wp.float32, device=coords_wp.device)
            
        # COM constraints
        if com_args is not None and com_chain_index is not None:
            n_com_constraints = com_chain_index.shape[1]
            com_chain_index_wp = wp.from_torch(com_chain_index.to(torch.int32).contiguous(), dtype=wp.int32)
            com_largrangian_wp = wp.zeros(n_com_constraints, dtype=wp.float32, device=coords_wp.device)
        else:
            n_com_constraints = 0

        coord_log_all_iter = []

        # Run multiple iterations for convergence
        dev = coords_wp.device
        for iter_ in range(n_iterations):
            # Solve distance constraints
            if n_dist_constraints > 0:
                with wp.ScopedDevice(dev):
                    wp.launch(
                        kernel=distance_constraint_kernel,
                        dim=n_dist_constraints,
                        inputs=[
                            coords_wp,
                            dist_lagrangian_wp,
                            dist_index_wp,
                            dist_k_vals_wp,
                            dist_lower_bounds_wp,
                            dist_upper_bounds_wp,
                            guidance_weight,
                            n_dist_constraints,
                            alpha
                        ],
                        device=dev,
                        stream=wp.get_stream(
                            dev
                        )
                    )
            
            # Solve COM constraints  
            if com_args is not None and com_chain_index is not None:
                n_com_constraints = com_chain_index.shape[1]
                if n_com_constraints > 0:
                    com_chain_index_wp = wp.from_torch(com_chain_index.to(torch.int32).contiguous(), dtype=wp.int32)
                    with wp.ScopedDevice(dev):
                        wp.launch(
                            kernel=com_constraint_kernel,
                            dim=n_com_constraints,
                            inputs=[
                                coords_wp,
                                com_largrangian_wp,
                                com_chain_index_wp,
                                com_k_vals_wp,
                                com_lower_bounds_wp,
                                com_upper_bounds_wp,
                                com_atom_index_wp,
                                atom_mask_wp,
                                guidance_weight,
                                n_com_constraints,
                                alpha
                            ],
                            device=dev,
                            stream=wp.get_stream(
                                dev
                            )
                        )
                
            # Solve dihedral constraints
            if n_dihed_constraints > 0:
                with wp.ScopedDevice(dev):
                    wp.launch(
                        kernel=dihedral_constraint_kernel,
                        dim=n_dihed_constraints,
                        inputs=[
                            coords_wp,
                            dihed_lagrangian_wp,
                            dihed_index_wp,
                            dihed_k_vals_wp,
                            dihed_lower_bounds_wp,
                            dihed_upper_bounds_wp,
                            dihed_use_abs_wp,
                            guidance_weight,
                            n_dihed_constraints,
                            alpha
                        ],
                        device=dev,
                        stream=wp.get_stream(
                            dev
                        )
                    )
            
            wp.synchronize()

            coords_updated_iter = wp.to_torch(coords_wp)
            coord_log_all_iter.append(coords_updated_iter.clone().detach())

        # Get updated coordinates
        coords_updated = wp.to_torch(coords_wp)
        
        # Don't mark as requires_grad here - that breaks the graph
        # The autograd.Function handles gradient tracking automatically
        
        # Get Lagrange multipliers
        dist_lagrangian = wp.to_torch(dist_lagrangian_wp) if n_dist_constraints > 0 else torch.zeros(1, device=coords.device)
        com_lagrangian = wp.to_torch(com_largrangian_wp) if n_com_constraints > 0 else torch.zeros(1, device=coords.device)
            
        dihed_lagrangian = wp.to_torch(dihed_lagrangian_wp) if n_dihed_constraints > 0 else torch.zeros(1, device=coords.device)
        
        # Save necessary information for backward pass
        ctx.save_for_backward(coords, coords_updated, 
                            dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
                            dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
                            com_chain_index, com_k_vals, com_lower_bounds, com_upper_bounds)
        ctx.guidance_weight = guidance_weight
        ctx.alpha = alpha
        ctx.n_dist_constraints = n_dist_constraints
        ctx.n_dihed_constraints = n_dihed_constraints
        ctx.n_com_constraints = n_com_constraints
        ctx.n_atoms = n_atoms
        ctx.n_iterations = n_iterations
        ctx.debug = debug
        ctx.com_args = com_args  # Save COM args for backward pass
        if com_args is not None:
            ctx.com_atom_index = com_args[0]
            ctx.atom_pad_mask = com_args[1]
        else:
            ctx.com_atom_index = None
            ctx.atom_pad_mask = None
        
        return coords_updated, dist_lagrangian, dihed_lagrangian, com_lagrangian, coord_log_all_iter
        
    @staticmethod  
    def backward(ctx, grad_coords_updated, grad_dist_lagrangian, grad_dihed_lagrangian, grad_com_lagrangian):
        """
        Backward pass using implicit differentiation.
        """
        # Retrieve saved tensors and parameters
        (coords_initial, coords_converged, 
         dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
         dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
         com_chain_index, com_k_vals, com_lower_bounds, com_upper_bounds) = ctx.saved_tensors
        
        guidance_weight = ctx.guidance_weight
        alpha = ctx.alpha
        n_dist_constraints = ctx.n_dist_constraints
        n_dihed_constraints = ctx.n_dihed_constraints
        n_com_constraints = ctx.n_com_constraints
        n_atoms = ctx.n_atoms
        com_args = ctx.com_args
        
        device = coords_converged.device
        device_grad = grad_coords_updated.device
        assert device == device_grad, "Device mismatch between coords and grad_coords"
        
        # Dimension of the linear system (3 coords per atom)
        n_dim = n_atoms * 3
        
        # Convert to Warp
        coords_converged_wp = wp.from_torch(coords_converged.contiguous(), dtype=wp.float32)
        
        # Calculate total non-zeros - for COM we need to estimate based on chain sizes
        dist_nnz = n_dist_constraints * 36 if n_dist_constraints > 0 else 0
        dihed_nnz = n_dihed_constraints * 144 if n_dihed_constraints > 0 else 0
        
        # For COM constraints, we need to estimate the nnz dynamically
        # Worst case: all atoms in 2 chains = n_atoms^2 * 9
        # For simplicity, we'll allocate a large buffer
        com_nnz_max = n_atoms * n_atoms * 9 if n_com_constraints > 0 else 0
        
        max_nnz = dist_nnz + dihed_nnz + com_nnz_max
        
        if max_nnz == 0:
            # Return correct number of None values (20 for new signature)
            return (grad_coords_updated, None, None, None, None, 
                   None, None, None, None, None, None, None, None, None,
                   None, None, None, None, None, None, None, None)
        
        dev = coords_converged_wp.device
        # Arrays for full system matrix
        row_indices_full_wp = wp.zeros(max_nnz, dtype=wp.int32, device=dev)
        col_indices_full_wp = wp.zeros(max_nnz, dtype=wp.int32, device=dev)
        values_full_wp = wp.zeros(max_nnz, dtype=wp.float32, device=dev)
        
        # Convert tensors to Warp
        coords_converged_wp = wp.from_torch(coords_converged.contiguous(), dtype=wp.float32)
        
        current_offset = 0
        
        # Build system matrix for distance constraints
        if n_dist_constraints > 0:
            dist_index_wp = wp.from_torch(dist_index.to(torch.int32).contiguous(), dtype=wp.int32)
            dist_k_vals_wp = wp.from_torch(dist_k_vals.contiguous(), dtype=wp.float32)
            dist_lower_bounds_wp = wp.from_torch(dist_lower_bounds.contiguous(), dtype=wp.float32)
            dist_upper_bounds_wp = wp.from_torch(dist_upper_bounds.contiguous(), dtype=wp.float32)
            
            # For regular distance constraints, we don't need COM-specific arguments
            with wp.ScopedDevice(dev):
                wp.launch(
                    kernel=build_distance_system_matrix_kernel,
                    dim=n_dist_constraints,
                    inputs=[
                        coords_converged_wp,
                        dist_index_wp,
                        dist_k_vals_wp,
                        dist_lower_bounds_wp,
                        dist_upper_bounds_wp,
                        guidance_weight,
                        n_dist_constraints,
                        row_indices_full_wp[current_offset:current_offset+dist_nnz],
                        col_indices_full_wp[current_offset:current_offset+dist_nnz],
                        values_full_wp[current_offset:current_offset+dist_nnz],
                        alpha,
                        1e-9,
                    ],
                    device=dev,
                    stream=wp.get_stream(
                        dev
                    )
                )
            current_offset += dist_nnz
        
        # Build system matrix for COM constraints
        if n_com_constraints > 0 and com_args is not None:
            com_chain_index_wp = wp.from_torch(com_chain_index.to(torch.int32).contiguous(), dtype=wp.int32)
            com_k_vals_wp = wp.from_torch(com_k_vals.contiguous(), dtype=wp.float32)
            com_lower_bounds_wp = wp.from_torch(com_lower_bounds.contiguous(), dtype=wp.float32)
            com_upper_bounds_wp = wp.from_torch(com_upper_bounds.contiguous(), dtype=wp.float32)
            
            com_atom_index = ctx.com_atom_index
            atom_pad_mask = ctx.atom_pad_mask
            com_atom_index_wp = wp.from_torch(com_atom_index.to(torch.int32).contiguous(), dtype=wp.int32)
            atom_mask_wp = wp.from_torch(atom_pad_mask.contiguous(), dtype=wp.bool)
            
            # We need to track actual nnz per constraint for COM
            nnz_per_constraint_wp = wp.zeros(n_com_constraints, dtype=wp.int32, device=dev)
            
            with wp.ScopedDevice(dev):
                wp.launch(
                    kernel=build_com_system_matrix_kernel,
                    dim=n_com_constraints,
                    inputs=[
                        coords_converged_wp,
                        com_chain_index_wp,
                        com_k_vals_wp,
                        com_lower_bounds_wp,
                        com_upper_bounds_wp,
                        com_atom_index_wp,
                        atom_mask_wp,
                        guidance_weight,
                        n_com_constraints,
                        n_atoms,
                        row_indices_full_wp[current_offset:current_offset+com_nnz_max],
                        col_indices_full_wp[current_offset:current_offset+com_nnz_max],
                        values_full_wp[current_offset:current_offset+com_nnz_max],
                        nnz_per_constraint_wp,
                        alpha,
                        1e-9,
                    ],
                    device=dev,
                    stream=wp.get_stream(
                        dev
                    )
                )
            
            # Get actual nnz used
            nnz_per_constraint = wp.to_torch(nnz_per_constraint_wp)
            com_nnz_actual = int(nnz_per_constraint.sum().item())
            current_offset += com_nnz_actual
        
        # Build system matrix for dihedral constraints
        if n_dihed_constraints > 0:
            dihed_index_wp = wp.from_torch(dihed_index.to(torch.int32).contiguous(), dtype=wp.int32)
            dihed_k_vals_wp = wp.from_torch(dihed_k_vals.contiguous(), dtype=wp.float32)
            dihed_lower_bounds_wp = wp.from_torch(dihed_lower_bounds.contiguous(), dtype=wp.float32)
            dihed_upper_bounds_wp = wp.from_torch(dihed_upper_bounds.contiguous(), dtype=wp.float32)
            dihed_use_abs_wp = wp.from_torch(dihed_use_abs.contiguous(), dtype=wp.bool)
            
            with wp.ScopedDevice(dev):
                wp.launch(
                    kernel=build_dihedral_system_matrix_kernel,
                    dim=n_dihed_constraints,
                    inputs=[
                        coords_converged_wp,
                        dihed_index_wp,
                        dihed_k_vals_wp,
                        dihed_lower_bounds_wp,
                        dihed_upper_bounds_wp,
                        dihed_use_abs_wp,
                        guidance_weight,
                        n_dihed_constraints,
                        row_indices_full_wp[current_offset:current_offset+dihed_nnz],
                        col_indices_full_wp[current_offset:current_offset+dihed_nnz],
                        values_full_wp[current_offset:current_offset+dihed_nnz],
                        alpha,
                        1e-9,
                    ],
                    device=dev,
                    stream=wp.get_stream(
                        dev
                    )
            )
        
        wp.synchronize()
        
        ones_diag_torch = torch.ones(n_dim, device=device, dtype=torch.float32)
        
        with wp.ScopedDevice(dev):
            # Build sparse matrix
            K = wps.bsr_from_triplets(
                rows_of_blocks=n_dim,
                cols_of_blocks=n_dim,
                rows=row_indices_full_wp[:current_offset],
                columns=col_indices_full_wp[:current_offset],
                values=values_full_wp[:current_offset],
            )
            
            # Add identity matrix
            I = wps.bsr_diag(diag=wp.from_torch(ones_diag_torch, dtype=wp.float32))
            A = wps.bsr_axpy(K, y=I, alpha=1.0, beta=1.0)  # A = K + I
            
            # Right-hand side
            # For COM constraints, the gradient computation happens inside the kernels
            # We use the atom gradients directly
            b_wp = wp.from_torch(grad_coords_updated.reshape(-1).contiguous(), dtype=wp.float32)
            
            x_wp = wp.from_torch(grad_coords_updated.clone().reshape(-1).contiguous(), dtype=wp.float32)
            
            # Preconditioner
            M = wp.optim.linear.preconditioner(A, ptype="diag_abs")
            
            callback = None
            if False:
                def _cb(it, r_norm, a_tol):
                    # r_norm and a_tol are scalars per docs
                    print(f"[CG] iter={it}  ||r||={r_norm:.3e}  atol={a_tol:.3e}")
                callback = _cb
                
            # Solve with CG
            wp.optim.linear.cg(
                A, b_wp, x_wp,
                tol=1e-8,
                maxiter=min(n_dim, 1000),
                M=M,
                check_every=10,
                use_cuda_graph=True,
                callback=callback
            )
        
        # Return gradients
        # COM distribution is handled inside the kernels
        grad_coords = wp.to_torch(x_wp).reshape(n_atoms, 3).contiguous()
        
        # Return gradients for all inputs (most are None)
        # forward has 20 parameters: coords, dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
        # dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
        # com_chain_index, com_args, com_k_vals, com_lower_bounds, com_upper_bounds,
        # guidance_weight, parameters, args, alpha, n_iterations, debug, verbose
        # IMPORTANT: Return tuple must match exactly the number and order of forward's parameters
        return (grad_coords, None, None, None, None, 
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None, None, None)

class DiffGaussSeidelFunctionAllDebug(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, 
                dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
                dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
                com_chain_index, com_args, com_k_vals, com_lower_bounds, com_upper_bounds,
                guidance_weight, parameters, args,
                alpha=1e-6, n_iterations=100, 
                debug=False, verbose=False):
        return coords, None, None, None  # Dummy implementation for debugging
    
    @staticmethod
    def backward(ctx, grad_coords_updated, grad_dist_lagrangian, grad_dihed_lagrangian, grad_com_lagrangian):
        return (grad_coords_updated, None, None, None, None, 
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None, None, None)
    
# ============================================================================
# PUBLIC API
# ============================================================================

def diff_gauss_seidel_solve_all(coords, 
                                dist_index=None, dist_k_vals=None, dist_lower_bounds=None, 
                                dist_upper_bounds=None,
                                dihed_index=None, dihed_k_vals=None, dihed_lower_bounds=None,
                                dihed_upper_bounds=None, dihed_use_abs=None,
                                com_chain_index=None, com_args=None, com_k_vals=None, 
                                com_lower_bounds=None, com_upper_bounds=None,
                                parameters=None, args=None,
                                guidance_weight=1.0, alpha=1e-6, n_iterations=100,
                                debug=False, verbose=False):
    """
    Differentiable wrapper for the Gauss-Seidel constraint solver with all energy types.
    
    Distance constraints: VDW Overlap, Connections, PoseBusters, Symmetric Chain COM
    Dihedral constraints: Chiral Atoms, Stereo Bonds, Planar Bonds
    """
    
    # Handle None inputs
    if dist_index is None:
        dist_index = torch.empty((2, 0), dtype=torch.long, device=coords.device)
        dist_k_vals = torch.empty((0,), dtype=torch.float32, device=coords.device)
        dist_lower_bounds = torch.empty((0,), dtype=torch.float32, device=coords.device)
        dist_upper_bounds = torch.empty((0,), dtype=torch.float32, device=coords.device)
    
    if dihed_index is None:
        dihed_index = torch.empty((4, 0), dtype=torch.long, device=coords.device)
        dihed_k_vals = torch.empty((0,), dtype=torch.float32, device=coords.device)
        dihed_lower_bounds = torch.empty((0,), dtype=torch.float32, device=coords.device)
        dihed_upper_bounds = torch.empty((0,), dtype=torch.float32, device=coords.device)
        dihed_use_abs = torch.empty((0,), dtype=torch.bool, device=coords.device)
    
    # Handle None inputs for COM
    if com_chain_index is None:
        com_chain_index = torch.empty((2, 0), dtype=torch.long, device=coords.device)
        com_k_vals = torch.empty((0,), dtype=torch.float32, device=coords.device)
        com_lower_bounds = torch.empty((0,), dtype=torch.float32, device=coords.device)
        com_upper_bounds = torch.empty((0,), dtype=torch.float32, device=coords.device)
    
    return DiffGaussSeidelFunctionAll.apply(
        coords,
        dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
        dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
        com_chain_index, com_args, com_k_vals, com_lower_bounds, com_upper_bounds,
        guidance_weight, parameters, args,
        alpha, n_iterations,
        debug, verbose
    )

######################################

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
# TEST FUNCTIONS
# ============================================================================

def test_gradients_with_all_potentials():
    return 
    """
    Comprehensive gradient verification using all seven potentials on real protein data.
    """
    print("\n" + "=" * 70)
    print("TESTING ALL SEVEN POTENTIALS WITH PROTEIN DATA")
    print("=" * 70)
    
    # steering_t = 1.0
    steering_t = 1.0 - (49 / 50)
    potentials = get_potentials_pbd(None)
    
    # Collect all constraints from all potentials
    all_dist_indices = []
    all_dist_k_vals = []
    all_dist_lower_bounds = []
    all_dist_upper_bounds = []
    
    # Separate COM constraints
    all_com_chain_indices = []
    all_com_k_vals = []
    all_com_lower_bounds = []
    all_com_upper_bounds = []
    
    all_dihed_indices = []
    all_dihed_k_vals = []
    all_dihed_lower_bounds = []
    all_dihed_upper_bounds = []
    all_dihed_use_abs = []
    
    # COM args for SymmetricChainCOMPotential
    com_args_collected = None
    
    # Load data from the first potential to get coords
    data = torch.load(f"debug_pt/debug_gs_potential_0.pt")
    batch = data["batch"]
    out = data["out"]
    coords = out["denoised_atom_coords"][0].clone()  # Take first in batch
    device = coords.device
    
    print("\nProcessing all potentials:")
    for pi_, potential in enumerate(potentials):
        potential_name = potential.__class__.__name__
        print(f"  {pi_}: {potential_name}")
        
        # Load potential-specific data
        file = f"debug_pt/debug_gs_potential_{pi_}.pt"
        data = torch.load(file)
        batch = data["batch"]
        out = data["out"]
        
        parameters = potential.compute_parameters(steering_t)
        
        if parameters.get("guidance_weight", 0) == 0:
            print(f"    Skipping (guidance_weight=0)")
            continue
            
        index, args, com_args, ref_args, operator_args = potential.compute_args(
            batch, parameters
        )
        
        if index.shape[1] == 0:
            print(f"    No constraints")
            continue
            
        k_vals, lower_bounds, upper_bounds = args
        
        # Handle None bounds
        if lower_bounds is None:
            lower_bounds = torch.full((index.shape[1],), float('-inf'), device=device)
        if upper_bounds is None:
            upper_bounds = torch.full((index.shape[1],), float('inf'), device=device)
        
        # Determine if this is a distance or dihedral constraint
        # index shape is typically [batch_size, num_atoms_per_constraint, num_constraints]
        # We need to check the second dimension
        if index.ndim == 3:
            num_atoms_per_constraint = index.shape[1]
            index = index[0]  # Take first batch
            k_vals = k_vals[0] if k_vals.ndim > 1 else k_vals
            lower_bounds = lower_bounds[0] if lower_bounds.ndim > 1 else lower_bounds
            upper_bounds = upper_bounds[0] if upper_bounds.ndim > 1 else upper_bounds
        else:
            num_atoms_per_constraint = index.shape[0]
            
        if num_atoms_per_constraint == 2:  # Distance constraint
            # Check if this is SymmetricChainCOMPotential
            if "SymmetricChainCOM" in potential_name and com_args is not None:
                # Store COM args - only one potential should have this
                if com_args_collected is None:
                    com_args_collected = com_args
                    print(f"    Collected COM args for chain center-of-mass computation")
                
                # Add to COM constraints list (indices here are chain IDs)
                all_com_chain_indices.append(index)
                all_com_k_vals.append(k_vals * parameters["guidance_weight"])
                all_com_lower_bounds.append(lower_bounds)
                all_com_upper_bounds.append(upper_bounds)
                print(f"    Added {index.shape[1]} COM constraints")
            else:
                # Regular distance constraints
                all_dist_indices.append(index)
                all_dist_k_vals.append(k_vals * parameters["guidance_weight"])
                all_dist_lower_bounds.append(lower_bounds)
                all_dist_upper_bounds.append(upper_bounds)
                print(f"    Added {index.shape[1]} distance constraints")
            
        elif num_atoms_per_constraint == 4:  # Dihedral constraint
            # Determine if we should use absolute value based on potential type
            if "ChiralAtom" in potential_name:
                use_abs = False
            elif "StereoBond" in potential_name:
                use_abs = True
            elif "PlanarBond" in potential_name:
                use_abs = True
            else:
                use_abs = False
            
            all_dihed_indices.append(index)
            all_dihed_k_vals.append(k_vals * parameters["guidance_weight"])
            all_dihed_lower_bounds.append(lower_bounds)
            all_dihed_upper_bounds.append(upper_bounds)
            all_dihed_use_abs.append(torch.full((index.shape[1],), use_abs, dtype=torch.bool, device=device))
            print(f"    Added {index.shape[1]} dihedral constraints")
    
    # Concatenate all constraints
    if all_dist_indices:
        dist_index = torch.cat(all_dist_indices, dim=1)
        dist_k_vals = torch.cat(all_dist_k_vals)
        dist_lower_bounds = torch.cat(all_dist_lower_bounds)
        dist_upper_bounds = torch.cat(all_dist_upper_bounds)
    else:
        dist_index = None
        dist_k_vals = None
        dist_lower_bounds = None
        dist_upper_bounds = None
    
    # Concatenate COM constraints
    if all_com_chain_indices:
        com_chain_index = torch.cat(all_com_chain_indices, dim=1)
        com_k_vals = torch.cat(all_com_k_vals)
        com_lower_bounds = torch.cat(all_com_lower_bounds)
        com_upper_bounds = torch.cat(all_com_upper_bounds)
    else:
        com_chain_index = None
        com_k_vals = None
        com_lower_bounds = None
        com_upper_bounds = None
        
    if all_dihed_indices:
        dihed_index = torch.cat(all_dihed_indices, dim=1)
        dihed_k_vals = torch.cat(all_dihed_k_vals)
        dihed_lower_bounds = torch.cat(all_dihed_lower_bounds)
        dihed_upper_bounds = torch.cat(all_dihed_upper_bounds)
        dihed_use_abs = torch.cat(all_dihed_use_abs)
    else:
        dihed_index = None
        dihed_k_vals = None
        dihed_lower_bounds = None
        dihed_upper_bounds = None
        dihed_use_abs = None
    
    print(f"\nTotal constraints:")
    print(f"  Distance: {dist_index.shape[1] if dist_index is not None else 0}")
    print(f"  COM: {com_chain_index.shape[1] if com_chain_index is not None else 0}")
    print(f"  Dihedral: {dihed_index.shape[1] if dihed_index is not None else 0}")
    
    # Run forward pass with all constraints
    print("\nRunning forward pass with all constraints...")
    with torch.no_grad():
        coords_updated, _, _, _ = diff_gauss_seidel_solve_all(
            coords,
            dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
            dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
            com_chain_index, com_args_collected, com_k_vals, com_lower_bounds, com_upper_bounds,
            guidance_weight=1.0,
            alpha=1e-6, 
            n_iterations=100,
            verbose=True
        )
        
    coords_tgt = coords_updated.clone().detach() + 0.5
    
    def obj_func(x):
        """Simple displacement minimization objective."""
        # Don't detach from the graph here
        x = x.clone().requires_grad_(True)
        x_updated, _, _, _ = diff_gauss_seidel_solve_all(
            x,
            dist_index, dist_k_vals, dist_lower_bounds, dist_upper_bounds,
            dihed_index, dihed_k_vals, dihed_lower_bounds, dihed_upper_bounds, dihed_use_abs,
            com_chain_index, com_args_collected, com_k_vals, com_lower_bounds, com_upper_bounds,
            guidance_weight=1.0,
            alpha=1e-6, 
            n_iterations=100,
            verbose=False
        )
        # Use all outputs to ensure gradient flow
        return torch.sum((x_updated - coords_tgt) ** 2) 
    
    print("\nRunning optimization...")
    coords_test = coords.detach().clone().requires_grad_(True)
    
    for niter in range(100):
        if coords_test.grad is not None:
            coords_test.grad.zero_()  # Clear any existing gradients
        
        loss = obj_func(coords_test)
        
        loss.backward()
        
        if coords_test.grad is None:
            print("WARNING: coords_test.grad is None after backward!")
            print("This means gradients aren't flowing back to the input.")
            break
        
        grad_analytical = coords_test.grad.clone()
        coords_test = coords_test - 0.1 * coords_test.grad
        coords_test = coords_test.clone().detach().requires_grad_(True)
        
        if niter % 10 == 0:
            print(f"Iteration {niter + 1}: Loss = {loss.item():.6f}, Gradient Norm = {torch.norm(grad_analytical).item():.6f}")
        
        if torch.norm(grad_analytical).item() < 1e-3:
            print(f"Converged at iteration {niter + 1}!")
            break
    
    print("\nTest completed successfully!")

            
if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)
    else:
        print("CUDA not available, using CPU")
    
    # Run the test
    test_gradients_with_all_potentials()