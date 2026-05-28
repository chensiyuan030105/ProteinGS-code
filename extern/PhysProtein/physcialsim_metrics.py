import os
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import argparse
import itertools

from boltz.data.mol import load_molecules
from boltz.data import const
from boltz.data.parse.mmcif_with_constraints import parse_mmcif

def signed_torsion(coords, idxs):
    """Compute signed torsion angle (−π, π)."""
    i, j, k, l = idxs
    r_ij = coords[i] - coords[j]
    r_kj = coords[k] - coords[j]
    r_kl = coords[k] - coords[l]

    n1 = np.cross(r_ij, r_kj)
    n2 = np.cross(r_kj, r_kl)

    angle = np.arctan2(
        np.dot(np.cross(n1, n2), r_kj / np.linalg.norm(r_kj)),
        np.dot(n1, n2)
    )
    return angle

def compute_rs_from_permutation(coords, center, neighbors):
    idxs = [neighbors[0], center, neighbors[1], neighbors[2]]
    angle = signed_torsion(coords, idxs)
    return angle > 0  # True = R, False = S

def check_chiral_center(coords, center_idx, neighbor_idxs, true_is_r):
    """
    neighbor_idxs: list of 3 neighbor atom indices
    true_is_r: CCD chirality (True=R, False=S)
    """

    for perm in itertools.permutations(neighbor_idxs, 3):
        pred_is_r = compute_rs_from_permutation(
            coords, center_idx, perm
        )
        if pred_is_r == true_is_r:
            return False  # no violation

    return True  # violation if all permutations fail


def robust_check_ligand_stereochemistry(structure, constraints):
    coords = structure.coords["coords"]

    chiral = constraints.chiral_atom_constraints
    atom_idxs = chiral["atom_idxs"]  # shape: [N, 4]
    true_is_r = chiral["is_r"]
    mask = chiral["is_reference"]

    violations = 0
    total = 0

    for i in range(len(atom_idxs)):
        if not mask[i]:
            continue

        c, a, b, d = atom_idxs[i]   # center + 3 neighbors
        center = c
        neighbors = [a, b, d]
        total += 1

        if check_chiral_center(coords, center, neighbors, true_is_r[i]):
            violations += 1

    return {
        "num_chiral_atom_violations": violations,
        "num_chiral_atoms": total,
    }

def compute_torsion_angles(coords, torsion_index):
    r_ij = coords[..., torsion_index[0], :] - coords[..., torsion_index[1], :]
    r_kj = coords[..., torsion_index[2], :] - coords[..., torsion_index[1], :]
    r_kl = coords[..., torsion_index[2], :] - coords[..., torsion_index[3], :]
    n_ijk = np.cross(r_ij, r_kj, axis=-1)
    n_jkl = np.cross(r_kj, r_kl, axis=-1)
    r_kj_norm = np.linalg.norm(r_kj, axis=-1)
    n_ijk_norm = np.linalg.norm(n_ijk, axis=-1)
    n_jkl_norm = np.linalg.norm(n_jkl, axis=-1)
    sign_phi = np.sign(
        r_kj[..., None, :] @ np.cross(n_ijk, n_jkl, axis=-1)[..., None]
    ).squeeze(axis=(-1, -2))
    phi = sign_phi * np.arccos(
        np.clip(
            (n_ijk[..., None, :] @ n_jkl[..., None]).squeeze(axis=(-1, -2))
            / (n_ijk_norm * n_jkl_norm),
            -1 + 1e-8,
            1 - 1e-8,
        )
    )
    return phi


def check_ligand_distance_geometry(
    structure, constraints, bond_buffer=0.25, angle_buffer=0.25, clash_buffer=0.2
):
    coords = structure.coords["coords"]
    rdkit_bounds_constraints = constraints.rdkit_bounds_constraints
    pair_index = rdkit_bounds_constraints["atom_idxs"].copy().astype(np.int64).T
    bond_mask = rdkit_bounds_constraints["is_bond"].copy().astype(bool)
    angle_mask = rdkit_bounds_constraints["is_angle"].copy().astype(bool)
    upper_bounds = rdkit_bounds_constraints["upper_bound"].copy().astype(np.float32)
    lower_bounds = rdkit_bounds_constraints["lower_bound"].copy().astype(np.float32)
    dists = np.linalg.norm(coords[pair_index[0]] - coords[pair_index[1]], axis=-1)
    bond_length_violations = (
        dists[bond_mask] <= lower_bounds[bond_mask] * (1.0 - bond_buffer)
    ) + (dists[bond_mask] >= upper_bounds[bond_mask] * (1.0 + bond_buffer))
    bond_angle_violations = (
        dists[angle_mask] <= lower_bounds[angle_mask] * (1.0 - angle_buffer)
    ) + (dists[angle_mask] >= upper_bounds[angle_mask] * (1.0 + angle_buffer))
    internal_clash_violations = dists[~bond_mask * ~angle_mask] <= lower_bounds[
        ~bond_mask * ~angle_mask
    ] * (1.0 - clash_buffer)
    num_ligands = sum(
        [
            int(const.chain_types[chain["mol_type"]] == "NONPOLYMER")
            for chain in structure.chains
        ]
    )
    return {
        "num_ligands": num_ligands,
        "num_bond_length_violations": bond_length_violations.sum(),
        "num_bonds": bond_mask.sum(),
        "num_bond_angle_violations": bond_angle_violations.sum(),
        "num_angles": angle_mask.sum(),
        "num_internal_clash_violations": internal_clash_violations.sum(),
        "num_non_neighbors": (~bond_mask * ~angle_mask).sum(),
    }


def check_ligand_stereochemistry(structure, constraints):
    coords = structure.coords["coords"]
    chiral_atom_constraints = constraints.chiral_atom_constraints
    stereo_bond_constraints = constraints.stereo_bond_constraints

    chiral_atom_index = chiral_atom_constraints["atom_idxs"].T
    true_chiral_atom_orientations = chiral_atom_constraints["is_r"]
    chiral_atom_ref_mask = chiral_atom_constraints["is_reference"]
    chiral_atom_index = chiral_atom_index[:, chiral_atom_ref_mask]
    true_chiral_atom_orientations = true_chiral_atom_orientations[chiral_atom_ref_mask]
    pred_chiral_atom_orientations = (
        compute_torsion_angles(coords, chiral_atom_index) > 0
    )
    chiral_atom_violations = (
        pred_chiral_atom_orientations != true_chiral_atom_orientations
    )

    stereo_bond_index = stereo_bond_constraints["atom_idxs"].T
    true_stereo_bond_orientations = stereo_bond_constraints["is_e"]
    stereo_bond_ref_mask = stereo_bond_constraints["is_reference"]
    stereo_bond_index = stereo_bond_index[:, stereo_bond_ref_mask]
    true_stereo_bond_orientations = true_stereo_bond_orientations[stereo_bond_ref_mask]
    pred_stereo_bond_orientations = (
        np.abs(compute_torsion_angles(coords, stereo_bond_index)) > np.pi / 2
    )
    stereo_bond_violations = (
        pred_stereo_bond_orientations != true_stereo_bond_orientations
    )

    return {
        # "num_chiral_atom_violations": chiral_atom_violations.sum(),
        # "num_chiral_atoms": chiral_atom_index.shape[1],
        "num_stereo_bond_violations": stereo_bond_violations.sum(),
        "num_stereo_bonds": stereo_bond_index.shape[1],
    }


def check_ligand_flatness(structure, constraints, buffer=0.25):
    coords = structure.coords["coords"]

    planar_ring_5_index = constraints.planar_ring_5_constraints["atom_idxs"]
    ring_5_coords = coords[planar_ring_5_index, :]
    centered_ring_5_coords = ring_5_coords - ring_5_coords.mean(axis=-2, keepdims=True)
    ring_5_vecs = np.linalg.svd(centered_ring_5_coords)[2][..., -1, :, None]
    ring_5_dists = np.abs((centered_ring_5_coords @ ring_5_vecs).squeeze(axis=-1))
    ring_5_violations = np.all(ring_5_dists <= buffer, axis=-1)

    planar_ring_6_index = constraints.planar_ring_6_constraints["atom_idxs"]
    ring_6_coords = coords[planar_ring_6_index, :]
    centered_ring_6_coords = ring_6_coords - ring_6_coords.mean(axis=-2, keepdims=True)
    ring_6_vecs = np.linalg.svd(centered_ring_6_coords)[2][..., -1, :, None]
    ring_6_dists = np.abs((centered_ring_6_coords @ ring_6_vecs)).squeeze(axis=-1)
    ring_6_violations = np.any(ring_6_dists >= buffer, axis=-1)

    planar_bond_index = constraints.planar_bond_constraints["atom_idxs"]
    bond_coords = coords[planar_bond_index, :]
    centered_bond_coords = bond_coords - bond_coords.mean(axis=-2, keepdims=True)
    bond_vecs = np.linalg.svd(centered_bond_coords)[2][..., -1, :, None]
    bond_dists = np.abs((centered_bond_coords @ bond_vecs)).squeeze(axis=-1)
    bond_violations = np.any(bond_dists >= buffer, axis=-1)

    return {
        "num_planar_5_ring_violations": ring_5_violations.sum(),
        "num_planar_5_rings": ring_5_violations.shape[0],
        "num_planar_6_ring_violations": ring_6_violations.sum(),
        "num_planar_6_rings": ring_6_violations.shape[0],
        "num_planar_double_bond_violations": bond_violations.sum(),
        "num_planar_double_bonds": bond_violations.shape[0],
    }


def check_steric_clash(structure, molecules, buffer=0.25):
    result = {}
    for type_i in const.chain_types:
        out_type_i = type_i.lower()
        out_type_i = out_type_i if out_type_i != "nonpolymer" else "ligand"
        result[f"num_chain_pairs_sym_{out_type_i}"] = 0
        result[f"num_chain_clashes_sym_{out_type_i}"] = 0
        for type_j in const.chain_types:
            out_type_j = type_j.lower()
            out_type_j = out_type_j if out_type_j != "nonpolymer" else "ligand"
            result[f"num_chain_pairs_asym_{out_type_i}_{out_type_j}"] = 0
            result[f"num_chain_clashes_asym_{out_type_i}_{out_type_j}"] = 0

    connected_chains = set()
    for bond in structure.bonds:
        if bond["chain_1"] != bond["chain_2"]:
            connected_chains.add(tuple(sorted((bond["chain_1"], bond["chain_2"]))))

    vdw_radii = []
    for res in structure.residues:
        mol = molecules[res["name"]]
        token_atoms = structure.atoms[
            res["atom_idx"] : res["atom_idx"] + res["atom_num"]
        ]
        atom_name_to_ref = {a.GetProp("name"): a for a in mol.GetAtoms()}
        token_atoms_ref = [atom_name_to_ref[a["name"]] for a in token_atoms]
        vdw_radii.extend(
            [const.vdw_radii[a.GetAtomicNum() - 1] for a in token_atoms_ref]
        )
    vdw_radii = np.array(vdw_radii, dtype=np.float32)

    np.array([a.GetAtomicNum() for a in token_atoms_ref])
    for i, chain_i in enumerate(structure.chains):
        for j, chain_j in enumerate(structure.chains):
            if (
                chain_i["atom_num"] == 1
                or chain_j["atom_num"] == 1
                or j <= i
                or (i, j) in connected_chains
            ):
                continue
            coords_i = structure.coords["coords"][
                chain_i["atom_idx"] : chain_i["atom_idx"] + chain_i["atom_num"]
            ]
            coords_j = structure.coords["coords"][
                chain_j["atom_idx"] : chain_j["atom_idx"] + chain_j["atom_num"]
            ]
            dists = np.linalg.norm(coords_i[:, None, :] - coords_j[None, :, :], axis=-1)
            radii_i = vdw_radii[
                chain_i["atom_idx"] : chain_i["atom_idx"] + chain_i["atom_num"]
            ]
            radii_j = vdw_radii[
                chain_j["atom_idx"] : chain_j["atom_idx"] + chain_j["atom_num"]
            ]
            radii_sum = radii_i[:, None] + radii_j[None, :]
            is_clashing = np.any(dists < radii_sum * (1.00 - buffer))
            type_i = const.chain_types[chain_i["mol_type"]].lower()
            type_j = const.chain_types[chain_j["mol_type"]].lower()
            type_i = type_i if type_i != "nonpolymer" else "ligand"
            type_j = type_j if type_j != "nonpolymer" else "ligand"
            is_symmetric = (
                chain_i["entity_id"] == chain_j["entity_id"]
                and chain_i["atom_num"] == chain_j["atom_num"]
            )
            if is_symmetric:
                key = "sym_" + type_i
            else:
                key = "asym_" + type_i + "_" + type_j
            result["num_chain_pairs_" + key] += 1
            result["num_chain_clashes_" + key] += int(is_clashing)
    return result

def check_steric_clash_ligand_protein(structure, molecules,
                                      buffer=0.25,
                                      search_distance=6.0):
    """
    Simplified Boltz-style steric clash check:
    - ONLY ligand vs protein chain pairs
    - PB-valid logic (ignore hydrogens, 6A filter, cutoff=0.75 vdW_sum)
    """

    result = {
        "num_chain_pairs_asym_ligand_protein": 0,
        "num_chain_clashes_asym_ligand_protein": 0,
    }

    # Identify ligand (nonpolymer) and protein chains
    ligand_chains = []
    protein_chains = []
    for idx, chain in enumerate(structure.chains):
        chain_type = const.chain_types[chain["mol_type"]].lower()
        if chain_type == "nonpolymer":
            ligand_chains.append(idx)
        elif chain_type == "protein":
            protein_chains.append(idx)

    # Build vdW radii aligned with structure atoms
    vdw_radii = []
    atom_Z = []
    for res in structure.residues:
        mol = molecules[res["name"]]
        atoms_struct = structure.atoms[res["atom_idx"]:res["atom_idx"] + res["atom_num"]]
        atom_map = {a.GetProp("name"): a for a in mol.GetAtoms()}
        atoms_ref = [atom_map[a["name"]] for a in atoms_struct]
        vdw_radii.extend([const.vdw_radii[a.GetAtomicNum() - 1] for a in atoms_ref])
        atom_Z.extend([a.GetAtomicNum() for a in atoms_ref])

    vdw_radii = np.array(vdw_radii, dtype=np.float32)
    atom_Z = np.array(atom_Z, dtype=np.int32)

    # check ligand–protein ONLY
    for i in ligand_chains:
        for j in protein_chains:

            chain_i = structure.chains[i]
            chain_j = structure.chains[j]

            # extract coords
            idx_i = slice(chain_i["atom_idx"], chain_i["atom_idx"] + chain_i["atom_num"])
            idx_j = slice(chain_j["atom_idx"], chain_j["atom_idx"] + chain_j["atom_num"])

            coords_i = structure.coords["coords"][idx_i]
            coords_j = structure.coords["coords"][idx_j]

            radii_i = vdw_radii[idx_i]
            radii_j = vdw_radii[idx_j]
            Z_i = atom_Z[idx_i]
            Z_j = atom_Z[idx_j]

            # remove hydrogens
            mask_i = Z_i != 1
            mask_j = Z_j != 1
            coords_i = coords_i[mask_i]
            coords_j = coords_j[mask_j]
            radii_i = radii_i[mask_i]
            radii_j = radii_j[mask_j]

            if coords_i.size == 0 or coords_j.size == 0:
                continue

            # pairwise distances
            dists = np.linalg.norm(coords_i[:, None, :] - coords_j[None, :, :], axis=-1)

            # PB-valid: 6A cutoff for selecting relevant protein atoms
            close_mask_j = dists.min(axis=0) <= search_distance
            if not close_mask_j.any():
                is_clashing = False
            else:
                coords_j = coords_j[close_mask_j]
                radii_j = radii_j[close_mask_j]
                dists = dists[:, close_mask_j]

                radii_sum = radii_i[:, None] + radii_j[None, :]
                cutoff = 1.0 - buffer  # 0.75
                is_clashing = np.any(dists < radii_sum * cutoff)

            # update Boltz-style counters
            result["num_chain_pairs_asym_ligand_protein"] += 1
            result["num_chain_clashes_asym_ligand_protein"] += int(is_clashing)

    return result

def process_fn_safe(key):
    tool, pdb_id, model_dir, seed, model_idx = key
    # Initialize the record dictionary with default fields
    record = {
        "tool": tool,
        "pdb_id": pdb_id,
        "model_idx": model_idx,
        "seed": seed,
        "error": None  # Store any exception message here
    }

    try:
        cif_path = None
        print("pdb_id =", )
        # Determine the CIF path based on the tool
        if tool == "boltz-1" or tool == "boltz-1x" or tool == "boltz-2" or tool == "boltz-gs":
            cif_path = model_dir / pdb_id / f"seed_{seed}" / f"boltz_results_{pdb_id}" / \
                       "predictions" / pdb_id / f"{pdb_id}_model_{model_idx}.cif"            
        elif tool == "protenix" or tool == "protenix-mini-10step" or tool == "protenix-mini-5step":
            cif_path = model_dir / pdb_id / pdb_id / f"seed_{seed}" / "predictions" / \
                       f"{pdb_id}_sample_{model_idx}.cif"
        elif tool == "AlphaFold3":
            cif_path = model_dir / pdb_id / f"seed-1_sample-{model_idx}/{pdb_id}_seed-1_sample-{model_idx}_model.cif"
        print("Processing pdb_id =", pdb_id, ", model_idx =", model_idx)

        # Parse the CIF file
        parsed_structure = parse_mmcif(
            cif_path,
            ccd,
            moldir,
        )
        structure = parsed_structure.data
        constraints = parsed_structure.residue_constraints
            
        # Update the record with various ligand and steric checks
        record.update(check_ligand_distance_geometry(structure, constraints))
        record.update(check_ligand_stereochemistry(structure, constraints))
        record.update(robust_check_ligand_stereochemistry(structure, constraints))
        record.update(check_ligand_flatness(structure, constraints))
        record.update(check_steric_clash(structure, molecules=ccd))
        # record.update(check_steric_clash_ligand_protein(structure, molecules=ccd))

    except Exception as e:
        import traceback
        # Store the full traceback for debugging purposes
        record["error"] = traceback.format_exc()
        print(f"Error processing {pdb_id}, model_idx {model_idx}: {e}")

    # Return the record (either successful results or error info)
    return record

# -------------------- args --------------------
parser = argparse.ArgumentParser(description="Check physical constraints for CASP models")
parser.add_argument("--tool", type=str, required=True,
                    choices=["AlphaFold3", "boltz-1", "protenix", "boltz-1x", "boltz-2", "protenix-mini-10step", "protenix-mini-5step", "boltz-gs"],
                    help="which tool's models to evaluate")
parser.add_argument("--num_samples", type=int, default=5, help="number of samples per pdb_id")
parser.add_argument("--model_dir", type=str, required=True, help="directory containing predicted models")
parser.add_argument("--output_csv", type=str, required=True, help="path to save results CSV")
parser.add_argument("--output_txt", type=str, required=True, help="path to save results txt")
args = parser.parse_args()
# -------------------- args --------------------

cache_dir = Path("./pretrained_model")
ccd_path = cache_dir / "ccd.pkl"
moldir = cache_dir / "mols"
with ccd_path.open("rb") as file:
    ccd = pickle.load(file)

# use user-specified model_dir
model_dir = Path(args.model_dir)
model_pdb_ids = sorted(os.listdir(model_dir))

print(f"Processing {len(model_pdb_ids)} PDB IDs (first 100)")

tools = [args.tool]
num_samples = args.num_samples

seeds = [101]
keys = []
for tool in tools:
    for pdb_id in model_pdb_ids:
        for seed in seeds:
            for model_idx in range(num_samples):
                keys.append((tool, pdb_id, model_dir, seed, model_idx))

# process_fn(keys[0])
records = []
with Pool(48) as p:
    with tqdm(total=len(keys)) as pbar:
        for record in p.imap_unordered(process_fn_safe, keys):
            records.append(record)
            pbar.update(1)
df = pd.DataFrame.from_records(records)
# Filter out entries that failed
errors = df[df["error"].notnull()]
print(f"Number of failed entries: {len(errors)}")

df["num_chain_clashes_all"] = df[
    [key for key in df.columns if "chain_clash" in key]
].sum(axis=1)
df["num_pairs_all"] = df[[key for key in df.columns if "chain_pair" in key]].sum(axis=1)
df["clash_free"] = df["num_chain_clashes_all"] == 0
keys = [key for key in df.columns if ("violations" in key or ("ligand" in key and "chain_clashes" in key))]
print("Selected keys for valid_ligand:")
for k in keys:
    print(k)
df["valid_ligand"] = (
    df[[key for key in df.columns if "violations" in key]].sum(axis=1) == 0
)
df["valid"] = (df["clash_free"]) & (df["valid_ligand"])
df_ok = df[df["error"].isnull()]
prop = (df["valid"] == True).mean()
print("Proportion =", prop)

# save results to user-specified location
df.to_csv(args.output_csv)
print(f"Saved results to {args.output_csv}")

# Filter out records that succeeded (no error)
successful_records = df[df["error"].isnull()]

# Extract unique pdb_ids
successful_pdb_ids = successful_records["pdb_id"].unique()

# Save to a text file, one PDB ID per line
with open(args.output_txt, "w") as f:
    for pdb_id in successful_pdb_ids:
        f.write(f"{pdb_id}\n")

print(f"Saved {len(successful_pdb_ids)} successful PDB IDs to successful_pdb_ids.txt")

output_root = Path(args.output_csv).parent / "physical_check"
output_root.mkdir(exist_ok=True)

physical_cols = [
    "tool","pdb_id","model_idx","seed","error",
    "num_ligands",
    "num_bond_length_violations","num_bonds",
    "num_bond_angle_violations","num_angles",
    "num_internal_clash_violations","num_non_neighbors",
    "num_chiral_atom_violations","num_chiral_atoms",
    "num_stereo_bond_violations","num_stereo_bonds",
    "num_planar_5_ring_violations","num_planar_5_rings",
    "num_planar_6_ring_violations","num_planar_6_rings",
    "num_planar_double_bond_violations","num_planar_double_bonds",
    "num_chain_pairs_sym_protein","num_chain_clashes_sym_protein",
    "num_chain_pairs_asym_protein_protein","num_chain_clashes_asym_protein_protein",
    "num_chain_pairs_asym_protein_dna","num_chain_clashes_asym_protein_dna",
    "num_chain_pairs_asym_protein_rna","num_chain_clashes_asym_protein_rna",
    "num_chain_pairs_asym_protein_ligand","num_chain_clashes_asym_protein_ligand",
    "num_chain_pairs_sym_dna","num_chain_clashes_sym_dna",
    "num_chain_pairs_asym_dna_protein","num_chain_clashes_asym_dna_protein",
    "num_chain_pairs_asym_dna_dna","num_chain_clashes_asym_dna_dna",
    "num_chain_pairs_asym_dna_rna","num_chain_clashes_asym_dna_rna",
    "num_chain_pairs_asym_dna_ligand","num_chain_clashes_asym_dna_ligand",
    "num_chain_pairs_sym_rna","num_chain_clashes_sym_rna",
    "num_chain_pairs_asym_rna_protein","num_chain_clashes_asym_rna_protein",
    "num_chain_pairs_asym_rna_dna","num_chain_clashes_asym_rna_dna",
    "num_chain_pairs_asym_rna_rna","num_chain_clashes_asym_rna_rna",
    "num_chain_pairs_asym_rna_ligand","num_chain_clashes_asym_rna_ligand",
    "num_chain_pairs_sym_ligand","num_chain_clashes_sym_ligand",
    "num_chain_pairs_asym_ligand_protein","num_chain_clashes_asym_ligand_protein",
    "num_chain_pairs_asym_ligand_dna","num_chain_clashes_asym_ligand_dna",
    "num_chain_pairs_asym_ligand_rna","num_chain_clashes_asym_ligand_rna",
    "num_chain_pairs_asym_ligand_ligand","num_chain_clashes_asym_ligand_ligand",
    "num_chain_clashes_all","num_pairs_all","clash_free","valid_ligand","valid"
]

print(f"\nSaving per-row physical checks (vertical txt) to {output_root} ...")

for idx, row in df.iterrows():
    pdb = str(row["pdb_id"])
    model_idx = str(row["model_idx"])

    row_dir = output_root / pdb / model_idx
    row_dir.mkdir(parents=True, exist_ok=True)

    row_path = row_dir / "physical_check.txt"

    with open(row_path, "w") as f:
        for col in physical_cols:
            val = row[col]
            f.write(f"{col}: {val}\n")

print("Per-row vertical physical_check.txt files saved!")