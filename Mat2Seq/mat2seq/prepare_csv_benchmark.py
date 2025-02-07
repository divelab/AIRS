import io
import pandas as pd
import tarfile
import argparse
from pymatgen.io.cif import Structure
from mycif import CifWriter
import re
import numpy as np
from tqdm import tqdm
from spglib import standardize_cell, find_primitive
from pymatgen.analysis.structure_matcher import StructureMatcher

import warnings
warnings.filterwarnings("ignore")

def standard_pymatgen_cell(structure, to_primitive=False, symprec=1e-5, test=False):
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    atomic_numbers = structure.atomic_numbers
    cell = (lattice, positions, atomic_numbers)
    standard_cell = standardize_cell(cell, to_primitive=to_primitive, symprec=symprec)
    standard_struct = Structure(species=standard_cell[2], coords=standard_cell[1], lattice = standard_cell[0])
    if test:
        struct_matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        assert struct_matcher.get_rms_dist(standard_struct, structure) < 1e-5
    return standard_struct

def correct_zero(structure):
    # find the original point of a unit cell, deterministically

    # Step 1: Find the indexes with the smallest atomic number
    min_atomic_number = min([site.specie.Z for site in structure])
    min_atomic_number_indexes = [i for i, site in enumerate(structure) if site.specie.Z == min_atomic_number]

    # Step 2 & 3: Calculate the neighborhood and density for these indexes using radius 1 Å, 2 Å, and 3 Å
    all_neighbors = []
    all_neighbors.append(structure.get_all_neighbors(r=1.0))
    all_neighbors.append(structure.get_all_neighbors(r=2.0))
    all_neighbors.append(structure.get_all_neighbors(r=3.0))
    density_results = []
    for index in min_atomic_number_indexes:
        densities = {}
        for radius in [1, 2, 3]:  # Radii in Angstroms
            neighbors = all_neighbors[radius - 1][index]
            density = sum([neighbor.specie.Z for neighbor in neighbors])
            densities[radius] = density
        
        # Radius of 3, and three directions
        neighbors = all_neighbors[-1][index]
        site_coords = structure.frac_coords[index]
        density_x = sum([neighbor.specie.Z for neighbor in neighbors if (neighbor.frac_coords[0] - site_coords[0]) > 1e-3])
        density_y = sum([neighbor.specie.Z for neighbor in neighbors if (neighbor.frac_coords[1] - site_coords[1]) > 1e-3])
        density_z = sum([neighbor.specie.Z for neighbor in neighbors if (neighbor.frac_coords[2] - site_coords[2]) > 1e-3])
        densities[4] = density_x
        densities[5] = density_y
        densities[6] = density_z
        density_results.append((index, densities))

    # Step 4: Sort these indexes by density, in decreasing order
    sorted_by_density = sorted(density_results, key=lambda x: (x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6]), reverse=True)
    # print(sorted_by_density)

    zero_idx = sorted_by_density[0][0]
    zero_pos = structure.frac_coords[zero_idx]
    pos = (structure.frac_coords - np.array(zero_pos)) % 1.
    # correct the structure by periodic shift
    structure = Structure(species=structure.species, coords=pos, lattice = structure.lattice)
    return structure


def process_cif_files(input_csv, output_tar_gz, pos_order=False, full_order=False):
    df = pd.read_csv(input_csv)
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="preparing CIF files..."):
            id = row["material_id"]
            struct = Structure.from_str(row["cif"], fmt="cif")
            struct = standard_pymatgen_cell(struct)
            struct = correct_zero(struct)
            cif_content = CifWriter(struct=struct, symprec=0.1, pos_order=pos_order, full_order=full_order).__str__()
            
            cif_file = tarfile.TarInfo(name=f"{id}.cif")
            cif_bytes = cif_content.encode("utf-8")
            cif_file.size = len(cif_bytes)
            tar.addfile(cif_file, io.BytesIO(cif_bytes))


"""
This script is meant to be used to prepare the CDVAE benchmark 
.csv files (https://github.com/txie-93/cdvae/tree/main/data and 
https://github.com/jiaor17/DiffCSP/tree/main/data) for the 
CrystaLLM pre-processing step.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark CIF files and save to a tar.gz file.")
    parser.add_argument("input_csv", help="Path to the .csv containing the benchmark CIF files.")
    parser.add_argument("output_tar_gz", help="Path to the output tar.gz file")
    parser.add_argument("--pos_order", type=int, default=0, help="0: original order, 1: pos order")
    parser.add_argument("--full_order", type=int, default=0, help="0: original order, 1: pos order")
    args = parser.parse_args()

    print("using pos order? ",  bool(args.pos_order))
    print("using full order? ",  bool(args.full_order))

    process_cif_files(args.input_csv, args.output_tar_gz, pos_order=bool(args.pos_order), full_order=bool(args.full_order))

    print(f"prepared CIF files have been saved to {args.output_tar_gz}")
