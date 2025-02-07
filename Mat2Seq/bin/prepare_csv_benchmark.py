import io
import pandas as pd
import tarfile
import argparse
from pymatgen.io.cif import CifWriter, Structure
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def process_cif_files(input_csv, output_tar_gz):
    df = pd.read_csv(input_csv)
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="preparing CIF files..."):
            id = row["material_id"]
            struct = Structure.from_str(row["cif"], fmt="cif")
            cif_content = CifWriter(struct=struct, symprec=0.1).__str__()

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
    args = parser.parse_args()

    process_cif_files(args.input_csv, args.output_tar_gz)

    print(f"prepared CIF files have been saved to {args.output_tar_gz}")
