import sys
sys.path.append(".")
import os
import argparse
import io
import tarfile

from crystallm import (
    extract_space_group_symbol,
    replace_symmetry_operators,
    remove_atom_props_block,
)


def postprocess(cif: str, fname: str) -> str:
    try:
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif = replace_symmetry_operators(cif, space_group_symbol)

        # remove atom props
        cif = remove_atom_props_block(cif)
    except Exception as e:
        cif = "# WARNING: CrystaLLM could not post-process this file properly!\n" + cif
        print(f"error post-processing CIF file '{fname}': {e}")

    return cif


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the directory or .tar.gz file containing the "
                             "raw CIF files to be post-processed.")
    parser.add_argument("out", type=str,
                        help="Path to the directory or .tar.gz file where the "
                             "post-processed CIF files should be written")

    args = parser.parse_args()

    input_path = args.name
    output_path = args.out

    if input_path.endswith(".tar.gz"):
        with tarfile.open(input_path, "r:gz") as tar, tarfile.open(output_path, "w:gz") as out_tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".cif"):
                    file = tar.extractfile(member)
                    cif_str = file.read().decode()
                    processed_cif = postprocess(cif_str, member.name)

                    processed_file = io.BytesIO(processed_cif.encode())
                    tarinfo = tarfile.TarInfo(name=member.name)
                    tarinfo.size = len(processed_cif.encode())
                    out_tar.addfile(tarinfo, fileobj=processed_file)

                    print(f"processed: {member.name}")

    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for filename in os.listdir(input_path):
            if filename.endswith(".cif"):
                file_path = os.path.join(input_path, filename)
                with open(file_path, "r") as file:
                    cif_str = file.read()
                    processed_cif = postprocess(cif_str, filename)

                output_file_path = os.path.join(output_path, filename)
                with open(output_file_path, "w") as file:
                    file.write(processed_cif)
                print(f"processed: {filename}")
