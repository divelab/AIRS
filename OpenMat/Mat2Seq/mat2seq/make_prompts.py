import sys
sys.path.append(".")
import argparse
import gzip
import pickle
import re
import io
import tarfile
from tqdm import tqdm
from mat2seq import CIFTokenizer

PATTERN_COMP = re.compile(r"(data_[^\n]*\n)", re.MULTILINE)
PATTERN_COMP_SG = re.compile(r"(data_[^\n]*\n)loop_[\s\S]*?(_symmetry_space_group_name_H-M[^\n]*\n)", re.MULTILINE)


def extract_prompt(cif_str, pattern, space=False):
    match = re.search(pattern, cif_str)
    if match:
        start_index, end_index = match.start(), match.end()
        extracted = cif_str[start_index:end_index]
        lines = extracted.split('\n')
        
        for line in lines:
            # Extract space group symbol
            res = ""
            if "_symmetry_space_group_name_H-M" in line:
                spaceg = "space_group_symbol " + line.split()[-1] + "_sg\n"
            # Extract formula
            elif line.startswith("data_"):
                form = ''
                form += "formula "
                formula = line.split("_")[1]
                elements_counts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
                for element, count in elements_counts:
                    if not element: break
                    if not count: count ="1"
                    form += element + " " + count + "_int "
                form += "\n"
        if space: res = form + spaceg
        else: res = form
        print(res)
        return res
    else:
        raise Exception(f"could not extract pattern: \n{cif_str}")


"""
Takes a collection of pre-processed CIFs, and constructs .txt files
with prompts for each CIF. Note that the CIFs must be the result of
the pre-processing step (i.e. the output of the `bin/preprocess.py` 
script).
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct prompts from the given pre-processed CIFs.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the pre-processed CIFs from which prompts will be extracted. "
                             "It is expected that the file contains the gzipped contents of a pickled Python "
                             "list of tuples, of (id, cif) pairs.")
    parser.add_argument("--out", "-o", action="store",
                        required=True,
                        help="Path to the gzipped tarball where the prompt .txt files will be stored. "
                             "It is recommended that the filename end in `.tar.gz`.")
    parser.add_argument("--with-spacegroup", action="store_true",
                        help="Include this flag if the prompts must contain the structure's space group.")

    args = parser.parse_args()

    cifs_fname = args.name
    out_fname = args.out
    with_spacegroup = args.with_spacegroup

    print(f"loading data from {cifs_fname}...")
    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    # since these cifs are pre-processed, extract first lines, optionally up to space group

    with tarfile.open(out_fname, "w:gz") as tar:
        for id, cif in tqdm(cifs, desc="preparing prompts..."):
            prompt = extract_prompt(cif, PATTERN_COMP_SG if with_spacegroup else PATTERN_COMP, space=with_spacegroup)

            prompt_file = tarfile.TarInfo(name=f"{id}.txt")
            prompt_bytes = prompt.encode("utf-8")
            prompt_file.size = len(prompt_bytes)
            tar.addfile(prompt_file, io.BytesIO(prompt_bytes))
