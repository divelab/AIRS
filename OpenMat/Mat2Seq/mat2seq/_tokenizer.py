import os
import re
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(THIS_DIR, "spacegroups.txt"), "rt") as f:
    SPACE_GROUPS = [sg.strip() for sg in f.readlines()]


ATOMS = ["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
         "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
         "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
         "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
         "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar"]

DIGITS = [str(d) for d in list(range(10))]

INTS = [str(d) for d in list(range(300))]

KEYWORDS = [
    "space_group_symbol",
    "formula",
    "atoms",
    "lattice_parameters",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma"
]

UNK_TOKEN = "<unk>"


class CIFTokenizer:
    def __init__(self):
        self._tokens = ["<pad>"]
        self._tokens.extend(self.atoms())
        self._tokens.extend(self.digits())
        self._tokens.extend(self.keywords())
        self._tokens.extend(self.symbols())

        space_groups = list(self.space_groups())
        # Replace 'Pm' space group with 'Pm_sg' to disambiguate from atom 'Pm',
        #  or 'P1' with 'P1_sg' to disambiguate from atom 'P' and number '1'
        space_groups_sg = [sg+"_sg" for sg in space_groups]
        self._tokens.extend(space_groups_sg)

        digits_int = [v+"_int" for v in INTS]
        self._tokens.extend(digits_int)

        self._escaped_tokens = [re.escape(token) for token in self._tokens]
        self._escaped_tokens.sort(key=len, reverse=True)

        # a mapping from characters to integers
        self._token_to_id = {ch: i for i, ch in enumerate(self._tokens)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._tokens)}
        # map the id of 'Pm_sg' back to 'Pm', or 'P1_sg' to 'P1',
        #  for decoding convenience
        for sg in space_groups_sg:
            self._id_to_token[self.token_to_id[sg]] = sg.replace("_sg", "")
        
        for v_int in digits_int:
            self._id_to_token[self.token_to_id[v_int]] = v_int.replace("_int", "")

    @staticmethod
    def atoms():
        return ATOMS

    @staticmethod
    def digits():
        return DIGITS

    @staticmethod
    def keywords():
        kws = list(KEYWORDS)
        return kws

    @staticmethod
    def symbols():
        # return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]
        return [",", " ", ":", ".", "\n"]

    @staticmethod
    def space_groups():
        return SPACE_GROUPS

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)
    
    def prompt_tokenize(self, cif):
        token_pattern = '|'.join(self._escaped_tokens)
        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f'({token_pattern}|\\w+|[\\.,;!?])'
        # Tokenize the input string using the regex pattern
        cif = re.sub(r'[ \t]+', ' ', cif)
        tokens = re.findall(full_pattern, cif)
        return tokens

    def encode(self, tokens):
        # encoder: take a list of tokens, output a list of integers
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids):
        # decoder: take a list of integers (i.e. encoded tokens), output a string
        return ''.join([self._id_to_token[i] for i in ids])

    def tokenize_cif(self, cif_string, max_length=1385):
        # Preprocessing step to replace '_symmetry_space_group_name_H-M Pm'
        #  with '_symmetry_space_group_name_H-M Pm_sg',to disambiguate from atom 'Pm',
        #  or any space group symbol to avoid problematic cases, like 'P1'
        spacegroups = "|".join(SPACE_GROUPS)
        cif_string = re.sub(fr'(_symmetry_space_group_name_H-M *\b({spacegroups}))\n', r'\1_sg\n', cif_string)

        extracted_data = self.tokenize_cif_preprocess(cif_string)

        seq_res = ''
        # formula
        seq_res += "formula "
        formula = extracted_data["formula"]
        elements_counts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        for element, count in elements_counts:
            if not element: break
            if not count: count ="1"
            seq_res += element + " " + count + "_int "
        seq_res += "\n"
        # space group name
        seq_res += "space_group_symbol " + extracted_data["space_group_symbol"] + "\n"
        # lattice
        seq_res += "lattice_parameters " + "a " + extracted_data["lattice_parameters"]["a"] + " "
        seq_res += "b " + extracted_data["lattice_parameters"]["b"] + " "
        seq_res += "c " + extracted_data["lattice_parameters"]["c"] + " "
        seq_res += "alpha " + extracted_data["lattice_parameters"]["alpha"] + " "
        seq_res += "beta " + extracted_data["lattice_parameters"]["beta"] + " "
        seq_res += "gamma " + extracted_data["lattice_parameters"]["gamma"] + " "
        seq_res += "\n"
        # atoms
        for idx in range(len(extracted_data["atoms"])):
            tmp = extracted_data["atoms"][idx]
            seq_res += tmp["type"] + " " + tmp["num"] + "_int " + tmp["coordinates"][0] + " " + tmp["coordinates"][1] + " " + tmp["coordinates"][2] + "\n"
        seq_res += "\n"
        # Create a regex pattern by joining the escaped tokens with '|'
        # print(seq_res)
        token_pattern = '|'.join(self._escaped_tokens)
        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f'({token_pattern}|\\w+|[\\.,;!?])'
        # Tokenize the input string using the regex pattern
        seq_res = re.sub(r'[ \t]+', ' ', seq_res)
        tokens = re.findall(full_pattern, seq_res)
        # print(tokens)
        padding_length = max_length - len(tokens)
        if padding_length > 0:
            tokens.extend(["<pad>"] * padding_length)

        return tokens

    def tokenize_cif_preprocess(self, cif_string):
        # Re-initialize the dictionary to hold the extracted data
        extracted_data = {
            "space_group_symbol": "",
            "formula": "",
            "atoms": [],
            "lattice_parameters": {}
        }

        # Split the text into lines for processing
        lines = cif_string.split('\n')

        # Iterate through each line to extract the required information
        atom_line_idx = -1
        for line_idx in range(len(lines)):
            line = lines[line_idx]
            # Extract space group symbol
            if "_symmetry_space_group_name_H-M" in line:
                extracted_data["space_group_symbol"] = line.split()[-1]
            # Extract formula
            elif line.startswith("data_"):
                extracted_data["formula"] = line.split("_")[1]
            # Extract lattice parameters
            elif line.startswith("_cell_length_a"):
                extracted_data["lattice_parameters"]["a"] = line.split()[-1]
            elif line.startswith("_cell_length_b"):
                extracted_data["lattice_parameters"]["b"] = line.split()[-1]
            elif line.startswith("_cell_length_c"):
                extracted_data["lattice_parameters"]["c"] = line.split()[-1]
            elif line.startswith("_cell_angle_alpha"):
                extracted_data["lattice_parameters"]["alpha"] = line.split()[-1]
            elif line.startswith("_cell_angle_beta"):
                extracted_data["lattice_parameters"]["beta"] = line.split()[-1]
            elif line.startswith("_cell_angle_gamma"):
                extracted_data["lattice_parameters"]["gamma"] = line.split()[-1]
            elif "_atom_site_occupancy" in line:
                atom_line_idx = line_idx + 1
                break

        for line_idx in range(atom_line_idx, len(lines)):
            line = lines[line_idx]
            if len(line) < 2:
                continue
            atom_info = line.split()
            atom_type = atom_info[0]
            num_atoms = atom_info[2]
            x, y, z = atom_info[3], atom_info[4], atom_info[5]
            extracted_data["atoms"].append({
                "type": atom_type,
                "num": num_atoms,
                "coordinates": (x, y, z)
            })

        return extracted_data


class CinDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx][:1500]
        # if self.conditions is not None:
        #     raw_input_ids = raw_input_ids[1:]  # Remove the first token (<s>)
        input_ids = text[:-1]
        targets = text[1:]
        return input_ids, targets
