import sys
sys.path.append(".")
import os
import argparse
import tarfile
import pickle
import torch

from pymatgen.core import Element

from crystallm import (
    CIFTokenizer,
    GPTConfig,
    GPT,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract learned embeddings.")
    parser.add_argument("name", type=str, required=True,
                        help="Path to the folder containing the model checkpoint file.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the tokenized dataset file (.tar.gz).")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to the .csv file where the results should be written.")
    parser.add_argument("--type", type=str, required=True, choices=["atom", "digit", "spacegroup"],
                        help="Type of the embedding to extract ('atom', 'digit', or 'spacegroup').")
    args = parser.parse_args()

    model_dir = args.name
    dataset_fname = args.dataset
    out_fname = args.out
    embedding_type = args.type

    base_path = os.path.splitext(os.path.basename(dataset_fname))[0]
    base_path = os.path.splitext(base_path)[0]

    with tarfile.open(dataset_fname, "r:gz") as file:
        file_content_byte = file.extractfile(f"{base_path}/meta.pkl").read()
        meta = pickle.loads(file_content_byte)

    tokenizer = CIFTokenizer()

    device = "cpu"
    checkpoint = torch.load(os.path.join(model_dir, "ckpt.pt"), map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)

    embedding_weights = model.transformer.wte.weight

    print(f"embedding weights shape: {embedding_weights.shape}")

    assert meta["vocab_size"] == embedding_weights.shape[0], \
        "the size of the vocab does not match the size of the embedding table"

    stoi = meta["stoi"]

    sorted_elems = sorted([(e, Element(e).number) for e in tokenizer.atoms()], key=lambda v: v[1])
    dim = embedding_weights.shape[1]

    with open(out_fname, "wt") as f:
        header = ["element"]
        header.extend([str(i) for i in range(dim)])
        f.write("%s\n" % ",".join(header))
        if embedding_type == "atom":
            for elem, _ in sorted_elems:
                vec = embedding_weights[stoi[elem]].tolist()
                row = [elem]
                row.extend([str(v) for v in vec])
                f.write("%s\n" % ",".join(row))
        elif embedding_type == "digit":
            for i in range(10):
                digit = str(i)
                vec = embedding_weights[stoi[digit]].tolist()
                row = [digit]
                row.extend([str(v) for v in vec])
                f.write("%s\n" % ",".join(row))
        elif embedding_type == "spacegroup":
            for sg in tokenizer.space_groups():
                spacegroup = f"{sg}_sg"
                vec = embedding_weights[stoi[spacegroup]].tolist()
                row = [spacegroup]
                row.extend([str(v) for v in vec])
                f.write("%s\n" % ",".join(row))
