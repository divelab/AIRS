"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/sample.py
"""
import sys
sys.path.append(".")
import os
from dataclasses import dataclass

from contextlib import nullcontext
from omegaconf import OmegaConf
import torch

from crystallm import (
    parse_config,
    CIFTokenizer,
    GPT,
    GPTConfig,
)


@dataclass
class SampleDefaults:
    out_dir: str = "out"  # the path to the directory containing the trained model
    start: str = "\n"  # the prompt; can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 2  # number of samples to draw
    max_new_tokens: int = 3000  # number of tokens generated in each sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 10  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster


if __name__ == "__main__":
    C = parse_config(SampleDefaults)

    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    torch.manual_seed(C.seed)
    torch.cuda.manual_seed(C.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=C.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(C.device)
    if C.compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    # encode the beginning of the prompt
    prompt = C.start
    if prompt.startswith("FILE:"):
        with open(prompt[5:], "r", encoding="utf-8") as f:
            prompt = f.read()
    start_ids = encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...]

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(C.num_samples):
                y = model.generate(x, C.max_new_tokens, temperature=C.temperature, top_k=C.top_k)
                print(decode(y[0].tolist()))
                print('---------------')
