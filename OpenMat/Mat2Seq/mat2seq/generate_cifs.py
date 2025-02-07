import sys
sys.path.append(".")
import os
import argparse
import io
import tarfile
import multiprocessing as mp

from tqdm import tqdm
from contextlib import nullcontext
import torch

from mat2seq import CIFTokenizer
from mat2seq import (
    GPT,
    GPTConfig,
)
from crystallm import array_split

# Set the visible CUDA devices to GPU 0 and GPU 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def progress_listener(queue, n):
    pbar = tqdm(total=n, desc="generating CIFs from prompts...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def generate(model_dir, seed, device, dtype, num_gens, temperature, top_k, chunk_of_prompts, queue):
    # init torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init tokenizer
    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    print(f"initializing model from {model_dir} on {device}...")
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    # model = torch.compile(model)  # requires PyTorch 2.0

    generated = []
    with torch.no_grad():
        with ctx:
            for id, prompt in chunk_of_prompts:
                print(prompt)
                start_ids = encode(tokenizer.prompt_tokenize(prompt))
                x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...].repeat(num_gens, 1)
                gens = []
                y, gen_ends = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, num_gens=num_gens)
                y = [y[i_idx, : gen_ends[i_idx]] for i_idx in range(num_gens)]
                for i_idx in range(num_gens):
                    output = decode(y[i_idx].tolist())
                    print(output)
                    gens.append(output)
                generated.append((id, gens))
                # print(gens)
                queue.put(1)
    return generated


"""
This script takes a collection of prompts and generates a corresponding CIF file for each prompt.

If there are multiple GPUs available on the same machine, generation can be done in parallel by 
distributing the prompts across the GPUs, and collecting all the results at the end. The number
of GPUs to be used can be specified with the `--gpus` argument.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIFs from the given prompts.")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to the directory containing the trained model checkpoint file.")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to the .tar.gz file containing the prompt .txt files.")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to the gzipped tarball where the generated CIF files will be stored. "
                             "It is recommended that the filename end in `.tar.gz`.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="The top-k value to use during sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=3000,
                        help="The maximum number of tokens to generate per CIF.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use.")
    parser.add_argument("--temperature", type=float, default=1.0, help="The sampling temperature.")
    parser.add_argument("--seed", type=int, default=1337, help="The random seed.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"],
                        help="The datatype to use.")
    parser.add_argument("--num-gens", type=int, default=1,
                        help="The number of times to generate for each CIF.")
    parser.add_argument("--gpus", type=int,
                        help="The number of GPUs to use. "
                             "The number of GPUs specified must be available on the same machine.")

    args, _ = parser.parse_known_args()
    if args.device == "cuda":
        gpus_avail = torch.cuda.device_count()
    else:
        gpus_avail = 0
    parser.set_defaults(gpus=gpus_avail)
    args = parser.parse_args()

    model_dir = args.model
    prompts_file = args.prompts
    out_file = args.out
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    device = args.device
    temperature = args.temperature
    seed = args.seed
    dtype = args.dtype
    num_gens = args.num_gens
    gpus = args.gpus

    if device == "cuda" and gpus > gpus_avail:
        print(f"ERROR: There are {gpus_avail} GPU(s) available but {gpus} was specified.")
        sys.exit(1)

    workers = 1 if device == "cpu" else gpus

    prompts = []
    with tarfile.open(prompts_file, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting prompts..."):
            f = tar.extractfile(member)
            if f is not None:
                content = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".txt", "")
                prompts.append((cif_id, content))

    chunks = array_split(prompts, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers + 1)  # add an extra worker for the watcher
    watcher = pool.apply_async(progress_listener, (queue, len(prompts),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        dev = f"cuda:{i}" if device == "cuda" else device
        job = pool.apply_async(generate, (model_dir, seed, dev, dtype, num_gens, temperature, top_k, chunk, queue))
        jobs.append(job)

    generated = []
    for job in jobs:
        generated.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()


    with tarfile.open(out_file, "w:gz") as tar:
        for id, gens in tqdm(generated, desc=f"writing CIF files to {out_file}..."):
            for i, cif in enumerate(gens):
                cif_file = tarfile.TarInfo(name=f"{id}__{i+1}.cif")
                cif_bytes = cif.encode("utf-8")
                cif_file.size = len(cif_bytes)
                tar.addfile(cif_file, io.BytesIO(cif_bytes))
