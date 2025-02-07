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
    ContextSensitiveTreeBuilder,
    GPT,
    GPTConfig,
    GreedySelector,
    MCTSEvaluator,
    MCTSSampler,
    PUCTSelector,
    RandomScorer,
    UCTSelector,
    ZMQScorer,
)


@dataclass
class MCTSDefaults:
    out_dir: str = "out"  # path to the folder containing the model checkpoint file
    temperature: float = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    start: str = "\n"  # the prompt; can also specify a file, use as: "FILE:prompt.txt"
    seed: int = 1337
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
    tree_width: int = 10  # the tree width
    max_depth: int = 1000  # the maximum depth of the tree
    c: float = 5.  # the selector constant: c_puct for PUCT, c for UCT, epsilon for greedy
    num_simulations: int = 200  # the number of simulations to perform during search
    bond_length_acceptability_cutoff: float = 1.0
    reward_k: float = 2.0  # the reward constant
    mcts_out_dir: str = "mcts"  # path to the directory where generated CIF files will be stored
    scorer: str = "zmq"  # supported values: 'zmq', 'random'
    scorer_host: str = "localhost"  # required if `scorer` is 'zmq'
    scorer_port: int = 5555  # required if `scorer` is 'zmq'
    use_context_sensitive_tree_builder: bool = True
    top_child_weight_cutoff: float = 0.99
    selector: str = "puct"  # valid values: 'puct', 'uct', 'greedy'
    n_space_groups: int = 0
    bypass_only_child: bool = False
    n_rollouts: int = 1  # the number of rollouts to perform per simulation


if __name__ == "__main__":
    C = parse_config(MCTSDefaults)

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

    # init from a model saved in a specific directory
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

    prompt = C.start
    if prompt.startswith("FILE:"):
        with open(prompt[5:], "r", encoding="utf-8") as f:
            prompt = f.read()

    cif_scorer = None
    if C.scorer == "zmq":
        cif_scorer = ZMQScorer(host=C.scorer_host, port=C.scorer_port)
    elif C.scorer == "random":
        cif_scorer = RandomScorer()
    else:
        raise Exception(f"unsupported scorer: {C.scorer}")

    evaluator = MCTSEvaluator(
        scorer=cif_scorer,
        tokenizer=tokenizer,
        bond_length_acceptability_cutoff=C.bond_length_acceptability_cutoff,
        reward_k=C.reward_k,
        out_dir=C.mcts_out_dir,
    )

    tree_builder = ContextSensitiveTreeBuilder(
        tokenizer=tokenizer,
        top_child_weight_cutoff=C.top_child_weight_cutoff,
        n_space_groups=C.n_space_groups,
        bypass_only_child=C.bypass_only_child,
    ) if C.use_context_sensitive_tree_builder else None

    if C.selector == "puct":
        node_selector = PUCTSelector(cpuct=C.c)
    elif C.selector == "greedy":
        node_selector = GreedySelector(epsilon=C.c)
    elif C.selector == "uct":
        node_selector = UCTSelector(c=C.c)
    else:
        raise Exception(f"unsupported selector: {C.selector}")

    sampler = MCTSSampler(
        model=model,
        config=gptconf,
        width=C.tree_width,
        max_depth=C.max_depth,
        eval_function=evaluator,
        node_selector=node_selector,
        tokenizer=tokenizer,
        temperature=C.temperature,
        device=C.device,
        tree_builder=tree_builder,
    )

    sampler.search(prompt, C.num_simulations, stepwise=False, n_rollouts=C.n_rollouts)
