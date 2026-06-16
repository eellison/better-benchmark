"""Quick single-graph bench for genai full graphs (A/B testing config changes).

Usage:
  TORCHINDUCTOR_CACHE_DIR=/tmp/ti_x python scripts/genai_single.py \
      repros/models/genai/SoftmaxBackward/full_graph_001.py [--cd] [--dump-kernels DIR]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--cd", action="store_true", help="coordinate descent tuning")
    ap.add_argument("--max-autotune", action="store_true")
    ap.add_argument("--dump-kernels", default=None)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-rep", type=int, default=50)
    args = ap.parse_args()

    if args.dump_kernels:
        os.environ["TORCH_COMPILE_DEBUG"] = "0"
        os.environ.setdefault("TORCHINDUCTOR_DEBUG_DIR", args.dump_kernels)

    import torch
    from torch._inductor import config as inductor_config

    if args.cd:
        inductor_config.coordinate_descent_tuning = True
    if args.max_autotune:
        inductor_config.max_autotune = True

    from full_graph_harness import load_full_graph

    gm, inputs, definition = load_full_graph(args.graph)
    gm = gm.cuda()
    inputs = [
        x.cuda() if torch.is_tensor(x) else x for x in inputs
    ]

    compiled = torch.compile(gm, dynamic=False)
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
    torch.cuda.synchronize()

    from triton.testing import do_bench

    with torch.no_grad():
        ms = do_bench(
            lambda: compiled(*inputs), warmup=args.n_warmup, rep=args.n_rep
        )
    print(f"RESULT {args.graph} {ms * 1000:.1f} us")


if __name__ == "__main__":
    main()
