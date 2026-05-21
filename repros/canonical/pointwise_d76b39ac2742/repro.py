"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_infer_001
Pattern hash: d76b39ac2742
Shape hash: 1258901a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1026, 768], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f16[1026, 768]"):
        # No stacktrace found for following nodes
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(iota_default, [1, -1]);  iota_default = None
        add_tensor: "i64[1, 512]" = torch.ops.aten.add.Tensor(expand_default, 2);  expand_default = None
        embedding_default: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, add_tensor);  arg0_1 = add_tensor = None
        return embedding_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
