"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_infer
Pattern hash: dfb3f69ba7a2
Shape hash: 2c3ab6db
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
_shapes_config = "(T([8, 2048], i64), T([2048, 512], f32), T([64, 512], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg5_1: "i64[8, 2048]", addmm: "f32[2048, 512]", arg3_1: "f32[64, 512]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_int: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 0)
        select_int_1: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 1)
        select_int_2: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 2)
        select_int_3: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 3)
        select_int_4: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 4)
        select_int_5: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 5)
        select_int_6: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 6)
        select_int_7: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 7);  arg5_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        relu_default: "f32[2048, 512]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_default: "f32[512, 64]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        return (select_int, select_int_1, select_int_2, select_int_3, select_int_4, select_int_5, select_int_6, select_int_7, relu_default, permute_default)



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
