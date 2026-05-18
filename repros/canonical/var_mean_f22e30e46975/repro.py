"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g62
Pattern hash: f22e30e46975
Shape hash: 1bafafdc
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[21128, 768]", arg3_1: "i64[4, 476]", arg4_1: "f32[512, 768]", arg5_1: "f32[2, 768]", arg1_1: "i64[4, 476]"):
        # No stacktrace found for following nodes
        iota_default: "i64[476]" = torch.ops.prims.iota.default(476, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 476]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        expand_default: "i64[4, 476]" = torch.ops.aten.expand.default(unsqueeze_default, [4, 476]);  unsqueeze_default = None
        embedding_default: "f32[4, 476, 768]" = torch.ops.aten.embedding.default(arg2_1, arg3_1, 0);  arg2_1 = arg3_1 = None
        embedding_default_1: "f32[4, 476, 768]" = torch.ops.aten.embedding.default(arg4_1, expand_default);  arg4_1 = expand_default = None
        embedding_default_2: "f32[4, 476, 768]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None
        add_tensor: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        add_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[4, 476, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 476, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([21128, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 476], dtype=torch.int64, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 476], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
