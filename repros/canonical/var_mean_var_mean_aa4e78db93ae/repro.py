"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: aa4e78db93ae
Shape hash: db544d80
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
    def forward(self, getitem: "f32[32, 50, 1]", add: "f32[32, 50, 768]", getitem_1: "f32[32, 50, 1]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg154_1: "f32[49408, 512]", arg153_1: "i64[32, 77]", arg155_1: "f32[77, 512]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[32, 50, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 50, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul_tensor: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg4_1);  mul_tensor = arg4_1 = None
        add_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg5_1);  mul_tensor_1 = arg5_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem_2: "f32[32, 50, 1]" = var_mean_correction[0]
        getitem_3: "f32[32, 50, 1]" = var_mean_correction[1];  var_mean_correction = None
        embedding_default: "f32[32, 77, 512]" = torch.ops.aten.embedding.default(arg154_1, arg153_1);  arg154_1 = arg153_1 = None
        add_tensor_2: "f32[32, 77, 512]" = torch.ops.aten.add.Tensor(embedding_default, arg155_1);  embedding_default = arg155_1 = None
        permute_default: "f32[77, 32, 512]" = torch.ops.aten.permute.default(add_tensor_2, [1, 0, 2]);  add_tensor_2 = None
        clone_default: "f32[77, 32, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(clone_default, [2], correction = 0, keepdim = True);  clone_default = None
        getitem_4: "f32[77, 32, 1]" = var_mean_correction_1[0]
        getitem_5: "f32[77, 32, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem_2, getitem_3, getitem_4, getitem_5)


def _default_make_inputs():
    return [
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([49408, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 49408, [32, 77], dtype=torch.int64, device='cuda'),
    torch.randn([77, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
