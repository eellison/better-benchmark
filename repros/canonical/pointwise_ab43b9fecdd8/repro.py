"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_inference
Pattern hash: ab43b9fecdd8
Shape hash: 9e49ff64
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
    def forward(self, relu_96: "f32[8, 128, 512]", getitem_1: "f32[8, 128, 1]", getitem: "f32[8, 128, 1]", arg1115_1: "f32[512]", arg1116_1: "f32[512]", _shape_param_0, arg2_1: "f32[30522, 128]", arg1117_1: "f32[384, 30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[8, 128, 512]" = torch.ops.aten.sub.Tensor(relu_96, getitem_1);  relu_96 = getitem_1 = None
        add_tensor: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg1115_1);  mul_tensor = arg1115_1 = None
        add_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1116_1);  mul_tensor_1 = arg1116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        reshape_default: "f32[1024, 512]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[128, 30522]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        cat_default: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_default, arg1117_1]);  permute_default = arg1117_1 = None
        return (reshape_default, cat_default)


def _default_make_inputs():
    return [
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_0
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    torch.randn([384, 30522], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
