"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer
Pattern hash: 7982e72007c9
Shape hash: e6f4cb23
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg6_1: "f32[50265]", mul_2: "f32[8, 1024, 768]", getitem_1: "f32[8, 1024, 1]", getitem: "f32[8, 1024, 1]", arg3_1: "f32[768]", arg4_1: "f32[768]", _shape_param_0, arg5_1: "f32[50265, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        full_default: "f32[3]" = torch.ops.aten.full.default([3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[50268]" = torch.ops.aten.cat.default([arg6_1, full_default]);  arg6_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  mul_2 = getitem_1 = None
        add_tensor: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg3_1);  mul_tensor = arg3_1 = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        reshape_default: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[768, 50265]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        constant_pad_nd_default: "f32[768, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (cat_default, reshape_default, constant_pad_nd_default)


def _default_make_inputs():
    return [
    torch.randn([50265], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [8192, 768],  # _shape_param_0
    torch.randn([50265, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
