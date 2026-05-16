"""
Standalone repro captured via capture_hook.
Label: bert_large
Pattern hash: c0f0885a79c6
Shape hash: bc95c35e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg397_1: "f16[30522]", convert_element_type_default: "f32[4, 512, 1024]", getitem_315: "f32[4, 512, 1]", getitem_314: "f32[4, 512, 1]", arg395_1: "f16[1024]", arg396_1: "f16[1024]", _shape_param_0, arg3_1: "f16[30522, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        full_default: "f16[6]" = torch.ops.aten.full.default([6], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f16[30528]" = torch.ops.aten.cat.default([arg397_1, full_default]);  arg397_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_315);  convert_element_type_default = getitem_315 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_314, 1e-12);  getitem_314 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg395_1);  mul_tensor = arg395_1 = None
        add_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg396_1);  mul_tensor_1 = arg396_1 = None
        convert_element_type_default_1: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f16[2048, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_0);  convert_element_type_default_1 = _shape_param_0 = None
        permute_default: "f16[1024, 30522]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        constant_pad_nd_default: "f16[1024, 30528]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 6, 0, 0]);  permute_default = None
        return (cat_default, reshape_default, constant_pad_nd_default)


def _default_make_inputs():
    return [
    torch.randn([30522], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    [2048, 1024],  # _shape_param_0
    torch.randn([30522, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
