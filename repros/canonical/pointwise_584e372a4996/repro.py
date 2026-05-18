"""
Standalone repro captured via capture_hook.
Label: falcon_rw_1b
Pattern hash: 584e372a4996
Shape hash: cea8b91e
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
    def forward(self, mm_94: "f16[2048, 8192]", _shape_param_0, arg289_1: "f16[8192]", _shape_param_1, arg290_1: "f16[2048, 8192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:62 in forward, code: hidden_states = input @ self.weight.T
        reshape_default: "f16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:65 in forward, code: return hidden_states + self.bias
        add_tensor: "f16[4, 512, 8192]" = torch.ops.aten.add.Tensor(reshape_default, arg289_1);  reshape_default = arg289_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_default: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        mul_tensor: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[4, 512, 8192]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor_1: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        convert_element_type_default_1: "f16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:62 in forward, code: hidden_states = input @ self.weight.T
        reshape_default_1: "f16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([2048, 8192], dtype=torch.float16, device='cuda'),
    [4, 512, 8192],  # _shape_param_0
    torch.randn([8192], dtype=torch.float16, device='cuda'),
    [2048, 8192],  # _shape_param_1
    torch.randn([2048, 8192], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
