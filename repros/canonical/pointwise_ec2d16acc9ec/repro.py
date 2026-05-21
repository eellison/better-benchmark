"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_infer
Pattern hash: ec2d16acc9ec
Shape hash: b3459786
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
_shapes_config = "(T([512, 3072], f16), S([1, 512, 3072]), S([512, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_4: "f16[512, 3072]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:297 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, _shape_param_0);  addmm_4 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_default: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:299 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        return reshape_default_1



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
