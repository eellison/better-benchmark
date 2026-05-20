"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 4d83649d940a
Shape hash: d2a55982
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 2048, 768], f32), T([8192, 768], b8), T([768, 3072], f32), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 2048, 768]", gt_1: "b8[8192, 768]", primals_17: "f32[768, 3072]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        reshape_default: "f32[8192, 768]" = torch.ops.aten.reshape.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:245 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (mul_tensor_1, permute_default_1)


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
