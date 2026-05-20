"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 1da7daeba28a
Shape hash: 4dc0e88c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 2048], f32), T([8, 1024, 2048], b8), T([8, 1024, 2048], b8), T([], f32), T([2048, 512], f32), S([8, 1024, 2048]), S([8192, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm_268: "f32[8192, 2048]", gt_8: "b8[8, 1024, 2048]", le_11: "b8[8, 1024, 2048]", full_default: "f32[]", primals_18: "f32[2048, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_268, _shape_param_0);  mm_268 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_tensor: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_self: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_11, full_default, mul_tensor_1);  le_11 = full_default = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default_1: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        permute_default: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_1: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


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
