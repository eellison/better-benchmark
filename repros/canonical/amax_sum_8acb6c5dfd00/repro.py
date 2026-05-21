"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_infer
Pattern hash: 8acb6c5dfd00
Shape hash: 4c5cae5a
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
_shapes_config = "(T([8, 2048, 2048], f16), T([1, 8, 2048, 2048], f16), S([1, 8, 2048, 2048]), S([1, 8, 2048, 2048]), S([8, 2048, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_34: "f16[8, 2048, 2048]", add_50: "f16[1, 8, 2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_34, _shape_param_0);  bmm_34 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(reshape_default, add_50);  reshape_default = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_default: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        amax_default: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        reshape_default_1: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
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
