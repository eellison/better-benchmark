"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bert_train
Pattern hash: 2f1249e5fe40
Shape hash: 0f32fa52
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
_shapes_config = "(T([2048, 768], f32), T([4, 512, 768], f32), T([2048, 768], f32), T([2048, 768], f32), T([768], f32), T([4, 512, 768], f32), T([4, 512, 1], f32), T([4, 512, 768], b8), S([4, 512, 768]), S([4, 512, 768]), S([4, 512, 768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_130: "f32[2048, 768]", mul_526: "f32[4, 512, 768]", mm_132: "f32[2048, 768]", mm_134: "f32[2048, 768]", primals_23: "f32[768]", mul_17: "f32[4, 512, 768]", div_37: "f32[4, 512, 1]", gt_3: "b8[4, 512, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_130, _shape_param_0);  mm_130 = _shape_param_0 = None
        add_tensor: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_526, reshape_default);  mul_526 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_132, _shape_param_1);  mm_132 = _shape_param_1 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_134, _shape_param_2);  mm_134 = _shape_param_2 = None
        add_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_23);  add_tensor_2 = primals_23 = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_17);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_dim_int_list_1);  mul_17 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_tensor_1);  div_37 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_tensor_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_3: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        return reshape_default_3



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
