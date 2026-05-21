"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train
Pattern hash: 985afdf85560
Shape hash: bacf6a98
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
_shapes_config = "(T([8, 4096, 512], b8), T([8, 4096, 512], f32), T([512], f32), T([8, 4096, 512], f32), T([8, 4096, 1], f32), T([8, 4096, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, gt: "b8[8, 4096, 512]", tangents_1: "f32[8, 4096, 512]", primals_1: "f32[512]", primals_3: "f32[8, 4096, 512]", getitem_1: "f32[8, 4096, 1]", rsqrt: "f32[8, 4096, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1799 in torch_dynamo_resume_in_forward_at_1781, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[8, 4096, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0526315789473684);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(tangents_1, mul_tensor);  tangents_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1796 in torch_dynamo_resume_in_forward_at_1781, code: hidden_states = self.layer_norm(hidden_states)
        mul_tensor_2: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_1);  primals_1 = None
        mul_tensor_3: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 512)
        sum_dim_int_list: "f32[8, 4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        sub_tensor: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor_4: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_5: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[8, 4096, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
        mul_tensor_7: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        return (sum_dim_int_list_3, sum_dim_int_list_2, mul_tensor_7)



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
