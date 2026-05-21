"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: b6007a189f72
Shape hash: 6be57a7c
"""
_shapes_config = "(T([64, 1024, 1024], f32), T([8, 8, 1024, 1024], b8), T([8, 8, 1024, 1024], f32), S([8, 8, 1024, 1024]), S([64, 1024, 1024]), S([8, 8, 1024, 1024]), S([64, 1024, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_101: "f32[64, 1024, 1024]", gt_6: "b8[8, 8, 1024, 1024]", div_3: "f32[8, 8, 1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_101, _shape_param_0);  bmm_101 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_tensor_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_3);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_default: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        reshape_default_1: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        reshape_default_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        reshape_default_3: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        return reshape_default_3



def make_inputs():
    return [
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 8, 1024, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 1024],  # _shape_param_0
    [64, 1024, 1024],  # _shape_param_1
    [8, 8, 1024, 1024],  # _shape_param_2
    [64, 1024, 1024],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
