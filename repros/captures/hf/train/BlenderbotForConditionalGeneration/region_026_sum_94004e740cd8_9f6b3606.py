"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 94004e740cd8
Shape hash: 9f6b3606
"""
_shapes_config = "(T([2048, 2560], f32), T([2560], f32), T([16, 128, 2560], f32), T([16, 128, 1], f32), T([16, 128, 2560], f32), T([16, 128, 2560], b8), T([2560, 2560], f32), S([16, 128, 2560]), S([2048, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[2048, 2560]", primals_25: "f32[2560]", mul_10: "f32[16, 128, 2560]", div_1: "f32[16, 128, 1]", tangents_1: "f32[16, 128, 2560]", gt_1: "b8[16, 128, 2560]", primals_23: "f32[2560, 2560]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(reshape_default, primals_25);  reshape_default = primals_25 = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, 2560)
        sum_dim_int_list: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_10);  mul_tensor = None
        sum_dim_int_list_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_10, sum_dim_int_list_1);  mul_10 = sum_dim_int_list_1 = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_1, sub_tensor_1);  div_1 = sub_tensor_1 = None
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(tangents_1, mul_tensor_4);  tangents_1 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:382 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor_5: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_5);  add_tensor = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[2048, 2560]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        permute_default: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([2048, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [16, 128, 2560], dtype=torch.bool, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [16, 128, 2560],  # _shape_param_0
    [2048, 2560],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
