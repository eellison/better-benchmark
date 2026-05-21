"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_train
Pattern hash: 287efb5bb271
Shape hash: 50902d21
"""
_shapes_config = "(T([16, 1024, 768], f32), T([768], f32), T([16, 1024, 768], f32), T([16, 1024, 1], f32), T([16, 1024, 768], b8), T([768, 3072], f32), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[16, 1024, 768]", primals_17: "f32[768]", mul_9: "f32[16, 1024, 768]", div: "f32[16, 1024, 1]", gt_1: "b8[16, 1024, 768]", primals_15: "f32[768, 3072]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, primals_17);  tangents_1 = primals_17 = None
        mul_tensor_1: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_9);  mul_tensor = None
        sum_dim_int_list_1: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_dim_int_list_1);  mul_9 = sum_dim_int_list_1 = None
        sub_tensor: "f32[16, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[16, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_tensor_1);  div = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:477 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[16, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor_5: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_0);  mul_tensor_6 = _shape_param_0 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)



def make_inputs():
    return [
    torch.randn([16, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [16, 1024, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
