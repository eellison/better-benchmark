"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: 99f65f99076c
Shape hash: dd40a609
"""
_shapes_config = "(T([32768, 768], f32), T([256, 128, 768], f32), T([768], f32), T([256, 128, 768], f32), T([256, 128, 1], f32), T([768, 768], f32), S([256, 128, 768]), S([32768, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_66: "f32[32768, 768]", mul_248: "f32[256, 128, 768]", primals_15: "f32[768]", mul_8: "f32[256, 128, 768]", div_20: "f32[256, 128, 1]", primals_13: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_66, _shape_param_0);  mm_66 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_248, reshape_default);  mul_248 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_15);  add_tensor = primals_15 = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_8);  mul_tensor = None
        sum_dim_int_list_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_dim_int_list_1);  mul_8 = sum_dim_int_list_1 = None
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_tensor_1);  div_20 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [256, 128, 768],  # _shape_param_0
    [32768, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
