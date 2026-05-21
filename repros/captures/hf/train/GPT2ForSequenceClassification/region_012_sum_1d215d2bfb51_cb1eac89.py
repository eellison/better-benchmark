"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train
Pattern hash: 1d215d2bfb51
Shape hash: cb1eac89
"""
_shapes_config = "(T([8192, 768], f32), T([768], f32), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], f32), T([8, 1024, 768], b8), T([3072, 768], f32), S([8, 1024, 768]), S([8192, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_89: "f32[8192, 768]", primals_16: "f32[768]", mul_14: "f32[8, 1024, 768]", div_24: "f32[8, 1024, 1]", add_145: "f32[8, 1024, 768]", gt_2: "b8[8, 1024, 768]", primals_15: "f32[3072, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_89, _shape_param_0);  mm_89 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_16);  reshape_default = primals_16 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_14);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_dim_int_list_1);  mul_14 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_tensor_1);  div_24 = sub_tensor_1 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_145, mul_tensor_4);  add_145 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_5);  add_tensor = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 768], dtype=torch.bool, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    [8, 1024, 768],  # _shape_param_0
    [8192, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
