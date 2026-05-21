"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train
Pattern hash: 36804d150a9a
Shape hash: 37313317
"""
_shapes_config = "(T([4096, 2048], f32), T([2048], f32), T([32, 128, 2048], f32), T([32, 128, 1], f32), T([32, 128, 2048], f32), T([2048, 2048], f32), S([32, 128, 2048]), S([4096, 2048]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_353: "f32[4096, 2048]", primals_12: "f32[2048]", mul_2: "f32[32, 128, 2048]", div_73: "f32[32, 128, 1]", add_360: "f32[32, 128, 2048]", primals_10: "f32[2048, 2048]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        reshape_default: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_353, _shape_param_0);  mm_353 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(reshape_default, primals_12);  reshape_default = primals_12 = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 2048)
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, sum_dim_int_list_1);  mul_2 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_73, sub_tensor_1);  div_73 = sub_tensor_1 = None
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_360, mul_tensor_4);  add_360 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 2048], dtype=torch.float32, device='cuda'),
    [32, 128, 2048],  # _shape_param_0
    [4096, 2048],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
