"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: 3e144a38b47d
Shape hash: a1091d37
"""
_shapes_config = "(T([16384, 768], f32), T([32, 512, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512, 768, 2], f32), S([32, 512, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_50: "f32[16384, 768]", mul_428: "f32[32, 512, 768]", primals_11: "f32[768]", mul_4: "f32[32, 512, 768]", div_26: "f32[32, 512, 1]", full_default_5: "f32[32, 512, 768, 2]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_50, _shape_param_0);  mm_50 = _shape_param_0 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_428, reshape_default);  mul_428 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_11);  add_tensor = primals_11 = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_dim_int_list_1);  mul_4 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_tensor_1);  div_26 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_default: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_tensor_4, 3, 0);  full_default_5 = mul_tensor_4 = None
        return select_scatter_default



def make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768, 2], dtype=torch.float32, device='cuda'),
    [32, 512, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
