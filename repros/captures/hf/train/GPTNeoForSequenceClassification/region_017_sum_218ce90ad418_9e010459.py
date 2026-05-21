"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train
Pattern hash: 218ce90ad418
Shape hash: 9e010459
"""
_shapes_config = "(T([4096, 2048], f32), T([4096, 2048], f32), T([4096, 2048], f32), T([2048], f32), T([32, 128, 2048], f32), T([32, 128, 1], f32), T([32, 128, 2048], f32), T([2048, 8192], f32), S([32, 128, 2048]), S([32, 128, 2048]), S([32, 128, 2048]), S([4096, 2048]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_346: "f32[4096, 2048]", mm_348: "f32[4096, 2048]", mm_350: "f32[4096, 2048]", primals_18: "f32[2048]", mul_8: "f32[32, 128, 2048]", div_72: "f32[32, 128, 1]", add_357: "f32[32, 128, 2048]", primals_16: "f32[2048, 8192]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        reshape_default: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_346, _shape_param_0);  mm_346 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_1: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_348, _shape_param_1);  mm_348 = _shape_param_1 = None
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        reshape_default_2: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_350, _shape_param_2);  mm_350 = _shape_param_2 = None
        add_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_18);  add_tensor_1 = primals_18 = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 2048)
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_8);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, sum_dim_int_list_1);  mul_8 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_72, sub_tensor_1);  div_72 = sub_tensor_1 = None
        add_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_357, mul_tensor_4);  add_357 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        reshape_default_3: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_1: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)



def make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 8192], dtype=torch.float32, device='cuda'),
    [32, 128, 2048],  # _shape_param_0
    [32, 128, 2048],  # _shape_param_1
    [32, 128, 2048],  # _shape_param_2
    [4096, 2048],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
