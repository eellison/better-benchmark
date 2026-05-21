"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train
Pattern hash: 0fce618f0331
Shape hash: d936a7d0
"""
_shapes_config = "(T([128, 4096], f32), T([4096], f32), T([1, 128, 4096], f32), T([1, 128, 1], f32), T([4096, 4096], f32), S([1, 128, 4096]), S([128, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_112: "f32[128, 4096]", primals_311: "f32[4096]", mul_280: "f32[1, 128, 4096]", div_58: "f32[1, 128, 1]", primals_306: "f32[4096, 4096]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:623 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_112, _shape_param_0);  mm_112 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(reshape_default, primals_311);  reshape_default = primals_311 = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_280);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, sum_dim_int_list_1);  mul_280 = sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_58, sub_tensor_1);  div_58 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        reshape_default_1: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [1, 128, 4096],  # _shape_param_0
    [128, 4096],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
