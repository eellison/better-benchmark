"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train
Pattern hash: aae7e91d6685
Shape hash: 9f6f0833
"""
_shapes_config = "(T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([4096], f32), T([1, 128, 4096], f32), T([1, 128, 1], f32), T([1, 128, 4096], f32), T([4096, 4096], f32), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([128, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_428: "f32[128, 4096]", mm_433: "f32[128, 4096]", mm_435: "f32[128, 4096]", mm_437: "f32[128, 4096]", primals_14: "f32[4096]", mul_10: "f32[1, 128, 4096]", div_112: "f32[1, 128, 1]", add_569: "f32[1, 128, 4096]", primals_9: "f32[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        reshape_default: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_428, _shape_param_0);  mm_428 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_1: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_433, _shape_param_1);  mm_433 = _shape_param_1 = None
        add_tensor: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_2: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_435, _shape_param_2);  mm_435 = _shape_param_2 = None
        add_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        reshape_default_3: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_437, _shape_param_3);  mm_437 = _shape_param_3 = None
        add_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_3);  add_tensor_1 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_14);  add_tensor_2 = primals_14 = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_10);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, sum_dim_int_list_1);  mul_10 = sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_112, sub_tensor_1);  div_112 = sub_tensor_1 = None
        add_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_569, mul_tensor_4);  add_569 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        reshape_default_4: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_4);  add_tensor_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_4, permute_default_1)



def make_inputs():
    return [
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [1, 128, 4096],  # _shape_param_0
    [1, 128, 4096],  # _shape_param_1
    [1, 128, 4096],  # _shape_param_2
    [1, 128, 4096],  # _shape_param_3
    [128, 4096],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
