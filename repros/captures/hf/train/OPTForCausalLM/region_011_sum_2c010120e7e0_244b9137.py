"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 2c010120e7e0
Shape hash: 244b9137
"""
_shapes_config = "(T([8192, 768], f32), T([768], f32), T([8192, 768], f32), T([4, 2048, 768], b8), T([4, 2048, 768], f32), T([8192, 1], f32), T([8192, 1], f32), T([8192, 768], f32), T([768, 768], f32), S([4, 2048, 768]), S([-1, 768]), S([4, 2048, 768]), S([8192, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[8192, 768]", primals_13: "f32[768]", addmm_3: "f32[8192, 768]", gt: "b8[4, 2048, 768]", primals_3: "f32[4, 2048, 768]", getitem_7: "f32[8192, 1]", rsqrt_1: "f32[8192, 1]", view_14: "f32[8192, 768]", primals_11: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mm_2, primals_13);  mm_2 = primals_13 = None
        mul_tensor_1: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_0);  addmm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_tensor_2: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(gt, reshape_default);  reshape_default = None
        mul_tensor_3: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(primals_3, mul_tensor_3);  primals_3 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(reshape_default_1, getitem_7);  reshape_default_1 = getitem_7 = None
        mul_tensor_4: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = None
        mul_tensor_5: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [1], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  mul_tensor_4 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[8192, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        mul_tensor_7: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[8192, 768]" = torch.ops.aten.add.Tensor(view_14, mul_tensor_7);  view_14 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        reshape_default_2: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_2);  add_tensor_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_8: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_9: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(reshape_default_2, mul_tensor_8);  reshape_default_2 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_3: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_tensor_9, _shape_param_3);  mul_tensor_9 = _shape_param_3 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)



def make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 2048, 768], dtype=torch.bool, device='cuda'),
    torch.randn([4, 2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 2048, 768],  # _shape_param_0
    [-1, 768],  # _shape_param_1
    [4, 2048, 768],  # _shape_param_2
    [8192, 768],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
