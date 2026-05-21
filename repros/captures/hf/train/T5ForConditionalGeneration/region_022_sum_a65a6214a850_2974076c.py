"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: a65a6214a850
Shape hash: 2974076c
"""
_shapes_config = "(T([8192, 512], f32), T([512], f32), T([8, 1024, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 512], f32), T([8, 1024, 512], b8), T([512, 512], f32), T([8192, 512], f32), T([512], f32), T([8, 1024, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 512], f32), T([8, 1024, 512], b8), T([512, 512], f32), S([8, 1024, 512]), S([8, 1024, 512]), S([8192, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_210: "f32[8192, 512]", primals_60: "f32[512]", add_48: "f32[8, 1024, 512]", rsqrt_14: "f32[8, 1024, 1]", add_153: "f32[8, 1024, 512]", gt_29: "b8[8, 1024, 512]", primals_59: "f32[512, 512]", mm_282: "f32[8192, 512]", primals_9: "f32[512]", add_9: "f32[8, 1024, 512]", rsqrt_1: "f32[8, 1024, 1]", add_198: "f32[8, 1024, 512]", gt_3: "b8[8, 1024, 512]", primals_8: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_210, _shape_param_0);  mm_210 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(reshape_default, primals_60);  reshape_default = primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, add_48)
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, rsqrt_14);  mul_tensor = None
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_153, mul_tensor_2);  add_153 = mul_tensor_2 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 1.0);  add_48 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_4);  add_tensor = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_5);  add_tensor_1 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[512, 512]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_282, _shape_param_3);  mm_282 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_9);  reshape_default_2 = primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_7, add_9)
        mul_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_7, rsqrt_1);  mul_tensor_7 = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True);  mul_tensor_8 = None
        add_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_198, mul_tensor_9);  add_198 = mul_tensor_9 = None
        pow_tensor_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_10: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default_1: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_10, _shape_param_4);  mul_tensor_10 = _shape_param_4 = None
        div_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default_1, 512);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 1.0);  add_9 = None
        mul_scalar_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_11);  add_tensor_2 = mul_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_default_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_tensor_12: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_13: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_12);  add_tensor_3 = mul_tensor_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_3: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_13, _shape_param_5);  mul_tensor_13 = _shape_param_5 = None
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_3: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_1, permute_default_1, reshape_default_3, permute_default_3)



def make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_0
    [8, 1024, 512],  # _shape_param_1
    [8192, 512],  # _shape_param_2
    [8, 1024, 512],  # _shape_param_3
    [8, 1024, 512],  # _shape_param_4
    [8192, 512],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
