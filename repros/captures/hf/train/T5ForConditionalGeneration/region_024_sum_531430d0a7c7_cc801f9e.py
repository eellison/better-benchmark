"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 531430d0a7c7
Shape hash: cc801f9e
"""
_shapes_config = "(T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([512], f32), T([8, 1024, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 512], f32), T([8, 1024, 512], b8), T([512, 2048], f32), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_274: "f32[8192, 512]", mm_276: "f32[8192, 512]", mm_278: "f32[8192, 512]", primals_12: "f32[512]", add_11: "f32[8, 1024, 512]", rsqrt_2: "f32[8, 1024, 1]", add_193: "f32[8, 1024, 512]", gt_5: "b8[8, 1024, 512]", primals_11: "f32[512, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_274, _shape_param_0);  mm_274 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_276, _shape_param_1);  mm_276 = _shape_param_1 = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_278, _shape_param_2);  mm_278 = _shape_param_2 = None
        add_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_12);  add_tensor_1 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, add_11)
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, rsqrt_2);  mul_tensor = None
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        add_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_193, mul_tensor_2);  add_193 = mul_tensor_2 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_3);  mul_tensor_3 = _shape_param_3 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_4);  add_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_5);  add_tensor_3 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_3: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_4);  mul_tensor_6 = _shape_param_4 = None
        permute_default: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_1: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)



def make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_0
    [8, 1024, 512],  # _shape_param_1
    [8, 1024, 512],  # _shape_param_2
    [8, 1024, 512],  # _shape_param_3
    [8192, 512],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
