"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 98a34a569c29
Shape hash: 878228aa
"""
_shapes_config = "(T([4096, 512], f32), T([512], f32), T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], f32), T([32, 128, 512], b8), T([512, 384], f32), S([32, 128, 512]), S([32, 128, 512]), S([4096, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_292: "f32[4096, 512]", primals_98: "f32[512]", add_85: "f32[32, 128, 512]", rsqrt_21: "f32[32, 128, 1]", add_243: "f32[32, 128, 512]", gt_43: "b8[32, 128, 512]", primals_97: "f32[512, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_292, _shape_param_0);  mm_292 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(reshape_default, primals_98);  reshape_default = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, add_85)
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, rsqrt_21);  mul_tensor = None
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_243, mul_tensor_2);  add_243 = mul_tensor_2 = None
        pow_tensor_scalar: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
        mul_scalar: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        div_scalar: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_85, 1.0);  add_85 = None
        mul_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_4);  add_tensor = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_default: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_tensor_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_5);  add_tensor_1 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        reshape_default_1: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[384, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512, 384], dtype=torch.float32, device='cuda'),
    [32, 128, 512],  # _shape_param_0
    [32, 128, 512],  # _shape_param_1
    [4096, 512],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
