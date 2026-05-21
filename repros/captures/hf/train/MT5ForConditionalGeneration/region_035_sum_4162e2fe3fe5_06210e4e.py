"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 4162e2fe3fe5
Shape hash: 06210e4e
"""
_shapes_config = "(T([4096, 512], f32), T([32, 128, 512], b8), T([512], f32), T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], b8), T([512, 1024], f32), S([32, 128, 512]), S([32, 128, 512]), S([4096, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_146: "f32[4096, 512]", gt_84: "b8[32, 128, 512]", primals_191: "f32[512]", add_152: "f32[32, 128, 512]", rsqrt_41: "f32[32, 128, 1]", gt_83: "b8[32, 128, 512]", primals_190: "f32[512, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        reshape_default: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_146, _shape_param_0);  mm_146 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:766 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_84, torch.float32);  gt_84 = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_191);  mul_tensor_1 = primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_152)
        mul_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_41);  mul_tensor_2 = None
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_41, 3);  rsqrt_41 = None
        mul_scalar: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_5: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_1);  mul_tensor_5 = _shape_param_1 = None
        div_scalar: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_152, 1.0);  add_152 = None
        mul_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_83, torch.float32);  gt_83 = None
        mul_tensor_7: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_7);  add_tensor = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_1: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_2);  mul_tensor_8 = _shape_param_2 = None
        permute_default: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_default_1: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512, 1024], dtype=torch.float32, device='cuda'),
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
