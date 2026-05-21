"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: 76ea81febb54
Shape hash: 57c1baec
"""
_shapes_config = "(T([8192, 1024], f32), T([1024], f32), T([16, 512, 1024], f32), T([16, 512, 1], f32), T([16, 512, 1024], b8), T([1024, 4096], f32), S([16, 512, 1024]), S([8192, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[8192, 1024]", primals_391: "f32[1024]", mul_315: "f32[16, 512, 1024]", div_51: "f32[16, 512, 1]", gt_72: "b8[16, 512, 1024]", primals_389: "f32[1024, 4096]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, primals_391);  reshape_default = primals_391 = None
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_315);  mul_tensor = None
        sum_dim_int_list_1: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_315, sum_dim_int_list_1);  mul_315 = sum_dim_int_list_1 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_51, sub_tensor_1);  div_51 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_tensor_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_389, [1, 0]);  primals_389 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [16, 512, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    [16, 512, 1024],  # _shape_param_0
    [8192, 1024],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
