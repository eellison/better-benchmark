"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_train
Pattern hash: 971a6a67ccc7
Shape hash: a7f4c6d2
"""
_shapes_config = "(T([16384, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512, 768], b8), T([768, 3072], f32), S([32, 512, 768]), S([32, 512, 768]), S([32, 512, 768]), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_130: "f32[16384, 768]", mul_527: "f32[32, 512, 768]", mm_132: "f32[16384, 768]", mm_134: "f32[16384, 768]", primals_23: "f32[768]", mul_18: "f32[32, 512, 768]", div_37: "f32[32, 512, 1]", gt_3: "b8[32, 512, 768]", primals_21: "f32[768, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_130, _shape_param_0);  mm_130 = _shape_param_0 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_527, reshape_default);  mul_527 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_132, _shape_param_1);  mm_132 = _shape_param_1 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_134, _shape_param_2);  mm_134 = _shape_param_2 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_23);  add_tensor_2 = primals_23 = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_18);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_dim_int_list_1);  mul_18 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_tensor_1);  div_37 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:400 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)



def make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [32, 512, 768],  # _shape_param_0
    [32, 512, 768],  # _shape_param_1
    [32, 512, 768],  # _shape_param_2
    [16384, 768],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
