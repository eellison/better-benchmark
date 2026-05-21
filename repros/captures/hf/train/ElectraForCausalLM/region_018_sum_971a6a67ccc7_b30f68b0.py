"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train
Pattern hash: 971a6a67ccc7
Shape hash: b30f68b0
"""
_shapes_config = "(T([32768, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([256], f32), T([64, 512, 256], f32), T([64, 512, 1], f32), T([64, 512, 256], b8), T([256, 1024], f32), S([64, 512, 256]), S([64, 512, 256]), S([64, 512, 256]), S([32768, 256]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_130: "f32[32768, 256]", mul_526: "f32[64, 512, 256]", mm_132: "f32[32768, 256]", mm_134: "f32[32768, 256]", primals_26: "f32[256]", mul_17: "f32[64, 512, 256]", div_37: "f32[64, 512, 1]", gt_3: "b8[64, 512, 256]", primals_24: "f32[256, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_130, _shape_param_0);  mm_130 = _shape_param_0 = None
        add_tensor: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_526, reshape_default);  mul_526 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_132, _shape_param_1);  mm_132 = _shape_param_1 = None
        add_tensor_1: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_134, _shape_param_2);  mm_134 = _shape_param_2 = None
        add_tensor_2: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_26);  add_tensor_2 = primals_26 = None
        mul_tensor_1: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, 256)
        sum_dim_int_list: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_17);  mul_tensor = None
        sum_dim_int_list_1: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, sum_dim_int_list_1);  mul_17 = sum_dim_int_list_1 = None
        sub_tensor: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_tensor_1);  div_37 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_tensor_5: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_3: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        permute_default: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_default_1: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 256], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 256], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [64, 512, 256], dtype=torch.bool, device='cuda'),
    torch.randn([256, 1024], dtype=torch.float32, device='cuda'),
    [64, 512, 256],  # _shape_param_0
    [64, 512, 256],  # _shape_param_1
    [64, 512, 256],  # _shape_param_2
    [32768, 256],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
