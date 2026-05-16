"""
Standalone repro captured via capture_hook.
Label: falcon_rw_1b
Pattern hash: 287090d98877
Shape hash: 0034df1f
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_329: "f32[4, 512, 2048]", getitem_300: "f32[4, 512, 1]", getitem_299: "f32[4, 512, 1]", arg280_1: "f16[2048]", arg281_1: "f16[2048]", _shape_param_0, arg282_1: "f16[6144, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:598 in forward, code: attention_layernorm_out = self.input_layernorm(hidden_states)
        sub_tensor: "f32[4, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_329, getitem_300);  convert_element_type_329 = getitem_300 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_299, 1e-05);  getitem_299 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg280_1);  mul_tensor = arg280_1 = None
        add_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg281_1);  mul_tensor_1 = arg281_1 = None
        convert_element_type_default: "f16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:62 in forward, code: hidden_states = input @ self.weight.T
        reshape_default: "f16[2048, 2048]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_0);  convert_element_type_default = _shape_param_0 = None
        permute_default: "f16[2048, 6144]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([4, 512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048], dtype=torch.float16, device='cuda'),
    [2048, 2048],  # _shape_param_0
    torch.randn([6144, 2048], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
