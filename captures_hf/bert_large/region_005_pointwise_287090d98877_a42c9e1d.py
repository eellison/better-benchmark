"""
Standalone repro captured via capture_hook.
Label: bert_large
Pattern hash: 287090d98877
Shape hash: a42c9e1d
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_567: "f32[4, 512, 1024]", getitem_311: "f32[4, 512, 1]", getitem_310: "f32[4, 512, 1]", arg385_1: "f16[1024]", arg386_1: "f16[1024]", _shape_param_0, arg387_1: "f16[4096, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_tensor: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_567, getitem_311);  convert_element_type_567 = getitem_311 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-12);  getitem_310 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg385_1);  mul_tensor = arg385_1 = None
        add_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg386_1);  mul_tensor_1 = arg386_1 = None
        convert_element_type_default: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f16[2048, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_0);  convert_element_type_default = _shape_param_0 = None
        permute_default: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    [2048, 1024],  # _shape_param_0
    torch.randn([4096, 1024], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
