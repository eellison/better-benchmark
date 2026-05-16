"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 983a9e1e3bcd
Shape hash: 1625e9ef
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_6: "f32[12544, 384]", _shape_param_0, _shape_param_1, primals_32: "f32[96, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 56, 56, 384]" = torch.ops.aten.reshape.default(addmm_6, _shape_param_0);  addmm_6 = _shape_param_0 = None
        mul_tensor: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[4, 56, 56, 384]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[4, 56, 56, 384]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        reshape_default_1: "f32[12544, 384]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[384, 96]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([12544, 384], dtype=torch.float32, device='cuda'),
    [4, 56, 56, 384],  # _shape_param_0
    [12544, 384],  # _shape_param_1
    torch.randn([96, 384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
