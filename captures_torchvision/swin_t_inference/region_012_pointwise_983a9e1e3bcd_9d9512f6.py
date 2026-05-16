"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 983a9e1e3bcd
Shape hash: 9d9512f6
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_38: "f32[196, 1536]", _shape_param_0, _shape_param_1, arg149_1: "f32[384, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[1, 14, 14, 1536]" = torch.ops.aten.reshape.default(addmm_38, _shape_param_0);  addmm_38 = _shape_param_0 = None
        mul_tensor: "f32[1, 14, 14, 1536]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[1, 14, 14, 1536]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[1, 14, 14, 1536]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[1, 14, 14, 1536]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[1, 14, 14, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        reshape_default_1: "f32[196, 1536]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[1536, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([196, 1536], dtype=torch.float32, device='cuda'),
    [1, 14, 14, 1536],  # _shape_param_0
    [196, 1536],  # _shape_param_1
    torch.randn([384, 1536], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
