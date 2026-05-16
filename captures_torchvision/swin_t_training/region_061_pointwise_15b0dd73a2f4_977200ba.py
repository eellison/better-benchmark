"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 15b0dd73a2f4
Shape hash: 977200ba
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_99: "f32[12544, 384]", _shape_param_0, addmm_2: "f32[12544, 384]", _shape_param_1, _shape_param_2, primals_16: "f32[384, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 56, 56, 384]" = torch.ops.aten.reshape.default(mm_99, _shape_param_0);  mm_99 = _shape_param_0 = None
        reshape_default_1: "f32[4, 56, 56, 384]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_1);  addmm_2 = _shape_param_1 = None
        mul_tensor: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[4, 56, 56, 384]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[4, 56, 56, 384]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_3: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[4, 56, 56, 384]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_4);  reshape_default_1 = mul_tensor_4 = None
        add_tensor_1: "f32[4, 56, 56, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[4, 56, 56, 384]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None
        reshape_default_2: "f32[12544, 384]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[96, 384]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_1: "f32[384, 96]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([12544, 384], dtype=torch.float32, device='cuda'),
    [4, 56, 56, 384],  # _shape_param_0
    torch.randn([12544, 384], dtype=torch.float32, device='cuda'),
    [4, 56, 56, 384],  # _shape_param_1
    [12544, 384],  # _shape_param_2
    torch.randn([384, 96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
