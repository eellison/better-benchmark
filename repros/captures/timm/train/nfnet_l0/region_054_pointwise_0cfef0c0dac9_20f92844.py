"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 0cfef0c0dac9
Shape hash: 20f92844
"""
_shapes_config = "(T([128, 384, 7, 7], f32, stride=(18816, 1, 2688, 384)), T([128, 384, 7, 7], f32, stride=(18816, 1, 2688, 384)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_63: "f32[128, 384, 7, 7]", getitem_162: "f32[128, 384, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_default: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_63)
        exp_default: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 384, 7, 7]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 384, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[128, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_162, mul_tensor);  getitem_162 = None
        sub_tensor: "f32[128, 384, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[128, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_63, sub_tensor);  convolution_63 = sub_tensor = None
        add_tensor_1: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 384, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        return mul_tensor_3



def make_inputs():
    return [
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([128, 384, 7, 7], [18816, 1, 2688, 384]),  # convolution_63
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([128, 384, 7, 7], [18816, 1, 2688, 384]),  # getitem_162
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
