"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: 2bb5337f43cc
Shape hash: 5b3c8e4d
"""
_shapes_config = "(T([128, 96, 112, 112], f32, stride=(1204224, 1, 10752, 96)), T([96], f32), T([96], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_5: "f32[128, 96, 112, 112]", primals_28: "f32[96]", primals_29: "f32[96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 96, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 96, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_1);  convolution_5 = getitem_1 = None
        mul_tensor: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1);  primals_29 = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 96, 112, 112]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 96, 112, 112]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 96, 112, 112]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 96, 112, 112]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 96, 113, 113]" = torch.ops.aten.constant_pad_nd.default(div_tensor, [0, 1, 0, 1], 0.0);  div_tensor = None
        return constant_pad_nd_default



def make_inputs():
    return [
    torch.randn(154140672, dtype=torch.float32, device='cuda').as_strided([128, 96, 112, 112], [1204224, 1, 10752, 96]),  # convolution_5
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
