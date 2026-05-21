"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: d31a434204b0
Shape hash: ad300cb4
"""
_shapes_config = "(T([128, 32, 112, 112], f32, stride=(401408, 1, 3584, 32)), T([64, 32, 3, 3], f32, stride=(288, 1, 96, 32)), T([64, 1, 1, 1], f32), S([1, 64, 288]), S([64, 32, 3, 3]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_1: "f32[128, 32, 112, 112]", primals_8: "f32[64, 32, 3, 3]", primals_9: "f32[64, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        neg_default: "f32[128, 32, 112, 112]" = torch.ops.aten.neg.default(convolution_1)
        exp_default: "f32[128, 32, 112, 112]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Tensor(convolution_1, add_tensor);  convolution_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default: "f32[64, 32, 3, 3]" = torch.ops.aten.clone.default(primals_8, memory_format = torch.contiguous_format);  primals_8 = None
        reshape_default: "f32[1, 64, 288]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_9, 0.10536653122135592);  primals_9 = None
        reshape_default_1: "f32[64]" = torch.ops.aten.reshape.default(mul_tensor, [-1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        mul_tensor_1: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default);  mul_tensor_1 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[64, 32, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return (div_tensor, reshape_default_2)



def make_inputs():
    return [
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([128, 32, 112, 112], [401408, 1, 3584, 32]),  # convolution_1
    torch.randn(18432, dtype=torch.float32, device='cuda').as_strided([64, 32, 3, 3], [288, 1, 96, 32]),  # primals_8
    torch.randn([64, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 64, 288],  # _shape_param_0
    [64, 32, 3, 3],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
