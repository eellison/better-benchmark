"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: 2aeefcdb8c7b
Shape hash: f81ded44
"""
_shapes_config = "(T([16, 3, 3, 3], f32, stride=(27, 1, 9, 3)), T([16, 1, 1, 1], f32), S([1, 16, 27]), S([16, 3, 3, 3]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[16, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        reshape_default: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_1: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg1_1, 0.34412564994580647);  arg1_1 = None
        reshape_default_1: "f32[16]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default);  mul_tensor = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return reshape_default_2



def make_inputs():
    return [
    torch.randn(432, dtype=torch.float32, device='cuda').as_strided([16, 3, 3, 3], [27, 1, 9, 3]),  # arg0_1
    torch.randn([16, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 16, 27],  # _shape_param_0
    [16, 3, 3, 3],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
