"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_infer
Pattern hash: e5a8078b814d
Shape hash: dae27bc8
"""
_shapes_config = "(T([192], f32), T([128, 192, 71, 71], f32, stride=(967872, 1, 13632, 192)), T([192], f32), T([192], f32), T([192], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg22_1: "f32[192]", convolution_4: "f32[128, 192, 71, 71]", arg23_1: "f32[192]", arg24_1: "f32[192]", arg25_1: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_default_1);  convolution_4 = unsqueeze_default_1 = None
        add_tensor: "f32[192]" = torch.ops.aten.add.Tensor(arg23_1, 0.001);  arg23_1 = None
        sqrt_default: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 192, 71, 71]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 192, 71, 71]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:399 in forward_preaux, code: x = self.Pool2(x)  # N x 192 x 35 x 35
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f32[128, 192, 35, 35]" = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None
        return getitem



def make_inputs():
    return [
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(123887616, dtype=torch.float32, device='cuda').as_strided([128, 192, 71, 71], [967872, 1, 13632, 192]),  # convolution_4
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
