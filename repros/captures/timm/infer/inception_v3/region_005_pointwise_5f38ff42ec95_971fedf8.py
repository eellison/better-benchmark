"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_infer
Pattern hash: 5f38ff42ec95
Shape hash: 971fedf8
"""
_shapes_config = "(T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)), T([320], f32), T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([320], f32), T([320], f32), T([320], f32), T([192], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([192], f32), T([192], f32), T([192], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, cat_7: "f32[128, 768, 17, 17]", arg357_1: "f32[320]", convolution_71: "f32[128, 320, 8, 8]", arg358_1: "f32[320]", arg359_1: "f32[320]", arg360_1: "f32[320]", arg377_1: "f32[192]", convolution_75: "f32[128, 192, 8, 8]", arg378_1: "f32[192]", arg379_1: "f32[192]", arg380_1: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:184 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False);  cat_7 = None
        getitem: "f32[128, 768, 8, 8]" = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_default_1);  convolution_71 = unsqueeze_default_1 = None
        add_tensor: "f32[320]" = torch.ops.aten.add.Tensor(arg358_1, 0.001);  arg358_1 = None
        sqrt_default: "f32[320]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_default_5: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_default_7: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_default_9);  convolution_75 = unsqueeze_default_9 = None
        add_tensor_2: "f32[192]" = torch.ops.aten.add.Tensor(arg378_1, 0.001);  arg378_1 = None
        sqrt_default_1: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_default_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_default_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:190 in forward, code: return torch.cat(outputs, 1)
        cat_default: "f32[128, 1280, 8, 8]" = torch.ops.aten.cat.default([relu_default, relu_default_1, getitem], 1);  relu_default = relu_default_1 = getitem = None
        return cat_default



def make_inputs():
    return [
    torch.randn(28409856, dtype=torch.float32, device='cuda').as_strided([128, 768, 17, 17], [221952, 1, 13056, 768]),  # cat_7
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn(2621440, dtype=torch.float32, device='cuda').as_strided([128, 320, 8, 8], [20480, 1, 2560, 320]),  # convolution_71
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(1572864, dtype=torch.float32, device='cuda').as_strided([128, 192, 8, 8], [12288, 1, 1536, 192]),  # convolution_75
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
