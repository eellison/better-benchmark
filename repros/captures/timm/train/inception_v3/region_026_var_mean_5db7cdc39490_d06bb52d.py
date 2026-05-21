"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 5db7cdc39490
Shape hash: d06bb52d
"""
_shapes_config = "(T([128, 64, 147, 147], f32, stride=(1382976, 1, 9408, 64)), T([64], f32), T([64], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_2: "f32[128, 64, 147, 147]", primals_18: "f32[64]", primals_19: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 64, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_1);  convolution_2 = getitem_1 = None
        mul_tensor: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 64, 147, 147]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 64, 147, 147]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:396 in forward_preaux, code: x = self.Pool1(x)  # N x 64 x 73 x 73
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem_2: "f32[128, 64, 73, 73]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_3: "i8[128, 64, 73, 73]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        return (getitem_2, getitem_3)



def make_inputs():
    return [
    torch.randn(177020928, dtype=torch.float32, device='cuda').as_strided([128, 64, 147, 147], [1382976, 1, 9408, 64]),  # convolution_2
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
