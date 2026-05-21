"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 5f5c323494f9
Shape hash: 6b937932
"""
_shapes_config = "(T([6272, 4096], f32), T([6272, 4096], f32), T([4096, 1024], f32), S([128, 49, 4096]), S([128, 49, 4096]), S([6272, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_13: "f32[6272, 4096]", addmm_90: "f32[6272, 4096]", primals_344: "f32[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(mm_13, _shape_param_0);  mm_13 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(addmm_90, _shape_param_1);  addmm_90 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_tensor: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_3: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[128, 49, 4096]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_4);  reshape_default_1 = mul_tensor_4 = None
        add_tensor_1: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_2: "f32[6272, 4096]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_344, [1, 0]);  primals_344 = None
        permute_default_1: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([6272, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([6272, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [128, 49, 4096],  # _shape_param_0
    [128, 49, 4096],  # _shape_param_1
    [6272, 4096],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
