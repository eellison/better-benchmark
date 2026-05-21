"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: aea5807f3176
Shape hash: 362dac1b
"""
_shapes_config = "(T([128, 240, 8, 8], f32, stride=(15360, 1, 1920, 240)), T([240], f32), T([512, 16, 240], f32), T([512, 16, 1], f32), T([240, 480], f32), S([122880, 2, 4, 2]), S([128, 240, 16, 4]), S([512, 16, 240]), S([8192, 240]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_175: "f32[128, 240, 8, 8]", primals_291: "f32[240]", mul_243: "f32[512, 16, 240]", div_35: "f32[512, 16, 1]", primals_289: "f32[240, 480]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_default: "f32[128, 240, 8, 8]" = torch.ops.aten.clone.default(getitem_175, memory_format = torch.contiguous_format);  getitem_175 = None
        reshape_default: "f32[122880, 2, 4, 2]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default: "f32[122880, 4, 2, 2]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_default_1: "f32[122880, 4, 2, 2]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 240, 16, 4]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        permute_default_1: "f32[128, 4, 16, 240]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_default_2: "f32[128, 4, 16, 240]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_2);  clone_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_tensor: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_291);  reshape_default_2 = primals_291 = None
        mul_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, 240)
        sum_dim_int_list: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_243);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_243, sum_dim_int_list_1);  mul_243 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(div_35, sub_tensor_1);  div_35 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_3: "f32[8192, 240]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_3);  mul_tensor_4 = _shape_param_3 = None
        permute_default_2: "f32[480, 240]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
        permute_default_3: "f32[240, 480]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_3, permute_default_3)



def make_inputs():
    return [
    torch.randn(1966080, dtype=torch.float32, device='cuda').as_strided([128, 240, 8, 8], [15360, 1, 1920, 240]),  # getitem_175
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 240], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240, 480], dtype=torch.float32, device='cuda'),
    [122880, 2, 4, 2],  # _shape_param_0
    [128, 240, 16, 4],  # _shape_param_1
    [512, 16, 240],  # _shape_param_2
    [8192, 240],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
