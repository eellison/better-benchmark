"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: 25898f81ca4d
Shape hash: 9a4e29e2
"""
_shapes_config = "(T([131072, 144], f32), T([512, 256, 144], f32), T([144], f32), T([144], f32), S([512, 256, 144]), S([128, 4, 256, -1]), S([294912, 16, 2, 2]), S([128, 144, 32, 32]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_7: "f32[131072, 144]", add_57: "f32[512, 256, 144]", arg111_1: "f32[144]", arg112_1: "f32[144]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(addmm_7, _shape_param_0);  addmm_7 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_tensor: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(add_57, reshape_default);  add_57 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_tensor, arg111_1);  mul_tensor = arg111_1 = None
        add_tensor_2: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg112_1);  mul_tensor_1 = arg112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        reshape_default_1: "f32[128, 4, 256, 144]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_default: "f32[128, 144, 256, 4]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None
        clone_default: "f32[128, 144, 256, 4]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[294912, 16, 2, 2]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_default_1: "f32[294912, 2, 16, 2]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_1: "f32[294912, 2, 16, 2]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[128, 144, 32, 32]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        return reshape_default_3



def make_inputs():
    return [
    torch.randn([131072, 144], dtype=torch.float32, device='cuda'),
    torch.randn([512, 256, 144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    [512, 256, 144],  # _shape_param_0
    [128, 4, 256, -1],  # _shape_param_1
    [294912, 16, 2, 2],  # _shape_param_2
    [128, 144, 32, 32],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
