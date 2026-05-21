"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 39ef1a1a9c14
Shape hash: e88a9195
"""
_shapes_config = "(T([128, 144, 32, 32], f32, stride=(147456, 1, 4608, 144)), T([144], f32), T([144], f32), T([432, 144], f32), S([294912, 2, 16, 2]), S([128, 144, 256, 4]), S([512, 256, 144]), S([131072, 144]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_17: "f32[128, 144, 32, 32]", primals_105: "f32[144]", primals_106: "f32[144]", primals_107: "f32[432, 144]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_default: "f32[128, 144, 32, 32]" = torch.ops.aten.clone.default(convolution_17, memory_format = torch.contiguous_format);  convolution_17 = None
        reshape_default: "f32[294912, 2, 16, 2]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default: "f32[294912, 16, 2, 2]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_default_1: "f32[294912, 16, 2, 2]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 144, 256, 4]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        permute_default_1: "f32[128, 4, 256, 144]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None
        clone_default_2: "f32[128, 4, 256, 144]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_2);  clone_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(reshape_default_2, getitem_1);  reshape_default_2 = getitem_1 = None
        mul_tensor: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_105);  mul_tensor = primals_105 = None
        add_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_106);  mul_tensor_1 = primals_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_3: "f32[131072, 144]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_3);  add_tensor_1 = _shape_param_3 = None
        permute_default_2: "f32[144, 432]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        return (reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn(18874368, dtype=torch.float32, device='cuda').as_strided([128, 144, 32, 32], [147456, 1, 4608, 144]),  # convolution_17
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([432, 144], dtype=torch.float32, device='cuda'),
    [294912, 2, 16, 2],  # _shape_param_0
    [128, 144, 256, 4],  # _shape_param_1
    [512, 256, 144],  # _shape_param_2
    [131072, 144],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
