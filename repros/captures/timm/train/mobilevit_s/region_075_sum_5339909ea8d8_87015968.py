"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 5339909ea8d8
Shape hash: 87015968
"""
_shapes_config = "(T([8192, 240], f32), T([240], f32), T([512, 16, 240], f32), T([512, 16, 1], f32), T([512, 16, 240], f32), S([512, 16, 240]), S([128, 4, 16, 240]), S([122880, 4, 2, 2]), S([128, 240, 8, 8]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_24: "f32[8192, 240]", primals_255: "f32[240]", mul_231: "f32[512, 16, 240]", div_41: "f32[512, 16, 1]", add_272: "f32[512, 16, 240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(mm_24, _shape_param_0);  mm_24 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(reshape_default, primals_255);  reshape_default = primals_255 = None
        mul_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, 240)
        sum_dim_int_list: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_231);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_231, sum_dim_int_list_1);  mul_231 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(div_41, sub_tensor_1);  div_41 = sub_tensor_1 = None
        add_tensor: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_272, mul_tensor_4);  add_272 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        reshape_default_1: "f32[128, 4, 16, 240]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[128, 240, 16, 4]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None
        clone_default: "f32[128, 240, 16, 4]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[122880, 4, 2, 2]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        permute_default_1: "f32[122880, 2, 4, 2]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_1: "f32[122880, 2, 4, 2]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[128, 240, 8, 8]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        return reshape_default_3



def make_inputs():
    return [
    torch.randn([8192, 240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 240], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 240], dtype=torch.float32, device='cuda'),
    [512, 16, 240],  # _shape_param_0
    [128, 4, 16, 240],  # _shape_param_1
    [122880, 4, 2, 2],  # _shape_param_2
    [128, 240, 8, 8],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
