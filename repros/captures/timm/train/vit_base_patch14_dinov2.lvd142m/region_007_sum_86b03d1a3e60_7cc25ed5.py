"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_train
Pattern hash: 86b03d1a3e60
Shape hash: 7cc25ed5
"""
_shapes_config = "(T([175360, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([1, 1370, 768], f32), T([128, 1370, 1], f32, stride=(1376, 1, 1)), T([128, 1370, 1], f32, stride=(1376, 1, 1)), T([128, 1370, 768], f32), S([128, 1370, 768]), S([128, 768, 37, 37]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_94: "f32[175360, 768]", primals_6: "f32[768]", cat: "f32[128, 1370, 768]", primals_5: "f32[1, 1370, 768]", getitem_1: "f32[128, 1370, 1]", rsqrt: "f32[128, 1370, 1]", add_133: "f32[128, 1370, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[128, 1370, 768]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_6);  reshape_default = primals_6 = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add_tensor: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(cat, primals_5);  cat = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 1370, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_5: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(add_133, mul_tensor_5);  add_133 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        slice_tensor: "f32[128, 1369, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 1, 1370);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_default: "f32[128, 768, 1369]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1]);  slice_tensor = None
        reshape_default_1: "f32[128, 768, 37, 37]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        return reshape_default_1



def make_inputs():
    return [
    torch.randn([175360, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn(176122, dtype=torch.float32, device='cuda').as_strided([128, 1370, 1], [1376, 1, 1]),  # getitem_1
    torch.randn(176122, dtype=torch.float32, device='cuda').as_strided([128, 1370, 1], [1376, 1, 1]),  # rsqrt
    torch.randn([128, 1370, 768], dtype=torch.float32, device='cuda'),
    [128, 1370, 768],  # _shape_param_0
    [128, 768, 37, 37],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
