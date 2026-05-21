"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_infer
Pattern hash: e69c3288347a
Shape hash: ad276e0f
"""
_shapes_config = "(T([1, 1, 768], f32), T([128, 768, 37, 37], f32, stride=(1051392, 1, 28416, 768)), T([1, 1370, 768], f32), T([768], f32), T([768], f32), T([2304, 768], f32), S([128, -1, -1]), S([128, 768, 1369]), S([175360, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[1, 1, 768]", convolution: "f32[128, 768, 37, 37]", arg4_1: "f32[1, 1370, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[2304, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        expand_default: "f32[128, 1, 768]" = torch.ops.aten.expand.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        reshape_default: "f32[128, 768, 1369]" = torch.ops.aten.reshape.default(convolution, _shape_param_1);  convolution = _shape_param_1 = None
        permute_default: "f32[128, 1369, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        cat_default: "f32[128, 1370, 768]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add_tensor: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(cat_default, arg4_1);  cat_default = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1370, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1370, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[128, 1370, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1370, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg5_1);  mul_tensor = arg5_1 = None
        add_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg6_1);  mul_tensor_1 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[175360, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default_1: "f32[768, 2304]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn(134578176, dtype=torch.float32, device='cuda').as_strided([128, 768, 37, 37], [1051392, 1, 28416, 768]),  # convolution
    torch.randn([1, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    [128, -1, -1],  # _shape_param_0
    [128, 768, 1369],  # _shape_param_1
    [175360, 768],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
