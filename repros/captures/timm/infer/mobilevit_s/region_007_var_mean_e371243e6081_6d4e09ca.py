"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: e371243e6081
Shape hash: 6d4e09ca
"""
_shapes_config = "(T([8192, 240], f32), T([512, 16, 240], f32), T([240], f32), T([240], f32), T([720, 240], f32), S([512, 16, 240]), S([8192, 240]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_31: "f32[8192, 240]", add_137: "f32[512, 16, 240]", arg249_1: "f32[240]", arg250_1: "f32[240]", arg251_1: "f32[720, 240]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(addmm_31, _shape_param_0);  addmm_31 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_tensor: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_137, reshape_default);  add_137 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, arg249_1);  mul_tensor = arg249_1 = None
        add_tensor_2: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg250_1);  mul_tensor_1 = arg250_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[240, 720]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([8192, 240], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([720, 240], dtype=torch.float32, device='cuda'),
    [512, 16, 240],  # _shape_param_0
    [8192, 240],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
