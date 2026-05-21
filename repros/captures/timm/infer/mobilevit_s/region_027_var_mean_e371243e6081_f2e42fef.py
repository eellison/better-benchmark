"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: e371243e6081
Shape hash: f2e42fef
"""
_shapes_config = "(T([131072, 144], f32), T([512, 256, 144], f32), T([144], f32), T([144], f32), T([288, 144], f32), S([512, 256, 144]), S([131072, 144]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[131072, 144]", add_54: "f32[512, 256, 144]", arg105_1: "f32[144]", arg106_1: "f32[144]", arg107_1: "f32[288, 144]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_tensor: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(add_54, reshape_default);  add_54 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_tensor, arg105_1);  mul_tensor = arg105_1 = None
        add_tensor_2: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg106_1);  mul_tensor_1 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[131072, 144]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[144, 288]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([131072, 144], dtype=torch.float32, device='cuda'),
    torch.randn([512, 256, 144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([288, 144], dtype=torch.float32, device='cuda'),
    [512, 256, 144],  # _shape_param_0
    [131072, 144],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
