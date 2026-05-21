"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: c04d3e42b115
Shape hash: 782ee32b
"""
_shapes_config = "(T([6272, 1024], f32), T([1024], f32), T([1024], f32), T([3072, 1024], f32), S([128, 7, 7, 1024]), S([128, 1, 7, 1, 7, 1024]), S([-1, 7, 7, 1024]), S([-1, 49, 1024]), S([6272, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[6272, 1024]", arg333_1: "f32[1024]", arg334_1: "f32[1024]", arg335_1: "f32[3072, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        reshape_default: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg333_1);  mul_tensor = arg333_1 = None
        add_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg334_1);  mul_tensor_1 = arg334_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        reshape_default_1: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_default: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None
        reshape_default_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_3: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default_4: "f32[6272, 1024]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_1: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        return (reshape_default_4, permute_default_1)



def make_inputs():
    return [
    torch.randn([6272, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 1024], dtype=torch.float32, device='cuda'),
    [128, 7, 7, 1024],  # _shape_param_0
    [128, 1, 7, 1, 7, 1024],  # _shape_param_1
    [-1, 7, 7, 1024],  # _shape_param_2
    [-1, 49, 1024],  # _shape_param_3
    [6272, 1024],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
