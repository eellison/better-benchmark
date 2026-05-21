"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 1ce588e43051
Shape hash: 9aae7c7e
"""
_shapes_config = "(T([401408, 128], f32), T([128, 56, 56, 128], f32), T([128], f32), T([128], f32), T([512, 128], f32), S([8192, 49, 128]), S([-1, 7, 7, 128]), S([-1, 8, 8, 7, 7, 128]), S([-1, 56, 56, 128]), S([128, -1, 128]), S([401408, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f32[401408, 128]", add_1: "f32[128, 56, 56, 128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[512, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_0);  addmm_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(add_1, reshape_default_3);  add_1 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_4: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_4, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 3136, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 3136, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(reshape_default_4, getitem_1);  reshape_default_4 = getitem_1 = None
        mul_tensor: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_14);  mul_tensor = primals_14 = None
        add_tensor_2: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_15);  mul_tensor_1 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_5: "f32[401408, 128]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_5);  add_tensor_2 = _shape_param_5 = None
        permute_default_1: "f32[128, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        return (reshape_default_5, permute_default_1)



def make_inputs():
    return [
    torch.randn([401408, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 56, 56, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128], dtype=torch.float32, device='cuda'),
    [8192, 49, 128],  # _shape_param_0
    [-1, 7, 7, 128],  # _shape_param_1
    [-1, 8, 8, 7, 7, 128],  # _shape_param_2
    [-1, 56, 56, 128],  # _shape_param_3
    [128, -1, 128],  # _shape_param_4
    [401408, 128],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
