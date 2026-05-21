"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: c7b3c08de59d
Shape hash: 0d619867
"""
_shapes_config = "(T([25088, 512], f32), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), T([2048, 512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_85: "f32[25088, 512]", view_573: "f32[128, 14, 14, 512]", arg324_1: "f32[512]", arg325_1: "f32[512]", arg326_1: "f32[2048, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_85, _shape_param_0);  addmm_85 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_1: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_2: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_default: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[14]" = torch.ops.aten.add.Tensor(iota_default, 11);  iota_default = None
        fmod_scalar: "i64[14]" = torch.ops.aten.fmod.Scalar(add_tensor, 14);  add_tensor = None
        index_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(reshape_default_3, [None, fmod_scalar]);  reshape_default_3 = fmod_scalar = None
        iota_default_1: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[14]" = torch.ops.aten.add.Tensor(iota_default_1, 11);  iota_default_1 = None
        fmod_scalar_1: "i64[14]" = torch.ops.aten.fmod.Scalar(add_tensor_1, 14);  add_tensor_1 = None
        index_tensor_1: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar_1]);  index_tensor = fmod_scalar_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_tensor_2: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_573, index_tensor_1);  view_573 = index_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_4: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_4);  add_tensor_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_4, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 196, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 196, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(reshape_default_4, getitem_1);  reshape_default_4 = getitem_1 = None
        add_tensor_3: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg324_1);  mul_tensor = arg324_1 = None
        add_tensor_4: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg325_1);  mul_tensor_1 = arg325_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_5: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_5);  add_tensor_4 = _shape_param_5 = None
        permute_default_1: "f32[512, 2048]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        return (reshape_default_5, permute_default_1)



def make_inputs():
    return [
    torch.randn([25088, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    [512, 49, 512],  # _shape_param_0
    [-1, 7, 7, 512],  # _shape_param_1
    [-1, 2, 2, 7, 7, 512],  # _shape_param_2
    [-1, 14, 14, 512],  # _shape_param_3
    [128, -1, 512],  # _shape_param_4
    [25088, 512],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
