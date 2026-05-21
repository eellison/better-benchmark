"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: f715055edb02
Shape hash: e16f12eb
"""
_shapes_config = "(T([100352, 256], f32), T([46], i64), T([128, 784, 256], f32), T([256], f32), T([256], f32), T([768, 256], f32), S([128, 784, 256]), S([128, 28, 28, 256]), S([128, 4, 7, 4, 7, 256]), S([-1, 7, 7, 256]), S([-1, 49, 256]), S([100352, 256]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_11: "f32[100352, 256]", inductor_seeds_default: "i64[46]", view_78: "f32[128, 784, 256]", primals_52: "f32[256]", primals_53: "f32[256]", primals_55: "f32[768, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(addmm_11, _shape_param_0);  addmm_11 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9913043472915888);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9913043472915888);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(reshape_default, div_tensor);  reshape_default = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(view_78, mul_tensor);  view_78 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_1: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_1, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 28, 28, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 28, 28, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(reshape_default_1, getitem_1);  reshape_default_1 = getitem_1 = None
        mul_tensor_1: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_52);  mul_tensor_1 = primals_52 = None
        add_tensor_2: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_tensor_2, primals_53);  mul_tensor_2 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_default: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_3: "i64[28]" = torch.ops.aten.add.Tensor(iota_default, 3);  iota_default = None
        fmod_scalar: "i64[28]" = torch.ops.aten.fmod.Scalar(add_tensor_3, 28);  add_tensor_3 = None
        index_tensor: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(add_tensor_2, [None, fmod_scalar]);  add_tensor_2 = None
        index_tensor_1: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        reshape_default_2: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_2);  index_tensor_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_default: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_4: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default_5: "f32[100352, 256]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_1: "f32[256, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        return (reshape_default_5, permute_default_1)



def make_inputs():
    return [
    torch.randn([100352, 256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 46, [46], dtype=torch.int64, device='cuda'),
    torch.randn([128, 784, 256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([768, 256], dtype=torch.float32, device='cuda'),
    [128, 784, 256],  # _shape_param_0
    [128, 28, 28, 256],  # _shape_param_1
    [128, 4, 7, 4, 7, 256],  # _shape_param_2
    [-1, 7, 7, 256],  # _shape_param_3
    [-1, 49, 256],  # _shape_param_4
    [100352, 256],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
