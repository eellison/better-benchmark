"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: e12e3efe4fc3
Shape hash: 6cf78a9b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_93: "f32[6272, 1024]", inductor_seeds_default: "i64[46]", view_631: "f32[128, 7, 7, 1024]", primals_356: "f32[1024]", primals_357: "f32[1024]", primals_358: "f32[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(addmm_93, _shape_param_0);  addmm_93 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_2: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_default: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        reshape_default_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8999999985098839);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8999999985098839);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_3, div_tensor);  reshape_default_3 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_631, mul_tensor);  view_631 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_4: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_4, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 49, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 49, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(reshape_default_4, getitem_1);  reshape_default_4 = getitem_1 = None
        mul_tensor_1: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_356);  mul_tensor_1 = primals_356 = None
        add_tensor_2: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, primals_357);  mul_tensor_2 = primals_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_5: "f32[6272, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_5);  add_tensor_2 = _shape_param_5 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_358, [1, 0]);  primals_358 = None
        return (reshape_default_5, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([6272, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 46, [46], dtype=torch.int64, device='cuda'),
    torch.randn([128, 7, 7, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [128, 49, 1024],  # _shape_param_0
    [-1, 7, 7, 1024],  # _shape_param_1
    [-1, 1, 1, 7, 7, 1024],  # _shape_param_2
    [-1, 7, 7, 1024],  # _shape_param_3
    [128, -1, 1024],  # _shape_param_4
    [6272, 1024],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
