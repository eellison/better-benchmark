"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 34321a43139e
Shape hash: 5f8faa93
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_93: "f32[12544, 96]", _shape_param_0, primals_28: "f32[96]", mul_14: "f32[4, 56, 56, 96]", div_59: "f32[4, 56, 56, 1]", add_215: "f32[4, 56, 56, 96]", lt: "b8[4, 1, 1, 1]", _shape_param_1, _shape_param_2, _shape_param_3, primals_25: "f32[96, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 56, 56, 96]" = torch.ops.aten.reshape.default(mm_93, _shape_param_0);  mm_93 = _shape_param_0 = None
        mul_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(reshape_default, primals_28);  reshape_default = primals_28 = None
        mul_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, 96)
        sum_dim_int_list: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_14);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_14, sum_dim_int_list_1);  mul_14 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(div_59, sub_tensor_1);  div_59 = sub_tensor_1 = None
        add_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(add_215, mul_tensor_4);  add_215 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.float32);  lt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9818181818181818);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_5: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(add_tensor, div_tensor);  add_tensor = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:167 in shifted_window_attention, code: x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        iota_default: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[56]" = torch.ops.aten.add.Tensor(iota_default, 3);  iota_default = None
        fmod_scalar: "i64[56]" = torch.ops.aten.fmod.Scalar(add_tensor_1, 56);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:224 in shifted_window_attention, code: x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        index_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.index.Tensor(mul_tensor_5, [None, None, fmod_scalar]);  mul_tensor_5 = None
        index_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_scalar]);  index_tensor = fmod_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        reshape_default_1: "f32[4, 8, 7, 8, 7, 96]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_1);  index_tensor_1 = _shape_param_1 = None
        permute_default: "f32[4, 8, 8, 7, 7, 96]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        clone_default: "f32[4, 8, 8, 7, 7, 96]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[256, 49, 96]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default_3: "f32[12544, 96]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[96, 96]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_default_2: "f32[96, 96]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_3, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [4, 56, 56, 96],  # _shape_param_0
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    [4, 8, 7, 8, 7, 96],  # _shape_param_1
    [256, 49, 96],  # _shape_param_2
    [12544, 96],  # _shape_param_3
    torch.randn([96, 96], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
