"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: b1090f19279f
Shape hash: 567b86b6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_21: "f32[784, 384]", _shape_param_0, _shape_param_1, _shape_param_2, iota_8: "i64[14]", inductor_seeds_default: "i64[22]", add_55: "f32[4, 14, 14, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default: "f32[16, 49, 384]" = torch.ops.aten.reshape.default(addmm_21, _shape_param_0);  addmm_21 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_1: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        permute_default: "f32[4, 2, 7, 2, 7, 384]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None
        clone_default: "f32[4, 2, 7, 2, 7, 384]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[4, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:224 in shifted_window_attention, code: x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        add_tensor: "i64[14]" = torch.ops.aten.add.Tensor(iota_8, 11);  iota_8 = None
        fmod_scalar: "i64[14]" = torch.ops.aten.fmod.Scalar(add_tensor, 14);  add_tensor = None
        index_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(reshape_default_2, [None, fmod_scalar]);  reshape_default_2 = None
        index_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 1, 1, 1]" = torch.ops.prims.inductor_random.default([4, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[4, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9090909090909091);  inductor_random_default = None
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9090909090909091);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(index_tensor_1, div_tensor);  index_tensor_1 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        add_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_55, mul_tensor);  add_55 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [3], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[4, 14, 14, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 14, 14, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [16, 49, 384],  # _shape_param_0
    [4, 2, 2, 7, 7, 384],  # _shape_param_1
    [4, 14, 14, 384],  # _shape_param_2
    torch.randint(0, 2, [14], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [22], dtype=torch.int64, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
