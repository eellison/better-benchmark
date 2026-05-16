"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_training
Pattern hash: ca1416b299f7
Shape hash: 625ee890
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_default_216: "f32[4096, 1024]", _shape_param_0, _shape_param_1, inductor_seeds_default: "i64[99]", add_252: "f32[512, 8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[1, 4096, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_216, 0);  mm_default_216 = None
        reshape_default: "f32[512, 8, 1, 1, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "f32[512, 8, 1024, 1, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 4, 2, 3]);  reshape_default = None
        reshape_default_1: "f32[512, 8, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 95);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 8, 1024]" = torch.ops.prims.inductor_random.default([512, 8, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 8, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default_1);  gt_scalar = reshape_default_1 = None
        mul_tensor_1: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_tensor: "f32[512, 8, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_252);  mul_tensor_1 = add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[512, 8, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 8, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [512, 8, 1, 1, 1024],  # _shape_param_0
    [512, 8, 1024],  # _shape_param_1
    torch.randint(0, 2, [99], dtype=torch.int64, device='cuda'),
    torch.randn([512, 8, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
