"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 670d2df8b892
Shape hash: 5cfcc2df
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f16[4096, 768]", inductor_seeds: "i64[25]", add_92: "f32[4, 1024, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 768]" = torch.ops.aten.reshape.default(addmm_47, [4, 1024, 768]);  addmm_47 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24);  inductor_seeds = None
        inductor_random_default: "f32[4, 1024, 768]" = torch.ops.prims.inductor_random.default([4, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[4, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[4, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f16[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_92, mul_tensor_1);  add_92 = mul_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[4, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [25], dtype=torch.int64, device='cuda'),
    torch.randn([4, 1024, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
