"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_train_000
Pattern hash: 9af0513b0d3e
Shape hash: 9930810c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 768], f32), T([25], i64), T([4, 512, 768], f32), T([768], f32), T([768], f32), S([4, 512, 768]), S([-1, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_43: "f32[2048, 768]", inductor_seeds: "i64[25]", add_84: "f32[4, 512, 768]", arg135_1: "f32[768]", arg136_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_43, _shape_param_0);  addmm_43 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22);  inductor_seeds = None
        inductor_random_default: "f32[4, 512, 768]" = torch.ops.prims.inductor_random.default([4, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_84, mul_tensor_1);  add_84 = mul_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg135_1);  mul_tensor_2 = arg135_1 = None
        add_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg136_1);  mul_tensor_3 = arg136_1 = None
        view_default_1: "f32[2048, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 2048]" = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        div_tensor: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (permute_default, div_tensor)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
