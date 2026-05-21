"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_009
Pattern hash: 4f2b9b838832
Shape hash: 793d218a
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
_shapes_config = "(T([8, 4096, 512], f32), T([512], f32), T([512], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[8, 4096, 512]", arg0_1: "f32[512]", arg1_1: "f32[512]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 4096, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[8, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  arg2_1 = getitem_1 = None
        mul_tensor: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg0_1);  mul_tensor = arg0_1 = None
        add_tensor_1: "f32[8, 4096, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1_1);  mul_tensor_1 = arg1_1 = None
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 4096, 512]" = torch.ops.prims.inductor_random.default([8, 4096, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 4096, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None
        mul_tensor_2: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_3: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.0526315789473684);  mul_tensor_2 = None
        return mul_tensor_3



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
