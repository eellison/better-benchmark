"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/alexnet/full_graph_000.py
Pattern hash: e06b1d1c4362
Shape hash: 7080dfbe
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 256, 13, 13]", arg1_1: "f32[4096]", arg2_1: "f32[4096, 9216]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        relu: "bf16[128, 256, 13, 13]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, _shape_param_0, _shape_param_1, [0, 0], [1, 1], False);  relu = _shape_param_0 = _shape_param_1 = None
        getitem: "bf16[128, 256, 6, 6]" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[128, 256, 6, 6]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = getitem_1 = None
        _adaptive_avg_pool2d: "bf16[128, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [6, 6]);  getitem = None
        view: "bf16[128, 9216]" = torch.ops.aten.view.default(_adaptive_avg_pool2d, _shape_param_2);  _adaptive_avg_pool2d = _shape_param_2 = None
        inductor_seeds: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[128, 9216]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type: "bf16[128, 9216]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[128, 9216]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.5);  convert_element_type = None
        mul: "bf16[128, 9216]" = torch.ops.aten.mul.Tensor(gt, view);  gt = view = None
        mul_1: "bf16[128, 9216]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        convert_element_type_1: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[4096, 9216]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        permute: "bf16[9216, 4096]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        addmm: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_1, mul_1, permute);  convert_element_type_1 = mul_1 = permute = None
        relu_1: "bf16[128, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_1: "f32[128, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_4, inductor_lookup_seed_1, 'rand');  _shape_param_4 = inductor_lookup_seed_1 = None
        convert_element_type_3: "bf16[128, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[128, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_3, 0.5);  convert_element_type_3 = None
        mul_2: "bf16[128, 4096]" = torch.ops.aten.mul.Tensor(gt_1, relu_1);  gt_1 = None
        mul_3: "bf16[128, 4096]" = torch.ops.aten.mul.Tensor(mul_2, 2.0);  mul_2 = None
        le: "b8[128, 4096]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        return (mul_3, le)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
