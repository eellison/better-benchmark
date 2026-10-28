"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/mobilenet_v3_large/full_graph_000.py
Pattern hash: 813c189afca5
Shape hash: 437d7762
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
    def forward(self, arg0_1: "f32[1280]", arg1_1: "f32[1280, 960]", arg2_1: "bf16[32, 960]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1280]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[1280, 960]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[960, 1280]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[32, 1280]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, permute);  convert_element_type = arg2_1 = permute = None
        convert_element_type_2: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        add: "f32[32, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_2, 3)
        clamp_min: "f32[32, 1280]" = torch.ops.aten.clamp_min.default(add, 0);  add = None
        clamp_max: "f32[32, 1280]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_2, clamp_max);  convert_element_type_2 = clamp_max = None
        div: "f32[32, 1280]" = torch.ops.aten.div.Tensor(mul, 6);  mul = None
        convert_element_type_3: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 1280]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(inductor_random, 0.8);  inductor_random = None
        convert_element_type_4: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div_1: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_4, 0.8);  convert_element_type_4 = None
        mul_1: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_3, div_1);  convert_element_type_3 = div_1 = None
        return mul_1



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
