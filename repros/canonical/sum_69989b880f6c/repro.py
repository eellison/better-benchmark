"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: 69989b880f6c
Shape hash: 764a0dde
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
    def forward(self, arg0_1: "f32[4, 5, 2, 426888]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        inductor_seeds: "i64[4]" = torch.ops.prims.inductor_seeds.default(4, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random: "f32[1, 4, 4, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        sort = torch.ops.aten.sort.default(inductor_random, 1);  inductor_random = None
        getitem: "f32[1, 4, 4, 1, 1]" = sort[0];  getitem = None
        getitem_1: "i64[1, 4, 4, 1, 1]" = sort[1];  sort = None
        slice_1: "f32[4, 4, 2, 426888]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807);  arg0_1 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_randint: "i64[4, 4, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 2, _shape_param_1, inductor_lookup_seed_1);  _shape_param_1 = inductor_lookup_seed_1 = None
        convert_element_type: "f32[4, 4, 1, 1]" = torch.ops.prims.convert_element_type.default(inductor_randint, torch.float32);  inductor_randint = None
        mul: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type, 2);  convert_element_type = None
        sub: "f32[4, 4, 1, 1]" = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        mul_1: "f32[4, 4, 2, 426888]" = torch.ops.aten.mul.Tensor(slice_1, sub);  slice_1 = sub = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_randint_1: "i64[4, 4, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 2, _shape_param_2, inductor_lookup_seed_2);  _shape_param_2 = inductor_lookup_seed_2 = None
        expand: "i64[4, 4, 1, 426888]" = torch.ops.aten.expand.default(inductor_randint_1, _shape_param_3);  inductor_randint_1 = _shape_param_3 = None
        gather: "f32[4, 4, 1, 426888]" = torch.ops.aten.gather.default(mul_1, 2, expand)
        sub_1: "i64[4, 4, 1, 426888]" = torch.ops.aten.sub.Tensor(1, expand);  expand = None
        gather_1: "f32[4, 4, 1, 426888]" = torch.ops.aten.gather.default(mul_1, 2, sub_1);  mul_1 = sub_1 = None
        cat: "f32[4, 4, 2, 426888]" = torch.ops.aten.cat.default([gather, gather_1], 2);  gather = gather_1 = None
        iota: "i64[382788]" = torch.ops.prims.iota.default(382788, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2);  inductor_seeds = None
        inductor_randint_2: "i64[4, 4, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 44100, _shape_param_4, inductor_lookup_seed_3);  _shape_param_4 = inductor_lookup_seed_3 = None
        expand_1: "i64[4, 4, 2, 1]" = torch.ops.aten.expand.default(inductor_randint_2, _shape_param_5);  inductor_randint_2 = _shape_param_5 = None
        add: "i64[4, 4, 2, 382788]" = torch.ops.aten.add.Tensor(iota, expand_1);  iota = expand_1 = None
        gather_2: "f32[4, 4, 2, 382788]" = torch.ops.aten.gather.default(cat, 3, add);  cat = add = None
        view: "f32[1, 4, 4, 2, 382788]" = torch.ops.aten.view.default(gather_2, _shape_param_6);  gather_2 = _shape_param_6 = None
        expand_2: "i64[1, 4, 4, 2, 382788]" = torch.ops.aten.expand.default(getitem_1, _shape_param_7);  getitem_1 = _shape_param_7 = None
        gather_3: "f32[1, 4, 4, 2, 382788]" = torch.ops.aten.gather.default(view, 1, expand_2);  view = expand_2 = None
        view_1: "f32[4, 4, 2, 382788]" = torch.ops.aten.view.default(gather_3, _shape_param_8);  gather_3 = _shape_param_8 = None
        sum_1: "f32[4, 2, 382788]" = torch.ops.aten.sum.dim_IntList(view_1, [1], dtype = torch.float32)
        return (view_1, sum_1)



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
