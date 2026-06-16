"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_train
Pattern hash: bc3fbecd2ce0
Shape hash: d9690337
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "f32[32, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view, arg1_1);  view = arg1_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add, full);  add = full = None
        view_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_1);  maximum = _shape_param_1 = None
        amax: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_1, [-1], True)
        sub: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_1, amax);  view_1 = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_seeds: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[512, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        gt: "b8[512, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(gt, div);  div = None
        mul_1: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        convert_element_type: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        permute: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(convert_element_type, [0, 2, 1])
        return (amax, sum_1, inductor_seeds, gt, convert_element_type, permute)



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
