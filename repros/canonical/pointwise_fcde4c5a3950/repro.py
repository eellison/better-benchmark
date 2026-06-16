"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_infer
Pattern hash: fcde4c5a3950
Shape hash: 2a837a19
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
    def forward(self, arg0_1: "bf16[]", _shape_param_0):
        # No stacktrace found for following nodes
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0)
        unsqueeze_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_2: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3)
        ge: "b8[1, 1, 1024, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0)
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(ge, _shape_param_0);  _shape_param_0 = None
        lift_fresh_copy: "bf16[]" = torch.ops.aten.lift_fresh_copy.default(arg0_1);  arg0_1 = None
        scalar_tensor: "bf16[]" = torch.ops.aten.scalar_tensor.default(-3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, lift_fresh_copy, scalar_tensor)
        return (iota, add, unsqueeze, unsqueeze_1, unsqueeze_2, ge, expand, lift_fresh_copy, scalar_tensor, where)



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
