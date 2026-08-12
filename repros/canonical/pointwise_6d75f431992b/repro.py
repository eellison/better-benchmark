"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: 6d75f431992b
Shape hash: 9210619b
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "i64[32, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
        unsqueeze_1: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(arg0_1, 1);  arg0_1 = None
        unsqueeze_4: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 3)
        unsqueeze_5: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_4)
        bitwise_and: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None
        index: "i64[32, 1, 512, 1]" = torch.ops.aten.index.Tensor(arg1_1, [unsqueeze_2, unsqueeze_4]);  unsqueeze_4 = None
        index_1: "i64[32, 1, 1, 512]" = torch.ops.aten.index.Tensor(arg1_1, [unsqueeze_2, unsqueeze_5]);  arg1_1 = unsqueeze_2 = unsqueeze_5 = None
        eq: "b8[32, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand: "b8[32, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_0);  bitwise_and_1 = _shape_param_0 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_2, full_1);  expand = full_2 = full_1 = None
        expand_1: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where, _shape_param_1);  _shape_param_1 = None
        return (where, expand_1)



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
