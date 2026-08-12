"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_infer
Pattern hash: c372431bff28
Shape hash: 5911cf96
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
    def forward(self, arg0_1: "b8[1, 1, 2048, 2048]", arg1_1: "f32[512, 128, 128]", arg2_1: "i64[32, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg0_1, 2, 0, 128);  arg0_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        view: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view, full);  slice_2 = view = full = None
        full_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_1, le);  full_1 = le = None
        iota_2: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 1);  iota_2 = None
        unsqueeze_7: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg2_1, [unsqueeze_8, unsqueeze_5]);  unsqueeze_5 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg2_1, [unsqueeze_8, unsqueeze_2]);  arg2_1 = unsqueeze_8 = unsqueeze_2 = None
        eq: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_1);  bitwise_and_1 = _shape_param_1 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_2, full_3);  expand = full_2 = full_3 = None
        add_2: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where, where_1);  where = None
        amax: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_2, [-1], True)
        sub: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_2, amax);  add_2 = amax = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_1, _shape_param_3);  expand_1 = _shape_param_3 = None
        return (where_1, view_1)



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
