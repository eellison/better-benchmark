"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer
Pattern hash: 68ce17870caf
Shape hash: 1972aff7
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
    def forward(self, arg0_1: "i64[16, 128]", arg1_1: "bf16[192, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        gt: "b8[16, 128]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        unsqueeze: "b8[16, 1, 128]" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
        repeat: "b8[16, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        unsqueeze_1: "b8[16, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None
        eq: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        div: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view, 8.0);  view = None
        where: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq, full, div);  eq = full = div = None
        convert_element_type: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        view_1: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        return (unsqueeze_1, view_1)



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
