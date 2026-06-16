"""
Standalone repro captured via capture_hook.
Label: hf_openai/gpt-oss-20b_infer
Pattern hash: 59acdb823e92
Shape hash: a9d89d2e
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
    def forward(self, arg0_1: "bf16[1000, 32]", arg1_1: "bf16[1000, 2880]"):
        # No stacktrace found for following nodes
        topk = torch.ops.aten.topk.default(arg0_1, 4);  arg0_1 = None
        getitem: "bf16[1000, 4]" = topk[0]
        getitem_1: "i64[1000, 4]" = topk[1];  topk = None
        view: "i64[4000]" = torch.ops.aten.view.default(getitem_1, [-1]);  getitem_1 = None
        sort = torch.ops.aten.sort.default(view);  view = None
        getitem_2: "i64[4000]" = sort[0]
        getitem_3: "i64[4000]" = sort[1];  sort = None
        ge: "b8[4000]" = torch.ops.aten.ge.Scalar(getitem_2, 32)
        unsqueeze: "b8[4000, 1]" = torch.ops.aten.unsqueeze.default(ge, -1);  ge = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div: "i64[4000]" = torch.ops.aten.div.Tensor_mode(getitem_3, 4, rounding_mode = 'floor')
        index: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg1_1, [div]);  arg1_1 = div = None
        where: "bf16[4000, 2880]" = torch.ops.aten.where.self(unsqueeze, full, index);  full = index = None
        convert_element_type: "i32[4000]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.int32)
        return (getitem, getitem_2, getitem_3, unsqueeze, where, convert_element_type)



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
