"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer
Pattern hash: 75f17d2ac9eb
Shape hash: a655df0f
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
    def forward(self, arg0_1: "bf16[20005, 768]", arg1_1: "i64[16, 128]", arg2_1: "bf16[1, 512, 768]", arg3_1: "bf16[3, 768]", arg4_1: "i64[16, 128]", arg5_1: "bf16[768]", arg6_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding: "bf16[16, 128, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        slice_1: "bf16[1, 128, 768]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        add: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(embedding, slice_1);  embedding = slice_1 = None
        embedding_1: "bf16[16, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1, 0);  arg3_1 = arg4_1 = None
        add_1: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None
        mean: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_1, [-1], True)
        sub: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_1, mean);  mean = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg5_1, sub);  arg5_1 = sub = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        var: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type, [-1], correction = 1.0, keepdim = True);  convert_element_type = None
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
        convert_element_type_1: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt, torch.bfloat16);  sqrt = None
        add_2: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-06);  convert_element_type_1 = None
        div: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_2);  mul = add_2 = None
        add_3: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg6_1);  div = arg6_1 = None
        view: "bf16[2048, 768]" = torch.ops.aten.view.default(add_3, _shape_param_0);  _shape_param_0 = None
        view_1: "bf16[2048, 768]" = torch.ops.aten.view.default(add_3, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[2048, 768]" = torch.ops.aten.view.default(add_3, _shape_param_2);  add_3 = _shape_param_2 = None
        return (add_1, view, view_1, view_2)



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
