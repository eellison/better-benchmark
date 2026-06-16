"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 139d37b65771
Shape hash: 3746d560
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
    def forward(self, arg0_1: "f32[16, 128, 20005]", arg1_1: "f32[16, 128, 20005]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [-1], True)
        exp: "f32[16, 128, 20005]" = torch.ops.aten.exp.default(arg1_1);  arg1_1 = None
        mul: "f32[16, 128, 20005]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(arg0_1, mul);  arg0_1 = mul = None
        convert_element_type: "bf16[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        view: "bf16[2048, 20005]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        constant_pad_nd: "bf16[2048, 20008]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[20005, 2048]" = torch.ops.aten.permute.default(view, [1, 0])
        constant_pad_nd_1: "bf16[20008, 2048]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        sum_2: "f32[1, 20005]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[20005]" = torch.ops.aten.view.default(sum_2, _shape_param_3);  sum_2 = _shape_param_3 = None
        convert_element_type_1: "bf16[20005]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_2: "f32[20005]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        return (constant_pad_nd, constant_pad_nd_1, convert_element_type_2)



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
