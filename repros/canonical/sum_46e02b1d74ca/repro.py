"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: 46e02b1d74ca
Shape hash: 71353b3b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([192, 128, 128], f32), T([32, 6, 128, 128], b8), T([1, 1, 128, 1], b8), T([], f32), T([192, 128, 128], f32), T([128, 128, 6], f32), T([32, 6, 128, 1], f32), T([32, 6, 128, 1], f32), S([32, 6, 128, 128]), S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_93: "f32[192, 128, 128]", arg199_1: "b8[32, 6, 128, 128]", arg190_1: "b8[1, 1, 128, 1]", full_1: "f32[]", arg194_1: "f32[192, 128, 128]", arg196_1: "f32[128, 128, 6]", arg197_1: "f32[32, 6, 128, 1]", arg198_1: "f32[32, 6, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_93, _shape_param_0);  bmm_93 = _shape_param_0 = None
        convert_element_type_default: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.float32);  arg199_1 = None
        mul_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        expand_default: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(arg190_1, _shape_param_1);  arg190_1 = _shape_param_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_1, full_default);  expand_default = full_1 = full_default = None
        view_default_1: "f32[32, 6, 128, 128]" = torch.ops.aten.view.default(arg194_1, _shape_param_2);  arg194_1 = _shape_param_2 = None
        permute_default: "f32[6, 128, 128]" = torch.ops.aten.permute.default(arg196_1, [2, 0, 1]);  arg196_1 = None
        unsqueeze_default: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None
        add_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_default, where_self);  unsqueeze_default = where_self = None
        add_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_1, add_tensor);  view_default_1 = add_tensor = None
        sub_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, arg197_1);  add_tensor_1 = arg197_1 = None
        exp_default: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, arg198_1);  exp_default = arg198_1 = None
        mul_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor);  mul_tensor_1 = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        view_default_2: "f32[192, 128, 128]" = torch.ops.aten.view.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        view_default_3: "f32[32, 6, 128, 128]" = torch.ops.aten.view.default(view_default_2, _shape_param_4);  view_default_2 = _shape_param_4 = None
        view_default_4: "f32[192, 128, 128]" = torch.ops.aten.view.default(view_default_3, _shape_param_5);  view_default_3 = _shape_param_5 = None
        return view_default_4



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
