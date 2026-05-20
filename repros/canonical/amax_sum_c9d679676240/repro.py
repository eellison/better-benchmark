"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-4-5-linux.aws.a100_graph10
Pattern hash: c9d679676240
Shape hash: 7cfcfdd2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([256, 1024], bf16), T([16, 256, 256], bf16), S([1, -1, 256, 256]), S([1, 256, 1024]), S([1, -1, 16, 64]), S([16, 256, 64]), S([1, 16, 256, 256]), S([16, 256, 256]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "bf16[256, 1024]", bmm: "bf16[16, 256, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[256]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        iota_default_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[256]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default: "i64[1, 256]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 256]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[1, 256]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_4: "i64[1, 1, 256]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 1, 256]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        le_tensor: "b8[1, 1, 256, 256]" = torch.ops.aten.le.Tensor(unsqueeze_default_5, unsqueeze_default_2);  unsqueeze_default_5 = unsqueeze_default_2 = None
        expand_default: "b8[1, 1, 256, 256]" = torch.ops.aten.expand.default(le_tensor, _shape_param_0);  le_tensor = _shape_param_0 = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 1, 256, 256]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None
        view_default: "bf16[1, 256, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_1);  addmm_2 = _shape_param_1 = None
        view_default_1: "bf16[1, 256, 16, 64]" = torch.ops.aten.view.default(view_default, _shape_param_2);  view_default = _shape_param_2 = None
        permute_default: "bf16[1, 16, 256, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "bf16[16, 256, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None
        view_default_3: "bf16[1, 16, 256, 256]" = torch.ops.aten.view.default(bmm, _shape_param_4);  bmm = _shape_param_4 = None
        add_tensor_2: "bf16[1, 16, 256, 256]" = torch.ops.aten.add.Tensor(view_default_3, where_self);  view_default_3 = where_self = None
        view_default_4: "bf16[16, 256, 256]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_5);  add_tensor_2 = _shape_param_5 = None
        convert_element_type_default: "f32[16, 256, 256]" = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        amax_default: "f32[16, 256, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[16, 256, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[16, 256, 256]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 256, 256]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[16, 256, 256]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        return (view_default_2, convert_element_type_default_1)


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
