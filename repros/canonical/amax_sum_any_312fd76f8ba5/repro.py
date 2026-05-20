"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-1-5-linux.aws.h100_graph33
Pattern hash: 312fd76f8ba5
Shape hash: cc64be29
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 768], f16), T([192, 512, 512], f16), T([37], i64, max=37), S([16, -1, 512, 512]), S([16, 512, 768]), S([16, 512, -1, 64]), S([16, 12, 512, 512]), S([16, 12, 512, 512]), S([192, 512, 512]), S([16, 12, 512, 64]), S([192, 512, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f16[8192, 768]", bmm: "f16[192, 512, 512]", inductor_seeds: "i64[37]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        ge_scalar: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0);  unsqueeze_default_2 = None
        expand_default: "b8[16, 1, 512, 512]" = torch.ops.aten.expand.default(ge_scalar, _shape_param_0);  ge_scalar = _shape_param_0 = None
        view_default: "f16[16, 512, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_1);  addmm_2 = _shape_param_1 = None
        view_default_1: "f16[16, 512, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_2);  view_default = _shape_param_2 = None
        permute_default: "f16[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[16, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default);  expand_default = full_default_1 = full_default = None
        view_default_2: "f16[16, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None
        add_tensor_1: "f16[16, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_default_2, where_self);  view_default_2 = where_self = None
        convert_element_type_default: "f32[16, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        amax_default: "f32[16, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[16, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[16, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "f16[16, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        eq_scalar: "b8[16, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor_1, -inf);  add_tensor_1 = None
        logical_not_default: "b8[16, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[16, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[16, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f16[16, 12, 512, 512]" = torch.ops.aten.full.default([16, 12, 512, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f16[16, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, convert_element_type_default_1);  logical_not_default_1 = full_default_2 = convert_element_type_default_1 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_default: "f32[16, 12, 512, 512]" = torch.ops.prims.inductor_random.default([16, 12, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_2: "f16[16, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[16, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor: "f16[16, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self_1);  gt_scalar = where_self_1 = None
        mul_tensor_1: "f16[16, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default_1: "f16[16, 12, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_4);  mul_tensor_1 = _shape_param_4 = None
        view_default_3: "f16[192, 512, 512]" = torch.ops.aten.view.default(expand_default_1, _shape_param_5);  expand_default_1 = _shape_param_5 = None
        expand_default_2: "f16[16, 12, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_6);  permute_default = _shape_param_6 = None
        clone_default: "f16[16, 12, 512, 64]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_4: "f16[192, 512, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None
        return (view_default_3, view_default_4)


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
