"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_000
Pattern hash: 2726d1c08e07
Shape hash: c03d75cd
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
_shapes_config = "(T([1, 1, 128], i64), T([1, 1, 128, 1], i64), T([], f32), T([], f32), T([192, 128, 128], f32), T([128, 128], i64), T([32, 6], f32), T([84], i64), S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, unsqueeze_1: "i64[1, 1, 128]", unsqueeze_2: "i64[1, 1, 128, 1]", full: "f32[]", full_1: "f32[]", bmm_16: "f32[192, 128, 128]", sub: "i64[128, 128]", arg81_1: "f32[32, 6]", inductor_seeds: "i64[84]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        unsqueeze_default: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        le_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_default, unsqueeze_2);  unsqueeze_default = unsqueeze_2 = None
        expand_default: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le_tensor, _shape_param_0);  le_tensor = _shape_param_0 = None
        where_self: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full, full_1);  expand_default = full = full_1 = None
        view_default: "f32[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_1);  bmm_16 = _shape_param_1 = None
        full_default: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[128, 128]" = torch.ops.aten.minimum.default(sub, full_default);  sub = full_default = None
        neg_default: "i64[128, 128]" = torch.ops.aten.neg.default(minimum_default);  minimum_default = None
        lt_scalar: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg_default, 16)
        convert_element_type_default: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg_default, torch.float32)
        div_tensor: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_default, 16);  convert_element_type_default = None
        log_default: "f32[128, 128]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_default, 2.0794415416798357);  log_default = None
        mul_tensor: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_tensor_1, 16);  div_tensor_1 = None
        convert_element_type_default_1: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 16);  convert_element_type_default_1 = None
        full_default_1: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_1: "i64[128, 128]" = torch.ops.aten.minimum.default(add_tensor, full_default_1);  add_tensor = full_default_1 = None
        where_self_1: "i64[128, 128]" = torch.ops.aten.where.self(lt_scalar, neg_default, minimum_default_1);  lt_scalar = neg_default = minimum_default_1 = None
        add_tensor_1: "i64[128, 128]" = torch.ops.aten.add.Tensor(where_self_1, 0);  where_self_1 = None
        embedding_default: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg81_1, add_tensor_1);  arg81_1 = add_tensor_1 = None
        permute_default: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_1: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None
        add_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, where_self);  unsqueeze_default_1 = where_self = None
        add_tensor_3: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default, add_tensor_2);  view_default = add_tensor_2 = None
        amax_default: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_3, [-1], True)
        sub_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_3, amax_default);  add_tensor_3 = amax_default = None
        exp_default: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35);  inductor_seeds = None
        inductor_random_default: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor_2);  gt_scalar = div_tensor_2 = None
        mul_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        expand_default_1: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_tensor_2, _shape_param_2);  mul_tensor_2 = _shape_param_2 = None
        view_default_1: "f32[192, 128, 128]" = torch.ops.aten.view.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None
        return view_default_1



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
