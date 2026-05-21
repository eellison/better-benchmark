"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train_000
Pattern hash: 2a81770def44
Shape hash: 95a42d0c
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
_shapes_config = "(T([96, 1024, 1024], f32), T([32, 12], f32), T([124], i64), T([96, 1024, 1024], f32), T([32, 12], f32), S([8, -1, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([96, 1024, 1024]), S([8, -1, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([96, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[96, 1024, 1024]", arg6_1: "f32[32, 12]", inductor_seeds: "i64[124]", bmm_24: "f32[96, 1024, 1024]", arg105_1: "f32[32, 12]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3)
        ge_scalar: "b8[1, 1, 1024, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0)
        expand_default: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(ge_scalar, _shape_param_0);  ge_scalar = _shape_param_0 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = None
        view_default: "f32[8, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        unsqueeze_default_3: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1)
        add_tensor_1: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze_default_3, 0);  unsqueeze_default_3 = None
        unsqueeze_default_4: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        sub_tensor: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze_default_4, add_tensor_1);  unsqueeze_default_4 = add_tensor_1 = None
        gt_scalar: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub_tensor, 0)
        convert_element_type_default: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_scalar, torch.int64);  gt_scalar = None
        mul_tensor: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 16);  convert_element_type_default = None
        add_tensor_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, 0);  mul_tensor = None
        abs_default: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub_tensor)
        lt_scalar: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_default, 8)
        convert_element_type_default_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_default, torch.float32)
        div_tensor: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_default_1, 8);  convert_element_type_default_1 = None
        log_default: "f32[1024, 1024]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_default, 2.772588722239781);  log_default = None
        mul_tensor_1: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_tensor_1, 8);  div_tensor_1 = None
        convert_element_type_default_2: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.int64);  mul_tensor_1 = None
        add_tensor_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, 8);  convert_element_type_default_2 = None
        full_default_2: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_tensor_3, full_default_2);  add_tensor_3 = full_default_2 = None
        where_self_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_scalar, abs_default, minimum_default);  lt_scalar = abs_default = minimum_default = None
        add_tensor_4: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, where_self_1);  add_tensor_2 = where_self_1 = None
        embedding_default: "f32[1024, 1024, 12]" = torch.ops.aten.embedding.default(arg6_1, add_tensor_4);  arg6_1 = add_tensor_4 = None
        permute_default: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None
        add_tensor_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default_5, where_self);  unsqueeze_default_5 = where_self = None
        add_tensor_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_default, add_tensor_5);  view_default = add_tensor_5 = None
        amax_default: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_tensor_6, [-1], True)
        sub_tensor_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_6, amax_default);  add_tensor_6 = amax_default = None
        exp_default: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_default: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar_1: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar_1, div_tensor_2);  gt_scalar_1 = div_tensor_2 = None
        mul_tensor_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        expand_default_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_2);  mul_tensor_3 = _shape_param_2 = None
        view_default_1: "f32[96, 1024, 1024]" = torch.ops.aten.view.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None
        unsqueeze_default_6: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default_6, unsqueeze_default_2);  unsqueeze_default_6 = unsqueeze_default_2 = None
        expand_default_2: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le_tensor, _shape_param_4);  le_tensor = _shape_param_4 = None
        where_self_2: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default_2, full_default, full_default_1);  expand_default_2 = full_default = full_default_1 = None
        view_default_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_24, _shape_param_5);  bmm_24 = _shape_param_5 = None
        full_default_3: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub_tensor, full_default_3);  sub_tensor = full_default_3 = None
        neg_default: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_default_1);  minimum_default_1 = None
        lt_scalar_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg_default, 16)
        convert_element_type_default_3: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg_default, torch.float32)
        div_tensor_3: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_default_3, 16);  convert_element_type_default_3 = None
        log_default_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_tensor_3);  div_tensor_3 = None
        div_tensor_4: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_default_1, 2.0794415416798357);  log_default_1 = None
        mul_tensor_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_tensor_4, 16);  div_tensor_4 = None
        convert_element_type_default_4: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.int64);  mul_tensor_4 = None
        add_tensor_7: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 16);  convert_element_type_default_4 = None
        full_default_4: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_tensor_7, full_default_4);  add_tensor_7 = full_default_4 = None
        where_self_3: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_scalar_1, neg_default, minimum_default_2);  lt_scalar_1 = neg_default = minimum_default_2 = None
        add_tensor_8: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_self_3, 0);  where_self_3 = None
        embedding_default_1: "f32[1024, 1024, 12]" = torch.ops.aten.embedding.default(arg105_1, add_tensor_8);  arg105_1 = add_tensor_8 = None
        permute_default_1: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(embedding_default_1, [2, 0, 1]);  embedding_default_1 = None
        unsqueeze_default_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default_1, 0);  permute_default_1 = None
        add_tensor_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default_7, where_self_2);  unsqueeze_default_7 = where_self_2 = None
        add_tensor_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_default_2, add_tensor_9);  view_default_2 = add_tensor_9 = None
        amax_default_1: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_tensor_10, [-1], True)
        sub_tensor_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_10, amax_default_1);  add_tensor_10 = amax_default_1 = None
        exp_default_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True)
        div_tensor_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 51);  inductor_seeds = None
        inductor_random_default_1: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_2: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar_2, div_tensor_5);  gt_scalar_2 = div_tensor_5 = None
        mul_tensor_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_5, 1.1111111111111112);  mul_tensor_5 = None
        expand_default_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_tensor_6, _shape_param_6);  mul_tensor_6 = _shape_param_6 = None
        view_default_3: "f32[96, 1024, 1024]" = torch.ops.aten.view.default(expand_default_3, _shape_param_7);  expand_default_3 = _shape_param_7 = None
        permute_default_2: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
        permute_default_3: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
        return (permute_default_2, permute_default_3)



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
