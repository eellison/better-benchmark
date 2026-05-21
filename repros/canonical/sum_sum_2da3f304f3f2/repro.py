"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: 2da3f304f3f2
Shape hash: d3e0ffa9
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
_shapes_config = "(T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([128, 128], i64, gen=Index(32)), T([], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([128, 128], i64, gen=Index(32)))"

class Repro(torch.nn.Module):
    def forward(self, view_44: "f32[32, 6, 128, 128]", view_95: "f32[32, 6, 128, 128]", view_146: "f32[32, 6, 128, 128]", view_197: "f32[32, 6, 128, 128]", view_248: "f32[32, 6, 128, 128]", view_299: "f32[32, 6, 128, 128]", view_350: "f32[32, 6, 128, 128]", view_401: "f32[32, 6, 128, 128]", arg322_1: "i64[128, 128]", full_1: "f32[]", view_432: "f32[32, 6, 128, 128]", view_462: "f32[32, 6, 128, 128]", view_492: "f32[32, 6, 128, 128]", view_522: "f32[32, 6, 128, 128]", view_552: "f32[32, 6, 128, 128]", view_582: "f32[32, 6, 128, 128]", view_612: "f32[32, 6, 128, 128]", view_643: "f32[32, 6, 128, 128]", arg195_1: "i64[128, 128]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_44, view_95);  view_44 = view_95 = None
        add_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor, view_146);  add_tensor = view_146 = None
        add_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_1, view_197);  add_tensor_1 = view_197 = None
        add_tensor_3: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_2, view_248);  add_tensor_2 = view_248 = None
        add_tensor_4: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_3, view_299);  add_tensor_3 = view_299 = None
        add_tensor_5: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_4, view_350);  add_tensor_4 = view_350 = None
        add_tensor_6: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_5, view_401);  add_tensor_5 = view_401 = None
        sum_dim_int_list: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor_6, [0], True);  add_tensor_6 = None
        squeeze_dim: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 0);  sum_dim_int_list = None
        permute_default: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        eq_scalar: "b8[128, 128]" = torch.ops.aten.eq.Scalar(arg322_1, -1)
        unsqueeze_default: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[128, 128, 6]" = torch.ops.aten.where.self(unsqueeze_default, full_1, permute_default);  unsqueeze_default = permute_default = None
        clone_default: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default: "f32[32, 6]" = torch.ops.aten.full.default([32, 6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 6]" = torch.ops.aten.index_put.default(full_default, [arg322_1], clone_default, True);  arg322_1 = clone_default = None
        add_tensor_7: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_432, view_462);  view_432 = view_462 = None
        add_tensor_8: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_7, view_492);  add_tensor_7 = view_492 = None
        add_tensor_9: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_8, view_522);  add_tensor_8 = view_522 = None
        add_tensor_10: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_9, view_552);  add_tensor_9 = view_552 = None
        add_tensor_11: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_10, view_582);  add_tensor_10 = view_582 = None
        add_tensor_12: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_11, view_612);  add_tensor_11 = view_612 = None
        add_tensor_13: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_12, view_643);  add_tensor_12 = view_643 = None
        sum_dim_int_list_1: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor_13, [0], True);  add_tensor_13 = None
        squeeze_dim_1: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default_1: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None
        eq_scalar_1: "b8[128, 128]" = torch.ops.aten.eq.Scalar(arg195_1, -1)
        unsqueeze_default_1: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[128, 128, 6]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, permute_default_1);  unsqueeze_default_1 = full_1 = permute_default_1 = None
        clone_default_1: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_self_1, memory_format = torch.contiguous_format);  where_self_1 = None
        index_put_default_1: "f32[32, 6]" = torch.ops.aten.index_put.default(full_default, [arg195_1], clone_default_1, True);  full_default = arg195_1 = clone_default_1 = None
        return (index_put_default, index_put_default_1)



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
