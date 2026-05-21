"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_000
Pattern hash: ca987107ccc7
Shape hash: bbddc58e
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
_shapes_config = "(T([1, 128], i64, gen=Index(128)), T([32, 128], i64, gen=Index(32)), T([512, 128, 128], f32), T([1, 1, 2048, 2048], b8), S([32, -1, 128, 128]), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, unsqueeze: "i64[1, 128]", cumsum: "i64[32, 128]", bmm: "f32[512, 128, 128]", arg8_1: "b8[1, 1, 2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        iota_default: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1);  iota_default = None
        unsqueeze_default_1: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_default_4: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 3)
        unsqueeze_default_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_default_5, unsqueeze_default_4)
        bitwise_and_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None
        index_tensor: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_4]);  unsqueeze_default_4 = None
        index_tensor_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_5]);  cumsum = unsqueeze_default_2 = unsqueeze_default_5 = None
        eq_tensor: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        bitwise_and_tensor_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None
        expand_default: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_0);  bitwise_and_tensor_1 = _shape_param_0 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  expand_default = full_default_1 = None
        view_default: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        slice_tensor: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg8_1, 2, 0, 128);  arg8_1 = None
        slice_tensor_1: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 128);  slice_tensor = None
        where_self_1: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_tensor_1, view_default, full_default_2);  slice_tensor_1 = view_default = full_default_2 = None
        add_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_self_1, where_self);  where_self_1 = where_self = None
        amax_default: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        expand_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_tensor, _shape_param_2);  div_tensor = _shape_param_2 = None
        view_default_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None
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
