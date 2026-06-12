"""
Standalone repro captured via capture_hook.
Label: hf_openai/whisper-tiny_infer
Pattern hash: 7ba7efb075cd
Shape hash: d7517139
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
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        iota: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_4: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        le: "b8[1, 1, 1, 1]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5);  unsqueeze_2 = unsqueeze_5 = None
        expand: "b8[1, 1, 1, 1]" = torch.ops.aten.expand.default(le, [1, -1, 1, 1]);  le = None
        scalar_tensor: "bf16[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "bf16[]" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where: "bf16[1, 1, 1, 1]" = torch.ops.aten.where.self(expand, scalar_tensor, scalar_tensor_1);  scalar_tensor = scalar_tensor_1 = None
        constant_pad_nd: "bf16[1, 1, 1, 8]" = torch.ops.aten.constant_pad_nd.default(where, _shape_param_0, 0.0);  where = _shape_param_0 = None
        slice_1: "bf16[1, 1, 1, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 1);  constant_pad_nd = None
        expand_1: "bf16[1, 6, 1, 1]" = torch.ops.aten.expand.default(slice_1, _shape_param_1);  slice_1 = _shape_param_1 = None
        scalar_tensor_2: "bf16[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_3: "bf16[]" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where_1: "bf16[1, 1, 1, 1]" = torch.ops.aten.where.self(expand, scalar_tensor_2, scalar_tensor_3);  scalar_tensor_2 = scalar_tensor_3 = None
        constant_pad_nd_1: "bf16[1, 1, 1, 8]" = torch.ops.aten.constant_pad_nd.default(where_1, _shape_param_2, 0.0);  where_1 = _shape_param_2 = None
        slice_2: "bf16[1, 1, 1, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_1, -1, 0, 1);  constant_pad_nd_1 = None
        expand_2: "bf16[1, 6, 1, 1]" = torch.ops.aten.expand.default(slice_2, _shape_param_3);  slice_2 = _shape_param_3 = None
        scalar_tensor_4: "bf16[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_5: "bf16[]" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where_2: "bf16[1, 1, 1, 1]" = torch.ops.aten.where.self(expand, scalar_tensor_4, scalar_tensor_5);  scalar_tensor_4 = scalar_tensor_5 = None
        constant_pad_nd_2: "bf16[1, 1, 1, 8]" = torch.ops.aten.constant_pad_nd.default(where_2, _shape_param_4, 0.0);  where_2 = _shape_param_4 = None
        slice_3: "bf16[1, 1, 1, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, -1, 0, 1);  constant_pad_nd_2 = None
        expand_3: "bf16[1, 6, 1, 1]" = torch.ops.aten.expand.default(slice_3, _shape_param_5);  slice_3 = _shape_param_5 = None
        scalar_tensor_6: "bf16[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_7: "bf16[]" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where_3: "bf16[1, 1, 1, 1]" = torch.ops.aten.where.self(expand, scalar_tensor_6, scalar_tensor_7);  expand = scalar_tensor_6 = scalar_tensor_7 = None
        constant_pad_nd_3: "bf16[1, 1, 1, 8]" = torch.ops.aten.constant_pad_nd.default(where_3, _shape_param_6, 0.0);  where_3 = _shape_param_6 = None
        slice_4: "bf16[1, 1, 1, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_3, -1, 0, 1);  constant_pad_nd_3 = None
        expand_4: "bf16[1, 6, 1, 1]" = torch.ops.aten.expand.default(slice_4, _shape_param_7);  slice_4 = _shape_param_7 = None
        return (expand_1, expand_2, expand_3, expand_4)



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
