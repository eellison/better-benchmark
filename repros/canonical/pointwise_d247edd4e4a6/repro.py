"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: d247edd4e4a6
Shape hash: 11f71568
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
    def forward(self, arg0_1: "bf16[512, 30528]", arg1_1: "bf16[32768, 384]", arg2_1: "i64[256, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        slice_1: "bf16[512, 30522]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, -6);  arg0_1 = None
        convert_element_type: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        slice_2: "f32[128, 30522]" = torch.ops.aten.slice.Tensor(convert_element_type, 0, 0, 128)
        slice_3: "f32[384, 30522]" = torch.ops.aten.slice.Tensor(convert_element_type, 0, 128, 512);  convert_element_type = None
        permute: "f32[30522, 128]" = torch.ops.aten.permute.default(slice_2, [1, 0]);  slice_2 = None
        view: "bf16[256, 128, 384]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        convert_element_type_1: "f32[256, 128, 384]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        slice_4: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 2, 0, 128)
        slice_5: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 2, 128, 256)
        slice_6: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 2, 256, 384);  convert_element_type_1 = None
        constant_pad_nd: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_6, [0, 0, -1, 0, 0, 0]);  slice_6 = None
        full: "f32[256, 128, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_scatter: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full, constant_pad_nd, 1, 0, -1);  constant_pad_nd = None
        add: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(slice_5, slice_scatter);  slice_5 = slice_scatter = None
        constant_pad_nd_1: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_4, [0, 0, 0, -1, 0, 0]);  slice_4 = None
        slice_scatter_1: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full, constant_pad_nd_1, 1, 1, 9223372036854775807);  full = constant_pad_nd_1 = None
        add_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(add, slice_scatter_1);  add = slice_scatter_1 = None
        ge: "b8[256, 128]" = torch.ops.aten.ge.Scalar(arg2_1, 0)
        lt: "b8[256, 128]" = torch.ops.aten.lt.Scalar(arg2_1, 30522)
        bitwise_and: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[256, 128]" = torch.ops.aten.ne.Scalar(arg2_1, 0)
        bitwise_and_1: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[256, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_1: "f32[30522, 128]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        _unsafe_masked_index_put_accumulate: "f32[30522, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze, [arg2_1], add_1);  full_1 = unsqueeze = arg2_1 = add_1 = None
        add_2: "f32[30522, 128]" = torch.ops.aten.add.Tensor(permute, _unsafe_masked_index_put_accumulate);  permute = _unsafe_masked_index_put_accumulate = None
        return (slice_3, add_2)



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
