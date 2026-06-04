"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train_001
Pattern hash: 28fded12b2f4
Shape hash: d2deffe4
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 128], i64, gen=Index(1025)), T([64, 128], i32, gen=Index(2)), T([1026, 1024], f32), S([64, 128, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[64, 128]", convert_element_type: "i32[64, 128]", arg1_1: "f32[1026, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add_tensor: "i32[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor: "i32[64, 128]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type);  add_tensor = convert_element_type = None
        convert_element_type_default_1: "i64[64, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor_1: "i64[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        view_default: "i64[8192]" = torch.ops.aten.view.default(add_tensor_1, [-1]);  add_tensor_1 = None
        index_tensor: "f32[8192, 1024]" = torch.ops.aten.index.Tensor(arg1_1, [view_default]);  arg1_1 = view_default = None
        view_default_1: "f32[64, 128, 1024]" = torch.ops.aten.view.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
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
