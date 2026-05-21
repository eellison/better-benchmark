"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_001
Pattern hash: 5d7c3bd1f499
Shape hash: 0a3b54e6
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
_shapes_config = "(T([8, 1024], f32), S([8, -1, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None
        iota_default_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_5);  unsqueeze_default_5 = None
        bitwise_and_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None
        convert_element_type_default: "b8[8, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bool);  arg0_1 = None
        iota_default_2: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_6: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 1);  iota_default_2 = None
        unsqueeze_default_7: "i64[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "i64[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        index_tensor: "b8[8, 1, 1, 1024]" = torch.ops.aten.index.Tensor(convert_element_type_default, [unsqueeze_default_8, unsqueeze_default_2]);  convert_element_type_default = unsqueeze_default_8 = unsqueeze_default_2 = None
        bitwise_and_tensor_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, index_tensor);  bitwise_and_tensor = index_tensor = None
        expand_default: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_0);  bitwise_and_tensor_1 = _shape_param_0 = None
        return expand_default



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
