"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s1_g5
Pattern hash: 5792ec02a0d7
Shape hash: d6720fbe
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "bf16[50265, 1024]", arg0_1: "i64[1, 1024]", arg2_1: "bf16[1026, 1024]"):
        # No stacktrace found for following nodes
        embedding_default: "bf16[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "bf16[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        add_tensor_1: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default_1: "bf16[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_tensor_1);  arg2_1 = add_tensor_1 = None
        add_tensor_2: "bf16[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None
        convert_element_type_default: "f32[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        _output_to_half_0: "bf16[1, 1024, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.bfloat16);  getitem = None
        _output_to_half_1: "bf16[1, 1024, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.bfloat16);  getitem_1 = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([50265, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 2, [1, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([1026, 1024], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
