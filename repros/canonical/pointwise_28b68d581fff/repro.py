"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph40
Pattern hash: 28b68d581fff
Shape hash: b5d1ca59
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1024], f16), T([1, 128, 1024], f32), T([4096, 1024], f16, stride=(1, 4096)), T([1024, 4096], f16, stride=(1, 1024)), T([1024, 1024], f16, stride=(1, 1024)), T([16, 128, 128], f16), T([16, 128, 64], f16, stride=(64, 1024, 1)), T([16, 128, 64], f16, stride=(64, 1024, 1)), T([16, 64, 128], f16, stride=(64, 1, 1024)), T([1024, 1024], f16, stride=(1, 1024)), T([1024, 1024], f16, stride=(1, 1024)), T([1024, 1024], f16, stride=(1, 1024)), S([1, 128, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f16[128, 1024]", add_3: "f32[1, 128, 1024]", permute_10: "f16[4096, 1024]", permute_9: "f16[1024, 4096]", permute_8: "f16[1024, 1024]", convert_element_type_7: "f16[16, 128, 128]", view_9: "f16[16, 128, 64]", view_7: "f16[16, 128, 64]", permute_6: "f16[16, 64, 128]", permute_2: "f16[1024, 1024]", permute_1: "f16[1024, 1024]", permute: "f16[1024, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f16[1, 128, 1024]" = torch.ops.aten.view.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None
        native_dropout_default = torch.ops.aten.native_dropout.default(view_default, 0.1, True);  view_default = None
        getitem: "f16[1, 128, 1024]" = native_dropout_default[0]
        getitem_1: "b8[1, 128, 1024]" = native_dropout_default[1];  native_dropout_default = None
        add_tensor: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, getitem);  add_3 = getitem = None
        permute_default: "f16[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_default_1: "f16[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_default_2: "f16[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_default_3: "f16[16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_7, [0, 2, 1]);  convert_element_type_7 = None
        permute_default_4: "f16[16, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_5: "f16[16, 64, 128]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_default_6: "f16[16, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        permute_default_7: "f16[1024, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_default_8: "f16[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_default_9: "f16[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (getitem_1, add_tensor, permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9)


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
