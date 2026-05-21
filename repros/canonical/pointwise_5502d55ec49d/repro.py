"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer_002
Pattern hash: 5502d55ec49d
Shape hash: 4fa461c7
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
_shapes_config = "(T([4096, 768], f16), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([12, 8, 512, 64]), S([180, 64, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_67: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f16[4096, 1, 768]" = torch.ops.aten.view.default(addmm_67, _shape_param_0);  addmm_67 = _shape_param_0 = None
        view_default_1: "f16[4096, 1, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_default_1, [1, 0, 2, 3]);  view_default_1 = None
        permute_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_2: "f16[12, 4096, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        view_default_3: "f16[12, 8, 512, 64]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        as_strided_default: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_default_3, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_default_3 = None
        unsqueeze_default: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_2: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 2, 3]);  unsqueeze_default = None
        permute_default_3: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 4, 3, 2]);  permute_default_2 = None
        clone_default: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_4: "f16[180, 64, 512]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        return view_default_4



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
