"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: 083e5f8c7cdc
Shape hash: d4c044fc
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
_shapes_config = "(T([768, 64, 64], f16), T([1, 12, 4096], i64, gen=Perm(4096)), S([1, 12, 64, 64, 64]), S([1, 12, 4096, 64]), S([1, 12, 4096]), S([1, 12, 4096, 64]), S([1, 4096, 768]), S([4096, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_14: "f16[768, 64, 64]", getitem_29: "i64[1, 12, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.view.default(bmm_14, _shape_param_0);  bmm_14 = _shape_param_0 = None
        view_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        empty_memory_format: "i64[1, 12, 4096]" = torch.ops.aten.empty.memory_format([1, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default_2: "i64[1, 1, 4096]" = torch.ops.aten.view.default(iota_default, [1, 1, -1]);  iota_default = None
        expand_default: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_default_2, _shape_param_2);  view_default_2 = _shape_param_2 = None
        scatter_src: "i64[1, 12, 4096]" = torch.ops.aten.scatter.src(empty_memory_format, -1, getitem_29, expand_default);  empty_memory_format = getitem_29 = expand_default = None
        unsqueeze_default: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter_src, -1);  scatter_src = None
        expand_default_1: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_3);  unsqueeze_default = _shape_param_3 = None
        gather_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(view_default_1, 2, expand_default_1);  view_default_1 = expand_default_1 = None
        permute_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_default, [0, 2, 1, 3]);  gather_default = None
        clone_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f16[1, 4096, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        view_default_4: "f16[4096, 768]" = torch.ops.aten.view.default(view_default_3, _shape_param_5);  view_default_3 = _shape_param_5 = None
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
