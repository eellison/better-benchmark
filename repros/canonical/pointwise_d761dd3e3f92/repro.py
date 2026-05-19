"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-2-5-linux.aws.h100_graph36
Pattern hash: d761dd3e3f92
Shape hash: d3bea589
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_20: "bf16[512, 2304]", expand: "b8[1, 1, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "bf16[1, 512, 2304]" = torch.ops.aten.view.default(addmm_20, _shape_param_0);  addmm_20 = _shape_param_0 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default, 768, 2);  view_default = None
        getitem: "bf16[1, 512, 768]" = split_tensor[0]
        getitem_1: "bf16[1, 512, 768]" = split_tensor[1]
        getitem_2: "bf16[1, 512, 768]" = split_tensor[2];  split_tensor = None
        view_default_1: "bf16[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_1, _shape_param_1);  getitem_1 = _shape_param_1 = None
        permute_default: "bf16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "bf16[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_2, _shape_param_2);  getitem_2 = _shape_param_2 = None
        permute_default_1: "bf16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        view_default_3: "bf16[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem, _shape_param_3);  getitem = _shape_param_3 = None
        permute_default_2: "bf16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  expand = full_default_1 = full_default = None
        expand_default: "bf16[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, _shape_param_4);  where_self = _shape_param_4 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default)


def _default_make_inputs():
    return [
    torch.randn([512, 2304], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 2, [1, 1, 512, 512], dtype=torch.bool, device='cuda'),
    [1, 512, 2304],  # _shape_param_0
    [1, 512, -1, 64],  # _shape_param_1
    [1, 512, -1, 64],  # _shape_param_2
    [1, 512, -1, 64],  # _shape_param_3
    [1, 12, 512, 512],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
