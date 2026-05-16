"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g53
Pattern hash: 6d4787bb47eb
Shape hash: c51476ab
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_33: "f16[8, 12, 198, 64]", getitem_34: "f16[8, 12, 198, 64]", getitem_35: "f16[8, 12, 198, 64]"):
        # No stacktrace found for following nodes
        cat_default: "f16[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_33, getitem_34, getitem_35]);  getitem_33 = getitem_34 = getitem_35 = None
        reshape_default: "f16[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_default, [3, 8, 12, 198, 64]);  cat_default = None
        permute_default: "f16[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 3, 0, 2, 4]);  reshape_default = None
        clone_default: "f16[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f16[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_default, [8, 198, 2304]);  clone_default = None
        reshape_default_2: "f16[1584, 2304]" = torch.ops.aten.reshape.default(reshape_default_1, [1584, 2304]);  reshape_default_1 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # getitem_33
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # getitem_34
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # getitem_35
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
