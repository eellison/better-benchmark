"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 4ef882a876a7
Shape hash: 595a70dd
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f16[32, 1024, 64]", bmm_2: "f16[32, 64, 1024]", bmm_8: "f16[32, 1024, 64]", bmm_10: "f16[32, 64, 1024]", bmm_16: "f16[32, 1024, 64]", bmm_18: "f16[32, 64, 1024]", bmm_24: "f16[32, 1024, 64]", bmm_26: "f16[32, 64, 1024]", bmm_32: "f16[32, 1024, 64]", bmm_34: "f16[32, 64, 1024]", bmm_40: "f16[32, 1024, 64]", bmm_42: "f16[32, 64, 1024]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm, [4, 8, 1024, 64]);  bmm = None
        reshape_default_1: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_2, [4, 8, 64, 1024]);  bmm_2 = None
        permute_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        permute_default_1: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, [4, 1024, 512]);  clone_default = None
        reshape_default_3: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_2, [4096, 512]);  reshape_default_2 = None
        permute_default_2: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_4: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_2, [4, 1024, 512]);  permute_default_2 = None
        clone_default_1: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_5: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_1, [4096, 512]);  clone_default_1 = None
        reshape_default_6: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_8, [4, 8, 1024, 64]);  bmm_8 = None
        reshape_default_7: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_10, [4, 8, 64, 1024]);  bmm_10 = None
        permute_default_3: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 3, 2]);  reshape_default_7 = None
        permute_default_4: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        clone_default_2: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_8: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, [4, 1024, 512]);  clone_default_2 = None
        reshape_default_9: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_8, [4096, 512]);  reshape_default_8 = None
        permute_default_5: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_3, [0, 2, 1, 3]);  permute_default_3 = None
        reshape_default_10: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_5, [4, 1024, 512]);  permute_default_5 = None
        clone_default_3: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_10, memory_format = torch.contiguous_format);  reshape_default_10 = None
        reshape_default_11: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_3, [4096, 512]);  clone_default_3 = None
        reshape_default_12: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_16, [4, 8, 1024, 64]);  bmm_16 = None
        reshape_default_13: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_18, [4, 8, 64, 1024]);  bmm_18 = None
        permute_default_6: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_13, [0, 1, 3, 2]);  reshape_default_13 = None
        permute_default_7: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_12, [0, 2, 1, 3]);  reshape_default_12 = None
        clone_default_4: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_14: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_4, [4, 1024, 512]);  clone_default_4 = None
        reshape_default_15: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_14, [4096, 512]);  reshape_default_14 = None
        permute_default_8: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_6, [0, 2, 1, 3]);  permute_default_6 = None
        reshape_default_16: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_8, [4, 1024, 512]);  permute_default_8 = None
        clone_default_5: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_16, memory_format = torch.contiguous_format);  reshape_default_16 = None
        reshape_default_17: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_5, [4096, 512]);  clone_default_5 = None
        reshape_default_18: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_24, [4, 8, 1024, 64]);  bmm_24 = None
        reshape_default_19: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_26, [4, 8, 64, 1024]);  bmm_26 = None
        permute_default_9: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_19, [0, 1, 3, 2]);  reshape_default_19 = None
        permute_default_10: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_18, [0, 2, 1, 3]);  reshape_default_18 = None
        clone_default_6: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_10, memory_format = torch.contiguous_format);  permute_default_10 = None
        reshape_default_20: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_6, [4, 1024, 512]);  clone_default_6 = None
        reshape_default_21: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_20, [4096, 512]);  reshape_default_20 = None
        permute_default_11: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_9, [0, 2, 1, 3]);  permute_default_9 = None
        reshape_default_22: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_11, [4, 1024, 512]);  permute_default_11 = None
        clone_default_7: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_22, memory_format = torch.contiguous_format);  reshape_default_22 = None
        reshape_default_23: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_7, [4096, 512]);  clone_default_7 = None
        reshape_default_24: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_32, [4, 8, 1024, 64]);  bmm_32 = None
        reshape_default_25: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_34, [4, 8, 64, 1024]);  bmm_34 = None
        permute_default_12: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_25, [0, 1, 3, 2]);  reshape_default_25 = None
        permute_default_13: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_24, [0, 2, 1, 3]);  reshape_default_24 = None
        clone_default_8: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_13, memory_format = torch.contiguous_format);  permute_default_13 = None
        reshape_default_26: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_8, [4, 1024, 512]);  clone_default_8 = None
        reshape_default_27: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_26, [4096, 512]);  reshape_default_26 = None
        permute_default_14: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_12, [0, 2, 1, 3]);  permute_default_12 = None
        reshape_default_28: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_14, [4, 1024, 512]);  permute_default_14 = None
        clone_default_9: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_28, memory_format = torch.contiguous_format);  reshape_default_28 = None
        reshape_default_29: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_9, [4096, 512]);  clone_default_9 = None
        reshape_default_30: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_40, [4, 8, 1024, 64]);  bmm_40 = None
        reshape_default_31: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_42, [4, 8, 64, 1024]);  bmm_42 = None
        permute_default_15: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_31, [0, 1, 3, 2]);  reshape_default_31 = None
        permute_default_16: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_30, [0, 2, 1, 3]);  reshape_default_30 = None
        clone_default_10: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_16, memory_format = torch.contiguous_format);  permute_default_16 = None
        reshape_default_32: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_10, [4, 1024, 512]);  clone_default_10 = None
        reshape_default_33: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_32, [4096, 512]);  reshape_default_32 = None
        permute_default_17: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_15, [0, 2, 1, 3]);  permute_default_15 = None
        reshape_default_34: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_17, [4, 1024, 512]);  permute_default_17 = None
        clone_default_11: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_34, memory_format = torch.contiguous_format);  reshape_default_34 = None
        reshape_default_35: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_11, [4096, 512]);  clone_default_11 = None
        return (reshape_default_3, reshape_default_5, reshape_default_9, reshape_default_11, reshape_default_15, reshape_default_17, reshape_default_21, reshape_default_23, reshape_default_27, reshape_default_29, reshape_default_33, reshape_default_35)


def _default_make_inputs():
    return [
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
