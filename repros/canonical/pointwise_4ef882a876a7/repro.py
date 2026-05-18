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
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f16[32, 1024, 64]", bmm_2: "f16[32, 64, 1024]", bmm_8: "f16[32, 1024, 64]", bmm_10: "f16[32, 64, 1024]", bmm_16: "f16[32, 1024, 64]", bmm_18: "f16[32, 64, 1024]", bmm_24: "f16[32, 1024, 64]", bmm_26: "f16[32, 64, 1024]", bmm_32: "f16[32, 1024, 64]", bmm_34: "f16[32, 64, 1024]", bmm_40: "f16[32, 1024, 64]", bmm_42: "f16[32, 64, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        reshape_default_1: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_2, _shape_param_1);  bmm_2 = _shape_param_1 = None
        permute_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        permute_default_1: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        reshape_default_3: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_4: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_4);  permute_default_2 = _shape_param_4 = None
        clone_default_1: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_5: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        reshape_default_6: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_8, _shape_param_6);  bmm_8 = _shape_param_6 = None
        reshape_default_7: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_10, _shape_param_7);  bmm_10 = _shape_param_7 = None
        permute_default_3: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 3, 2]);  reshape_default_7 = None
        permute_default_4: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        clone_default_2: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_8: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        reshape_default_9: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        permute_default_5: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_3, [0, 2, 1, 3]);  permute_default_3 = None
        reshape_default_10: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_10);  permute_default_5 = _shape_param_10 = None
        clone_default_3: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_10, memory_format = torch.contiguous_format);  reshape_default_10 = None
        reshape_default_11: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_11);  clone_default_3 = _shape_param_11 = None
        reshape_default_12: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_16, _shape_param_12);  bmm_16 = _shape_param_12 = None
        reshape_default_13: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_18, _shape_param_13);  bmm_18 = _shape_param_13 = None
        permute_default_6: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_13, [0, 1, 3, 2]);  reshape_default_13 = None
        permute_default_7: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_12, [0, 2, 1, 3]);  reshape_default_12 = None
        clone_default_4: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_14: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_14);  clone_default_4 = _shape_param_14 = None
        reshape_default_15: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_15);  reshape_default_14 = _shape_param_15 = None
        permute_default_8: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_6, [0, 2, 1, 3]);  permute_default_6 = None
        reshape_default_16: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_16);  permute_default_8 = _shape_param_16 = None
        clone_default_5: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_16, memory_format = torch.contiguous_format);  reshape_default_16 = None
        reshape_default_17: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_17);  clone_default_5 = _shape_param_17 = None
        reshape_default_18: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_24, _shape_param_18);  bmm_24 = _shape_param_18 = None
        reshape_default_19: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_26, _shape_param_19);  bmm_26 = _shape_param_19 = None
        permute_default_9: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_19, [0, 1, 3, 2]);  reshape_default_19 = None
        permute_default_10: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_18, [0, 2, 1, 3]);  reshape_default_18 = None
        clone_default_6: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_10, memory_format = torch.contiguous_format);  permute_default_10 = None
        reshape_default_20: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_20);  clone_default_6 = _shape_param_20 = None
        reshape_default_21: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_20, _shape_param_21);  reshape_default_20 = _shape_param_21 = None
        permute_default_11: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_9, [0, 2, 1, 3]);  permute_default_9 = None
        reshape_default_22: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_11, _shape_param_22);  permute_default_11 = _shape_param_22 = None
        clone_default_7: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_22, memory_format = torch.contiguous_format);  reshape_default_22 = None
        reshape_default_23: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_7, _shape_param_23);  clone_default_7 = _shape_param_23 = None
        reshape_default_24: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_32, _shape_param_24);  bmm_32 = _shape_param_24 = None
        reshape_default_25: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_34, _shape_param_25);  bmm_34 = _shape_param_25 = None
        permute_default_12: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_25, [0, 1, 3, 2]);  reshape_default_25 = None
        permute_default_13: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_24, [0, 2, 1, 3]);  reshape_default_24 = None
        clone_default_8: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_13, memory_format = torch.contiguous_format);  permute_default_13 = None
        reshape_default_26: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_8, _shape_param_26);  clone_default_8 = _shape_param_26 = None
        reshape_default_27: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_26, _shape_param_27);  reshape_default_26 = _shape_param_27 = None
        permute_default_14: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_12, [0, 2, 1, 3]);  permute_default_12 = None
        reshape_default_28: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_14, _shape_param_28);  permute_default_14 = _shape_param_28 = None
        clone_default_9: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_28, memory_format = torch.contiguous_format);  reshape_default_28 = None
        reshape_default_29: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_9, _shape_param_29);  clone_default_9 = _shape_param_29 = None
        reshape_default_30: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_40, _shape_param_30);  bmm_40 = _shape_param_30 = None
        reshape_default_31: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_42, _shape_param_31);  bmm_42 = _shape_param_31 = None
        permute_default_15: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_31, [0, 1, 3, 2]);  reshape_default_31 = None
        permute_default_16: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_30, [0, 2, 1, 3]);  reshape_default_30 = None
        clone_default_10: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_16, memory_format = torch.contiguous_format);  permute_default_16 = None
        reshape_default_32: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_10, _shape_param_32);  clone_default_10 = _shape_param_32 = None
        reshape_default_33: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_32, _shape_param_33);  reshape_default_32 = _shape_param_33 = None
        permute_default_17: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_15, [0, 2, 1, 3]);  permute_default_15 = None
        reshape_default_34: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_17, _shape_param_34);  permute_default_17 = _shape_param_34 = None
        clone_default_11: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_34, memory_format = torch.contiguous_format);  reshape_default_34 = None
        reshape_default_35: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_11, _shape_param_35);  clone_default_11 = _shape_param_35 = None
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
    [4, 8, 1024, 64],  # _shape_param_0
    [4, 8, 64, 1024],  # _shape_param_1
    [4, 1024, 512],  # _shape_param_2
    [4096, 512],  # _shape_param_3
    [4, 1024, 512],  # _shape_param_4
    [4096, 512],  # _shape_param_5
    [4, 8, 1024, 64],  # _shape_param_6
    [4, 8, 64, 1024],  # _shape_param_7
    [4, 1024, 512],  # _shape_param_8
    [4096, 512],  # _shape_param_9
    [4, 1024, 512],  # _shape_param_10
    [4096, 512],  # _shape_param_11
    [4, 8, 1024, 64],  # _shape_param_12
    [4, 8, 64, 1024],  # _shape_param_13
    [4, 1024, 512],  # _shape_param_14
    [4096, 512],  # _shape_param_15
    [4, 1024, 512],  # _shape_param_16
    [4096, 512],  # _shape_param_17
    [4, 8, 1024, 64],  # _shape_param_18
    [4, 8, 64, 1024],  # _shape_param_19
    [4, 1024, 512],  # _shape_param_20
    [4096, 512],  # _shape_param_21
    [4, 1024, 512],  # _shape_param_22
    [4096, 512],  # _shape_param_23
    [4, 8, 1024, 64],  # _shape_param_24
    [4, 8, 64, 1024],  # _shape_param_25
    [4, 1024, 512],  # _shape_param_26
    [4096, 512],  # _shape_param_27
    [4, 1024, 512],  # _shape_param_28
    [4096, 512],  # _shape_param_29
    [4, 8, 1024, 64],  # _shape_param_30
    [4, 8, 64, 1024],  # _shape_param_31
    [4, 1024, 512],  # _shape_param_32
    [4096, 512],  # _shape_param_33
    [4, 1024, 512],  # _shape_param_34
    [4096, 512],  # _shape_param_35
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
