"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: dd4fde67ef99
Shape hash: efc0c4f3
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
_shapes_config = "(T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), T([768, 768], bf16), T([2304, 768], bf16), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]), S([3, 768, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_62: "bf16[768, 768]", mm_66: "bf16[2304, 768]", mm_72: "bf16[768, 768]", mm_76: "bf16[2304, 768]", mm_82: "bf16[768, 768]", mm_86: "bf16[2304, 768]", mm_92: "bf16[768, 768]", mm_96: "bf16[2304, 768]", mm_106: "bf16[768, 768]", mm_110: "bf16[2304, 768]", mm_116: "bf16[768, 768]", mm_120: "bf16[2304, 768]", mm_126: "bf16[768, 768]", mm_130: "bf16[2304, 768]", mm_136: "bf16[768, 768]", mm_140: "bf16[2304, 768]", mm_146: "bf16[768, 768]", mm_150: "bf16[2304, 768]", mm_156: "bf16[768, 768]", mm_160: "bf16[2304, 768]", mm_166: "bf16[768, 768]", mm_170: "bf16[2304, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        full_default: "f32[4, 768, 768]" = torch.ops.aten.full.default([4, 768, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default, 0, 3);  convert_element_type_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_1: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        reshape_default: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_0);  convert_element_type_default_1 = _shape_param_0 = None
        slice_scatter_default: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default, 0, 0, 3);  reshape_default = None
        add_tensor: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default, slice_scatter_default);  select_scatter_default = slice_scatter_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_2: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        select_scatter_default_1: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_2, 0, 3);  convert_element_type_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_3: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        reshape_default_1: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_1);  convert_element_type_default_3 = _shape_param_1 = None
        slice_scatter_default_1: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_1, 0, 0, 3);  reshape_default_1 = None
        add_tensor_1: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_1, slice_scatter_default_1);  select_scatter_default_1 = slice_scatter_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_4: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        select_scatter_default_2: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_4, 0, 3);  convert_element_type_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_5: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        reshape_default_2: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_5, _shape_param_2);  convert_element_type_default_5 = _shape_param_2 = None
        slice_scatter_default_2: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_2, 0, 0, 3);  reshape_default_2 = None
        add_tensor_2: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_2, slice_scatter_default_2);  select_scatter_default_2 = slice_scatter_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_6: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        select_scatter_default_3: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_6, 0, 3);  convert_element_type_default_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_7: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        reshape_default_3: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_7, _shape_param_3);  convert_element_type_default_7 = _shape_param_3 = None
        slice_scatter_default_3: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_3, 0, 0, 3);  reshape_default_3 = None
        add_tensor_3: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_3, slice_scatter_default_3);  select_scatter_default_3 = slice_scatter_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_8: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        select_scatter_default_4: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_8, 0, 3);  convert_element_type_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_9: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        reshape_default_4: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_9, _shape_param_4);  convert_element_type_default_9 = _shape_param_4 = None
        slice_scatter_default_4: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_4, 0, 0, 3);  reshape_default_4 = None
        add_tensor_4: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_4, slice_scatter_default_4);  select_scatter_default_4 = slice_scatter_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_10: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        select_scatter_default_5: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_10, 0, 3);  convert_element_type_default_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_11: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        reshape_default_5: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_11, _shape_param_5);  convert_element_type_default_11 = _shape_param_5 = None
        slice_scatter_default_5: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_5, 0, 0, 3);  reshape_default_5 = None
        add_tensor_5: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_5, slice_scatter_default_5);  select_scatter_default_5 = slice_scatter_default_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_12: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        select_scatter_default_6: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_12, 0, 3);  convert_element_type_default_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_13: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        reshape_default_6: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_13, _shape_param_6);  convert_element_type_default_13 = _shape_param_6 = None
        slice_scatter_default_6: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_6, 0, 0, 3);  reshape_default_6 = None
        add_tensor_6: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_6, slice_scatter_default_6);  select_scatter_default_6 = slice_scatter_default_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_14: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        select_scatter_default_7: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_14, 0, 3);  convert_element_type_default_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_15: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        reshape_default_7: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_15, _shape_param_7);  convert_element_type_default_15 = _shape_param_7 = None
        slice_scatter_default_7: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_7, 0, 0, 3);  reshape_default_7 = None
        add_tensor_7: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_7, slice_scatter_default_7);  select_scatter_default_7 = slice_scatter_default_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_16: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        select_scatter_default_8: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_16, 0, 3);  convert_element_type_default_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_17: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        reshape_default_8: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_17, _shape_param_8);  convert_element_type_default_17 = _shape_param_8 = None
        slice_scatter_default_8: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_8, 0, 0, 3);  reshape_default_8 = None
        add_tensor_8: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_8, slice_scatter_default_8);  select_scatter_default_8 = slice_scatter_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_18: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        select_scatter_default_9: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_18, 0, 3);  convert_element_type_default_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_19: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        reshape_default_9: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_19, _shape_param_9);  convert_element_type_default_19 = _shape_param_9 = None
        slice_scatter_default_9: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_9, 0, 0, 3);  reshape_default_9 = None
        add_tensor_9: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_9, slice_scatter_default_9);  select_scatter_default_9 = slice_scatter_default_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        convert_element_type_default_20: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        select_scatter_default_10: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_20, 0, 3);  convert_element_type_default_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        convert_element_type_default_21: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        reshape_default_10: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_21, _shape_param_10);  convert_element_type_default_21 = _shape_param_10 = None
        slice_scatter_default_10: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default, reshape_default_10, 0, 0, 3);  full_default = reshape_default_10 = None
        add_tensor_10: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_default_10, slice_scatter_default_10);  select_scatter_default_10 = slice_scatter_default_10 = None
        return (add_tensor, add_tensor_7, add_tensor_1, add_tensor_8, add_tensor_10, add_tensor_5, add_tensor_3, add_tensor_4, add_tensor_6, add_tensor_9, add_tensor_2)



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
