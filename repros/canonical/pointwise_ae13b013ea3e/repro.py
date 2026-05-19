"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: ae13b013ea3e
Shape hash: 90974ccb
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
    def forward(self, mm: "bf16[512, 1024]", mm_1: "bf16[512, 1024]", embedding: "bf16[512, 1, 1024]", _param_constant1: "bf16[16, 64]", _param_constant2: "bf16[16, 64]", mm_10: "bf16[1024, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        unsqueeze_default: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_default: "bf16[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default, [0, 2, 3, 4, 1]);  view_default = None
        view_default_1: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default_1: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm_1, 0);  mm_1 = None
        view_default_2: "bf16[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default_1, _shape_param_2);  unsqueeze_default_1 = _shape_param_2 = None
        permute_default_1: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 3, 4, 1]);  view_default_2 = None
        view_default_3: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        unsqueeze_default_2: "bf16[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3);  embedding = None
        unsqueeze_default_3: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        permute_default_2: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_3, [0, 1, 3, 4, 2]);  unsqueeze_default_3 = None
        permute_default_3: "bf16[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_2, [0, 4, 1, 2, 3]);  permute_default_2 = None
        view_default_4: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(permute_default_3, _shape_param_4);  permute_default_3 = _shape_param_4 = None
        squeeze_dim: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default_4, 0);  view_default_4 = None
        add_tensor: "bf16[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_default_1, _param_constant1);  _param_constant1 = None
        unsqueeze_default_4: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 4);  add_tensor = None
        permute_default_4: "bf16[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_4, [1, 2, 0, 4, 3]);  unsqueeze_default_4 = None
        unsqueeze_default_5: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_3, 4);  view_default_3 = None
        permute_default_5: "bf16[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default_5, [1, 2, 4, 0, 3]);  unsqueeze_default_5 = None
        permute_default_6: "bf16[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_4, [1, 2, 4, 0, 3]);  permute_default_4 = None
        view_default_5: "bf16[16, 512, 64]" = torch.ops.aten.view.default(permute_default_6, _shape_param_5);  permute_default_6 = _shape_param_5 = None
        permute_default_7: "bf16[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_default_5, [1, 4, 0, 3, 2]);  permute_default_5 = None
        view_default_6: "bf16[16, 64, 512]" = torch.ops.aten.view.default(permute_default_7, _shape_param_6);  permute_default_7 = _shape_param_6 = None
        add_tensor_1: "bf16[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_default_1, _param_constant2);  view_default_1 = _param_constant2 = None
        unsqueeze_default_6: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 4);  add_tensor_1 = None
        permute_default_8: "bf16[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_6, [1, 2, 0, 4, 3]);  unsqueeze_default_6 = None
        permute_default_9: "bf16[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_8, [1, 2, 4, 0, 3]);  permute_default_8 = None
        view_default_7: "bf16[16, 512, 64]" = torch.ops.aten.view.default(permute_default_9, _shape_param_7);  permute_default_9 = _shape_param_7 = None
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(mm_10, [1024, 1024, 1024], 1);  mm_10 = None
        getitem: "bf16[1024, 1024]" = split_with_sizes_default[0]
        getitem_1: "bf16[1024, 1024]" = split_with_sizes_default[1]
        getitem_2: "bf16[1024, 1024]" = split_with_sizes_default[2];  split_with_sizes_default = None
        unsqueeze_default_7: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(getitem_1, 0);  getitem_1 = None
        view_default_8: "bf16[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default_7, _shape_param_8);  unsqueeze_default_7 = _shape_param_8 = None
        permute_default_10: "bf16[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default_8, [0, 2, 3, 4, 1]);  view_default_8 = None
        view_default_9: "bf16[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_default_10, _shape_param_9);  permute_default_10 = _shape_param_9 = None
        unsqueeze_default_8: "bf16[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_9, 4);  view_default_9 = None
        permute_default_11: "bf16[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_default_8, [1, 2, 4, 0, 3]);  unsqueeze_default_8 = None
        permute_default_12: "bf16[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_default_11, [1, 4, 0, 3, 2]);  permute_default_11 = None
        view_default_10: "bf16[16, 64, 1024]" = torch.ops.aten.view.default(permute_default_12, _shape_param_10);  permute_default_12 = _shape_param_10 = None
        return (squeeze_dim, view_default_5, view_default_6, view_default_7, view_default_10, getitem, getitem_2)


def _default_make_inputs():
    return [
    torch.randn([512, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([16, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([16, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 3072], dtype=torch.bfloat16, device='cuda'),
    [512, 1, 1, 16, 64],  # _shape_param_0
    [512, 1, 16, 64],  # _shape_param_1
    [512, 1, 1, 16, 64],  # _shape_param_2
    [512, 1, 16, 64],  # _shape_param_3
    [1, 512, 1024],  # _shape_param_4
    [16, 512, 64],  # _shape_param_5
    [16, 64, 512],  # _shape_param_6
    [16, 512, 64],  # _shape_param_7
    [1024, 1, 1, 16, 64],  # _shape_param_8
    [1024, 1, 16, 64],  # _shape_param_9
    [16, 64, 1024],  # _shape_param_10
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
