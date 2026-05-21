"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: 8b2bd28722b3
Shape hash: e3dba49e
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_14: "f32[256, 32, 33]", arg85_1: "f32[64, 1024, 8, 64]", mm_51: "f32[1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_14, _shape_param_0);  bmm_14 = _shape_param_0 = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(reshape_default, 1);  reshape_default = None
        amax_default: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor, 8.0);  sub_tensor = None
        exp_default: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        sum_dim_int_list: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        expand_default: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_tensor_1, _shape_param_1);  div_tensor_1 = _shape_param_1 = None
        reshape_default_1: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        slice_tensor: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg85_1, 0, 0, 32)
        slice_tensor_1: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg85_1, 0, 0, 32)
        slice_tensor_2: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_tensor_1, 1, 1, 33);  slice_tensor_1 = None
        reshape_default_2: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_51, _shape_param_3);  mm_51 = _shape_param_3 = None
        reshape_default_3: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        copy_default: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_tensor_2, reshape_default_3);  slice_tensor_2 = reshape_default_3 = None
        slice_scatter_default: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default, 1, 1, 33);  slice_tensor = copy_default = None
        slice_scatter_default_1: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg85_1, slice_scatter_default, 0, 0, 32);  arg85_1 = slice_scatter_default = None
        slice_tensor_3: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_default_1, 0, 0, 32);  slice_scatter_default_1 = None
        slice_tensor_4: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_tensor_3, 1, 0, 33);  slice_tensor_3 = None
        permute_default: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_tensor_4, [0, 2, 1, 3]);  slice_tensor_4 = None
        expand_default_1: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_1, reshape_default_4)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
