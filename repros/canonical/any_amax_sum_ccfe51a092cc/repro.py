"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_infer
Pattern hash: ccfe51a092cc
Shape hash: 71fecf09
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
    def forward(self, bmm_default: "f32[64, 1024, 1024]", add_46: "f32[8, 8, 1024, 1024]", mm_88: "f32[8192, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        reshape_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_default, _shape_param_0);  bmm_default = _shape_param_0 = None
        add_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default, add_46);  reshape_default = add_46 = None
        eq_scalar: "b8[8, 8, 1024, 1024]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[8, 8, 1024, 1024]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[8, 8, 1024, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[8, 8, 1024, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.full.default([8, 8, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self: "f32[8, 8, 1024, 1024]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        reshape_default_1: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_88, _shape_param_3);  mm_88 = _shape_param_3 = None
        reshape_default_3: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None

        # No stacktrace found for following nodes
        permute_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        expand_default_1: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_1, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn(67108864, dtype=torch.float32, device='cuda').as_strided([8, 8, 1024, 1024], [8388608, 1, 8192, 8]),  # add_46
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 1024],  # _shape_param_0
    [8, 8, 1024, 1024],  # _shape_param_1
    [64, 1024, 1024],  # _shape_param_2
    [8, 1024, 512],  # _shape_param_3
    [8, 1024, -1, 64],  # _shape_param_4
    [8, 8, 1024, 64],  # _shape_param_5
    [64, 1024, 64],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
