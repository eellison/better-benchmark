"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_training
Pattern hash: feeb7320203d
Shape hash: 66447969
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
    def forward(self, bmm_79: "f32[64, 1024, 64]", _shape_param_0, _shape_param_1, _shape_param_2, primals_61: "f32[512, 512]", mm_280: "f32[8192, 2048]", _shape_param_3, gt_4: "b8[8, 1024, 2048]", le_12: "b8[8, 1024, 2048]", full_default: "f32[]", _shape_param_4, primals_10: "f32[2048, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_79, _shape_param_0);  bmm_79 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_3: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_280, _shape_param_3);  mm_280 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_tensor: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(reshape_default_3, mul_tensor);  reshape_default_3 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_self: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_12, full_default, mul_tensor_1);  le_12 = full_default = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default_4: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_self, _shape_param_4);  where_self = _shape_param_4 = None
        permute_default_3: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_4: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None
        return (reshape_default_2, permute_default_2, reshape_default_4, permute_default_4)


def _default_make_inputs():
    return [
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_0
    [8, 1024, 512],  # _shape_param_1
    [8192, 512],  # _shape_param_2
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    [8, 1024, 2048],  # _shape_param_3
    torch.randint(0, 2, [8, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [8, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [8192, 2048],  # _shape_param_4
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
