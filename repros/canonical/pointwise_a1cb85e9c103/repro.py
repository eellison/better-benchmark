"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 64, 64, 64]", arg1_1: "f32[8, 64, 64, 192]", arg2_1: "i64[8, 4096]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:296 in torch_dynamo_resume_in_forward_at_292, code: position_encodings = torch.cat(
        cat_default: "f32[8, 64, 64, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], -1);  arg0_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:299 in torch_dynamo_resume_in_forward_at_292, code: position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))
        reshape_default: "f32[8, 4096, 256]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:304 in torch_dynamo_resume_in_forward_at_292, code: torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
        select_int: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 0)
        select_int_1: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 0)
        index_tensor: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int, [select_int_1]);  select_int = select_int_1 = None
        unsqueeze_default: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor, 0);  index_tensor = None
        select_int_2: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 1)
        select_int_3: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 1)
        index_tensor_1: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int_2, [select_int_3]);  select_int_2 = select_int_3 = None
        unsqueeze_default_1: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor_1, 0);  index_tensor_1 = None
        select_int_4: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 2)
        select_int_5: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 2)
        index_tensor_2: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int_4, [select_int_5]);  select_int_4 = select_int_5 = None
        unsqueeze_default_2: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor_2, 0);  index_tensor_2 = None
        select_int_6: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 3)
        select_int_7: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 3)
        index_tensor_3: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int_6, [select_int_7]);  select_int_6 = select_int_7 = None
        unsqueeze_default_3: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor_3, 0);  index_tensor_3 = None
        select_int_8: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 4)
        select_int_9: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 4)
        index_tensor_4: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int_8, [select_int_9]);  select_int_8 = select_int_9 = None
        unsqueeze_default_4: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor_4, 0);  index_tensor_4 = None
        select_int_10: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 5)
        select_int_11: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 5)
        index_tensor_5: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int_10, [select_int_11]);  select_int_10 = select_int_11 = None
        unsqueeze_default_5: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor_5, 0);  index_tensor_5 = None
        select_int_12: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 6)
        select_int_13: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 6)
        index_tensor_6: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int_12, [select_int_13]);  select_int_12 = select_int_13 = None
        unsqueeze_default_6: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor_6, 0);  index_tensor_6 = None
        select_int_14: "f32[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 7);  reshape_default = None
        select_int_15: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 7);  arg2_1 = None
        index_tensor_7: "f32[4096, 256]" = torch.ops.aten.index.Tensor(select_int_14, [select_int_15]);  select_int_14 = select_int_15 = None
        unsqueeze_default_7: "f32[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor_7, 0);  index_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:302 in torch_dynamo_resume_in_forward_at_292, code: position_encodings = torch.cat(
        cat_default_1: "f32[8, 4096, 256]" = torch.ops.aten.cat.default([unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, unsqueeze_default_6, unsqueeze_default_7]);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = unsqueeze_default_6 = unsqueeze_default_7 = None
        return cat_default_1


def _default_make_inputs():
    return [
    torch.randn([8, 64, 64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 64, 64, 192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 4096], dtype=torch.int64, device='cuda'),
    [8, -1, 256],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
