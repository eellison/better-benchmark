"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: b929d7bea283
Shape hash: 3bf9da8b
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
_shapes_config = "(T([1, 64, 64, 64], f16, stride=(4096, 64, 0, 1)), T([1, 64, 64, 192], f16, stride=(0, 0, 192, 1)), T([1, 4096], i64, gen=Index(4096)), S([1, -1, 256]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 64, 64, 64]", arg1_1: "f16[1, 64, 64, 192]", arg2_1: "i64[1, 4096]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:270 in torch_dynamo_resume_in_forward_at_266, code: position_encodings = torch.cat(
        cat_default: "f16[1, 64, 64, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], -1);  arg0_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:273 in torch_dynamo_resume_in_forward_at_266, code: position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))
        reshape_default: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:278 in torch_dynamo_resume_in_forward_at_266, code: torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
        select_int: "f16[4096, 256]" = torch.ops.aten.select.int(reshape_default, 0, 0);  reshape_default = None
        select_int_1: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        index_tensor: "f16[4096, 256]" = torch.ops.aten.index.Tensor(select_int, [select_int_1]);  select_int = select_int_1 = None
        unsqueeze_default: "f16[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index_tensor, 0);  index_tensor = None
        return unsqueeze_default



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
