"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_training
Pattern hash: 43703965300a
Shape hash: d150ca1d
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
    def forward(self, cumsum: "i64[8, 128]", convert_element_type: "i32[8, 128]", arg1_1: "f32[1026, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:178 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        convert_element_type_default: "i32[8, 128]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add_tensor: "i32[8, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor: "i32[8, 128]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type);  add_tensor = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:179 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_default_1: "i64[8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor_1: "i64[8, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:144 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        reshape_default: "i64[1024]" = torch.ops.aten.reshape.default(add_tensor_1, [-1]);  add_tensor_1 = None
        index_tensor: "f32[1024, 1024]" = torch.ops.aten.index.Tensor(arg1_1, [reshape_default]);  arg1_1 = reshape_default = None
        reshape_default_1: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 128], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [8, 128], dtype=torch.int32, device='cuda'),
    torch.randn([1026, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, 1024],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
