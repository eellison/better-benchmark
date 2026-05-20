"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: de48dad9bc4d
Shape hash: d2deffe4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([64, 128], i64, max=128), T([64, 128], i32, max=128), T([1026, 1024], f32), S([64, 128, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[64, 128]", convert_element_type: "i32[64, 128]", arg1_1: "f32[1026, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:178 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add_tensor: "i32[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor: "i32[64, 128]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type);  add_tensor = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:179 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_default_1: "i64[64, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor_1: "i64[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:144 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        reshape_default: "i64[8192]" = torch.ops.aten.reshape.default(add_tensor_1, [-1]);  add_tensor_1 = None
        index_tensor: "f32[8192, 1024]" = torch.ops.aten.index.Tensor(arg1_1, [reshape_default]);  arg1_1 = reshape_default = None
        reshape_default_1: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        return reshape_default_1


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
