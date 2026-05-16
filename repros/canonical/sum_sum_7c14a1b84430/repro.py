"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '64', '192'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '64', '1', '64'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 4096, 256]", lt: "b8[8, 64, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:275 in forward, code: position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))
        reshape_default: "f32[8, 64, 64, 256]" = torch.ops.aten.reshape.default(tangents_1, [8, 64, 64, 256]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:273 in forward, code: dropped_weights = dropped_transposed_weights.transpose(2, 1)
        permute_default: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:270 in forward, code: dropped_transposed_weights = nn.functional.dropout2d(
        convert_element_type_default: "f32[8, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.float32);  lt = None
        div_scalar: "f32[8, 64, 1, 1]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.95);  convert_element_type_default = None
        mul_tensor: "f32[8, 64, 64, 256]" = torch.ops.aten.mul.Tensor(permute_default, div_scalar);  permute_default = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:268 in forward, code: transposed_weights = weights.transpose(2, 1)
        permute_default_1: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1, 3]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:266 in forward, code: weights = torch.cat(broadcasted_weights, dim=-1)
        slice_tensor: "f32[8, 64, 64, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64)
        slice_tensor_1: "f32[8, 64, 64, 192]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 256);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:253 in forward, code: weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        sum_dim_int_list: "f32[1, 1, 64, 192]" = torch.ops.aten.sum.dim_IntList(slice_tensor_1, [0, 1], True);  slice_tensor_1 = None
        reshape_default_1: "f32[1, 64, 192]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1, 64, 192]);  sum_dim_int_list = None
        sum_dim_int_list_1: "f32[1, 64, 1, 64]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0, 2], True);  slice_tensor = None
        reshape_default_2: "f32[64, 1, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [64, 1, 64]);  sum_dim_int_list_1 = None
        return (reshape_default_1, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 4096, 256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 64, 1, 1], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
