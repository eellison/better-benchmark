"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_as_real_23: "f32[32, 512, 768, 2]", mul_446: "f32[32, 512, 768]", gt: "b8[32, 512, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:180 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_int: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_23, 3, 0);  view_as_real_23 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_446, select_int);  mul_446 = select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:142 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:141 in forward, code: embeddings = self.projection(embeddings)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_1, [16384, 768]);  mul_tensor_1 = None
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(reshape_default, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0], True);  reshape_default = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, [768]);  sum_dim_int_list = None
        return (permute_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 512, 768, 2], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 512, 768], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
