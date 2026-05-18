"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['512', '16', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['512', '16', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_96: "f32[8192, 1024]", mul_1114: "f32[512, 16, 1024]", primals_10: "f32[1024]", mul_11: "f32[512, 16, 1024]", div_74: "f32[512, 16, 1]", gt_3: "b8[512, 16, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:463 in forward, code: output = self.layer_1(output)
        reshape_default: "f32[512, 16, 1024]" = torch.ops.aten.reshape.default(mm_96, [512, 16, 1024]);  mm_96 = None
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_1114, reshape_default);  mul_1114 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:309 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, primals_10);  primals_10 = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_11);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_11, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(div_74, sub_tensor_1);  div_74 = sub_tensor_1 = None
        mul_tensor_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, mul_11);  mul_11 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:306 in post_attention, code: attn_out = self.dropout(attn_out)
        convert_element_type_default: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_tensor_6: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_7: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        reshape_default_1: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_7, [512, 16, 1024, 1, 1]);  mul_tensor_7 = None
        permute_default: "f32[512, 16, 1, 1, 1024]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 4, 2]);  reshape_default_1 = None
        reshape_default_2: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(permute_default, [1, 8192, 1024]);  permute_default = None
        squeeze_dim: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, squeeze_dim)


def _default_make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [512, 16, 1024], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
