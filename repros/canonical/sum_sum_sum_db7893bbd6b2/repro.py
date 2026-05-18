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
#   type=sum, ranges=['1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[8192, 1024]", gt_98: "b8[512, 16, 1024]", primals_361: "f32[1024]", mul_388: "f32[512, 16, 1024]", div_27: "f32[512, 16, 1]", gt_97: "b8[512, 16, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1633 in forward, code: logits = self.lm_loss(transformer_outputs[0])
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1391 in forward, code: output = output.permute(1, 0, 2).contiguous()
        permute_default: "f32[512, 16, 1024]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1388 in forward, code: output = self.dropout(output_g if output_g is not None else output_h)
        convert_element_type_default: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(gt_98, torch.float32);  gt_98 = None
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(permute_default, mul_tensor);  permute_default = mul_tensor = None
        clone_default: "f32[512, 16, 1024]" = torch.ops.aten.clone.default(mul_tensor_1, memory_format = torch.contiguous_format);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:468 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(clone_default, primals_361);  primals_361 = None
        mul_tensor_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1024)
        sum_dim_int_list: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_388);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_388, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(div_27, sub_tensor_1);  div_27 = sub_tensor_1 = None
        mul_tensor_7: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(clone_default, mul_388);  mul_388 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(clone_default, [0, 1]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:467 in forward, code: output = self.dropout(output)
        convert_element_type_default_1: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(gt_97, torch.float32);  gt_97 = None
        mul_tensor_8: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_9: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:466 in forward, code: output = self.layer_2(output)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_tensor_9, _shape_param_1);  mul_tensor_9 = _shape_param_1 = None
        permute_default_1: "f32[1024, 8192]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [0], True);  reshape_default_1 = None
        reshape_default_2: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default_1, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [512, 16, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [512, 16, 1024], dtype=torch.bool, device='cuda'),
    [16, 512, 1024],  # _shape_param_0
    [8192, 1024],  # _shape_param_1
    [1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
