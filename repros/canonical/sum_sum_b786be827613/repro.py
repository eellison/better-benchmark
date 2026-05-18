"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['32', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_156: "f32[4096, 512]", tangents_3: "f32[32, 128, 512]", mm_158: "f32[4096, 512]", mm_178: "f32[4096, 512]", mm_180: "f32[4096, 512]", mm_200: "f32[4096, 512]", mm_202: "f32[4096, 512]", mm_222: "f32[4096, 512]", mm_224: "f32[4096, 512]", mm_244: "f32[4096, 512]", mm_246: "f32[4096, 512]", mm_266: "f32[4096, 512]", mm_268: "f32[4096, 512]", mm_288: "f32[4096, 512]", mm_290: "f32[4096, 512]", mm_310: "f32[4096, 512]", mm_312: "f32[4096, 512]", gt_34: "b8[32, 128, 512]", primals_76: "f32[512]", add_59: "f32[32, 128, 512]", rsqrt_16: "f32[32, 128, 1]", gt_33: "b8[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_156, _shape_param_0);  mm_156 = _shape_param_0 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(tangents_3, reshape_default);  tangents_3 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_1: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_158, _shape_param_1);  mm_158 = _shape_param_1 = None
        add_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default_2: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_178, _shape_param_2);  mm_178 = _shape_param_2 = None
        add_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_3: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_180, _shape_param_3);  mm_180 = _shape_param_3 = None
        add_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_3);  add_tensor_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default_4: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_200, _shape_param_4);  mm_200 = _shape_param_4 = None
        add_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_4);  add_tensor_3 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_5: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_202, _shape_param_5);  mm_202 = _shape_param_5 = None
        add_tensor_5: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_5);  add_tensor_4 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default_6: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_222, _shape_param_6);  mm_222 = _shape_param_6 = None
        add_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_6);  add_tensor_5 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_7: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_224, _shape_param_7);  mm_224 = _shape_param_7 = None
        add_tensor_7: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_7);  add_tensor_6 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default_8: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_244, _shape_param_8);  mm_244 = _shape_param_8 = None
        add_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_8);  add_tensor_7 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_9: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_246, _shape_param_9);  mm_246 = _shape_param_9 = None
        add_tensor_9: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_9);  add_tensor_8 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default_10: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_266, _shape_param_10);  mm_266 = _shape_param_10 = None
        add_tensor_10: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_10);  add_tensor_9 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_11: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_268, _shape_param_11);  mm_268 = _shape_param_11 = None
        add_tensor_11: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_10, reshape_default_11);  add_tensor_10 = reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default_12: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_288, _shape_param_12);  mm_288 = _shape_param_12 = None
        add_tensor_12: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_11, reshape_default_12);  add_tensor_11 = reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_13: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_290, _shape_param_13);  mm_290 = _shape_param_13 = None
        add_tensor_13: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_12, reshape_default_13);  add_tensor_12 = reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:389 in forward, code: value_states = self.v(current_states)
        reshape_default_14: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_310, _shape_param_14);  mm_310 = _shape_param_14 = None
        add_tensor_14: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_13, reshape_default_14);  add_tensor_13 = reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:388 in forward, code: key_states = self.k(current_states)
        reshape_default_15: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_312, _shape_param_15);  mm_312 = _shape_param_15 = None
        add_tensor_15: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_14, reshape_default_15);  add_tensor_14 = reshape_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1124 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_15, mul_tensor);  add_tensor_15 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:147 in forward, code: return self.weight * hidden_states
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_76);  primals_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:141 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:147 in forward, code: return self.weight * hidden_states
        mul_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = mul_tensor_3 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        reshape_default_16: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_16);  sum_dim_int_list = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:141 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_59)
        mul_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_16);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        pow_tensor_scalar: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_scalar: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_7, _shape_param_17);  mul_tensor_7 = _shape_param_17 = None
        div_scalar: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_59, 1.0);  add_59 = None
        mul_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_16: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:218 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_tensor_9: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_10: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_16, mul_tensor_9);  add_tensor_16 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:199 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_17: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_10, _shape_param_18);  mul_tensor_10 = _shape_param_18 = None
        permute_default: "f32[512, 4096]" = torch.ops.aten.permute.default(reshape_default_17, [1, 0]);  reshape_default_17 = None
        return (reshape_default_16, permute_default)


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128, 512], dtype=torch.bool, device='cuda'),
    [32, 128, 512],  # _shape_param_0
    [32, 128, 512],  # _shape_param_1
    [32, 128, 512],  # _shape_param_2
    [32, 128, 512],  # _shape_param_3
    [32, 128, 512],  # _shape_param_4
    [32, 128, 512],  # _shape_param_5
    [32, 128, 512],  # _shape_param_6
    [32, 128, 512],  # _shape_param_7
    [32, 128, 512],  # _shape_param_8
    [32, 128, 512],  # _shape_param_9
    [32, 128, 512],  # _shape_param_10
    [32, 128, 512],  # _shape_param_11
    [32, 128, 512],  # _shape_param_12
    [32, 128, 512],  # _shape_param_13
    [32, 128, 512],  # _shape_param_14
    [32, 128, 512],  # _shape_param_15
    [512],  # _shape_param_16
    [32, 128, 512],  # _shape_param_17
    [4096, 512],  # _shape_param_18
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
