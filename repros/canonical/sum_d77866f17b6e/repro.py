"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train
Pattern hash: d77866f17b6e
Shape hash: 0b47cc3e
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
_shapes_config = "(T([8192, 768], f32), T([8, 1024, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8, 1024, 768], b8), T([768], f32), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], b8), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_202: "f32[8192, 768]", tangents_3: "f32[8, 1024, 768]", mm_204: "f32[8192, 768]", mm_222: "f32[8192, 768]", mm_224: "f32[8192, 768]", mm_242: "f32[8192, 768]", mm_244: "f32[8192, 768]", mm_262: "f32[8192, 768]", mm_264: "f32[8192, 768]", mm_282: "f32[8192, 768]", mm_284: "f32[8192, 768]", mm_302: "f32[8192, 768]", mm_304: "f32[8192, 768]", mm_322: "f32[8192, 768]", mm_324: "f32[8192, 768]", mm_342: "f32[8192, 768]", mm_344: "f32[8192, 768]", mm_362: "f32[8192, 768]", mm_364: "f32[8192, 768]", mm_382: "f32[8192, 768]", mm_384: "f32[8192, 768]", mm_402: "f32[8192, 768]", mm_404: "f32[8192, 768]", mm_422: "f32[8192, 768]", mm_424: "f32[8192, 768]", gt_50: "b8[8, 1024, 768]", primals_100: "f32[768]", add_66: "f32[8, 1024, 768]", rsqrt_24: "f32[8, 1024, 1]", gt_49: "b8[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_202, _shape_param_0);  mm_202 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(tangents_3, reshape_default);  tangents_3 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_204, _shape_param_1);  mm_204 = _shape_param_1 = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_222, _shape_param_2);  mm_222 = _shape_param_2 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_224, _shape_param_3);  mm_224 = _shape_param_3 = None
        add_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_3);  add_tensor_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_4: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_242, _shape_param_4);  mm_242 = _shape_param_4 = None
        add_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_4);  add_tensor_3 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_5: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_244, _shape_param_5);  mm_244 = _shape_param_5 = None
        add_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_5);  add_tensor_4 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_6: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_262, _shape_param_6);  mm_262 = _shape_param_6 = None
        add_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_6);  add_tensor_5 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_7: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_264, _shape_param_7);  mm_264 = _shape_param_7 = None
        add_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_7);  add_tensor_6 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_8: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_282, _shape_param_8);  mm_282 = _shape_param_8 = None
        add_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_8);  add_tensor_7 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_9: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_284, _shape_param_9);  mm_284 = _shape_param_9 = None
        add_tensor_9: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_9);  add_tensor_8 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_10: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_302, _shape_param_10);  mm_302 = _shape_param_10 = None
        add_tensor_10: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_10);  add_tensor_9 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_11: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_304, _shape_param_11);  mm_304 = _shape_param_11 = None
        add_tensor_11: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_10, reshape_default_11);  add_tensor_10 = reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_12: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_322, _shape_param_12);  mm_322 = _shape_param_12 = None
        add_tensor_12: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_11, reshape_default_12);  add_tensor_11 = reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_13: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_324, _shape_param_13);  mm_324 = _shape_param_13 = None
        add_tensor_13: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_12, reshape_default_13);  add_tensor_12 = reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_14: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_342, _shape_param_14);  mm_342 = _shape_param_14 = None
        add_tensor_14: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_13, reshape_default_14);  add_tensor_13 = reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_15: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_344, _shape_param_15);  mm_344 = _shape_param_15 = None
        add_tensor_15: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_14, reshape_default_15);  add_tensor_14 = reshape_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_16: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_362, _shape_param_16);  mm_362 = _shape_param_16 = None
        add_tensor_16: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_15, reshape_default_16);  add_tensor_15 = reshape_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_17: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_364, _shape_param_17);  mm_364 = _shape_param_17 = None
        add_tensor_17: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_16, reshape_default_17);  add_tensor_16 = reshape_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_18: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_382, _shape_param_18);  mm_382 = _shape_param_18 = None
        add_tensor_18: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_17, reshape_default_18);  add_tensor_17 = reshape_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_19: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_384, _shape_param_19);  mm_384 = _shape_param_19 = None
        add_tensor_19: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_18, reshape_default_19);  add_tensor_18 = reshape_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_20: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_402, _shape_param_20);  mm_402 = _shape_param_20 = None
        add_tensor_20: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_19, reshape_default_20);  add_tensor_19 = reshape_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_21: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_404, _shape_param_21);  mm_404 = _shape_param_21 = None
        add_tensor_21: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_20, reshape_default_21);  add_tensor_20 = reshape_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_22: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_422, _shape_param_22);  mm_422 = _shape_param_22 = None
        add_tensor_22: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_21, reshape_default_22);  add_tensor_21 = reshape_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_23: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_424, _shape_param_23);  mm_424 = _shape_param_23 = None
        add_tensor_23: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_22, reshape_default_23);  add_tensor_22 = reshape_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_23, mul_tensor);  add_tensor_23 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_100);  mul_tensor_1 = primals_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_66)
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_24);  mul_tensor_2 = None
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_5: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_24);  mul_tensor_5 = _shape_param_24 = None
        div_scalar: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_66, 1.0);  add_66 = None
        mul_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_24: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default_1: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_24, mul_tensor_7);  add_tensor_24 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_24: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_25);  mul_tensor_8 = _shape_param_25 = None
        return reshape_default_24



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
