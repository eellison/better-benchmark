"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 11cbe48c9258
Shape hash: b8328eba
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_408: "f32[4096, 1024]", gt_8: "b8[32, 128, 1024]", mm_11: "f32[4096, 1024]", mm_12: "f32[4096, 1024]", primals_20: "f32[1024, 512]", primals_19: "f32[1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_408, _shape_param_0);  mm_408 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_1: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_11, _shape_param_1);  mm_11 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        pow_tensor_scalar: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_tensor_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_2: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_12, _shape_param_2);  mm_12 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_tensor_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, reshape_default_2);  mul_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_3: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        permute_default: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_default_1: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_8: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_1);  mul_tensor_7 = add_tensor_1 = None
        mul_tensor_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_tensor_10);  mul_tensor_10 = None
        mul_tensor_11: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_8, sub_tensor);  mul_tensor_8 = sub_tensor = None
        mul_tensor_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.7978845608028654);  mul_tensor_11 = None
        mul_tensor_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.044715)
        pow_tensor_scalar_1: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_scalar);  mul_tensor_13 = mul_scalar = None
        add_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_14);  mul_tensor_12 = mul_tensor_14 = None
        mul_tensor_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_9, 0.5);  mul_tensor_9 = None
        add_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_15);  add_tensor_2 = mul_tensor_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_4: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_4);  add_tensor_3 = _shape_param_4 = None
        permute_default_2: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_default_3: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_3, permute_default_1, reshape_default_4, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [32, 128, 1024],  # _shape_param_0
    [32, 128, 1024],  # _shape_param_1
    [32, 128, 1024],  # _shape_param_2
    [4096, 1024],  # _shape_param_3
    [4096, 1024],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
