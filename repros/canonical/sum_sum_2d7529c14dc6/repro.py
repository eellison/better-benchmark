"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_training
Pattern hash: 2d7529c14dc6
Shape hash: eb21941f
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
    def forward(self, mm_2: "f32[1024, 512]", _shape_param_0, primals_1116: "f32[512]", addmm_361: "f32[1024, 512]", _shape_param_1, getitem_1: "f32[8, 128, 1]", rsqrt: "f32[8, 128, 1]", full_default_2: "f32[]", _shape_param_2, primals_1114: "f32[512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        reshape_default: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(reshape_default, primals_1116);  reshape_default = primals_1116 = None
        mul_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[8, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:489 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(addmm_361, _shape_param_1);  addmm_361 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:490 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        relu_default: "f32[8, 128, 512]" = torch.ops.aten.relu.default(reshape_default_1);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[8, 128, 512]" = torch.ops.aten.sub.Tensor(relu_default, getitem_1);  getitem_1 = None
        mul_tensor_2: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 128, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[8, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
        mul_tensor_5: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:490 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        le_scalar: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[8, 128, 512]" = torch.ops.aten.where.self(le_scalar, full_default_2, mul_tensor_5);  le_scalar = full_default_2 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:489 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[1024, 512]" = torch.ops.aten.reshape.default(where_self, _shape_param_2);  where_self = _shape_param_2 = None
        permute_default: "f32[512, 512]" = torch.ops.aten.permute.default(primals_1114, [1, 0]);  primals_1114 = None
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_0
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_1
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_2
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
