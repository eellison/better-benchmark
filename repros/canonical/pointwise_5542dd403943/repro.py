"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_training
Pattern hash: 5542dd403943
Shape hash: 26906a14
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
    def forward(self, bmm_24: "f32[512, 512, 64]", _shape_param_0, bmm_26: "f32[512, 64, 512]", _shape_param_1, bmm_27: "f32[512, 512, 64]", _shape_param_2, _shape_param_3, _shape_param_4, primals_15: "f32[4096, 4096]", _shape_param_5, _shape_param_6, primals_13: "f32[4096, 4096]", _shape_param_7, _shape_param_8, primals_11: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_24, _shape_param_0);  bmm_24 = _shape_param_0 = None
        reshape_default_1: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_26, _shape_param_1);  bmm_26 = _shape_param_1 = None
        reshape_default_2: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, _shape_param_2);  bmm_27 = _shape_param_2 = None
        mul_scalar: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.3535533905932738);  reshape_default_1 = None
        permute_default: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        mul_scalar_1: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(reshape_default_2, 0.3535533905932738);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_1: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[4096, 4096]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_default_3: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_4: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_5: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_5);  permute_default_4 = _shape_param_5 = None
        clone_default_1: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_5: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_default_6: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_7: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_scalar_1, [0, 2, 1, 3]);  mul_scalar_1 = None
        clone_default_2: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[4096, 4096]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_8: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_9: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_4, permute_default_3, reshape_default_6, permute_default_6, reshape_default_8, permute_default_9)


def _default_make_inputs():
    return [
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    [8, 64, 512, 64],  # _shape_param_0
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    [8, 64, 64, 512],  # _shape_param_1
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    [8, 64, 512, 64],  # _shape_param_2
    [8, 512, 4096],  # _shape_param_3
    [4096, 4096],  # _shape_param_4
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [8, 512, 4096],  # _shape_param_5
    [4096, 4096],  # _shape_param_6
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [8, 512, 4096],  # _shape_param_7
    [4096, 4096],  # _shape_param_8
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
