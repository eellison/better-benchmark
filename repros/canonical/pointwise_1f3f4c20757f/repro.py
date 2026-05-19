"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 1f3f4c20757f
Shape hash: 48f7cda3
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
    def forward(self, bmm_142: "f32[1024, 32, 128]", bmm_143: "f32[1024, 128, 32]", primals_20: "f32[128, 128]", primals_18: "f32[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[256, 4, 32, 128]" = torch.ops.aten.reshape.default(bmm_142, _shape_param_0);  bmm_142 = _shape_param_0 = None
        reshape_default_1: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_1);  bmm_143 = _shape_param_1 = None
        mul_scalar: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(reshape_default, 0.4204482076268573);  reshape_default = None
        permute_default: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        mul_scalar_1: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.4204482076268573);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_default_1: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_2: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        clone_default: "f32[256, 128, 128]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_3: "f32[32768, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        permute_default_2: "f32[128, 128]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_default_3: "f32[128, 128]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        permute_default_4: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(mul_scalar_1, [0, 2, 1, 3]);  mul_scalar_1 = None
        clone_default_1: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_4: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        reshape_default_5: "f32[32768, 128]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_5: "f32[128, 128]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_6: "f32[128, 128]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None
        return (reshape_default_3, permute_default_3, reshape_default_5, permute_default_6)


def _default_make_inputs():
    return [
    torch.randn([1024, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
    [256, 4, 32, 128],  # _shape_param_0
    [256, 4, 128, 32],  # _shape_param_1
    [256, 128, 128],  # _shape_param_2
    [32768, 128],  # _shape_param_3
    [256, 128, 128],  # _shape_param_4
    [32768, 128],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
