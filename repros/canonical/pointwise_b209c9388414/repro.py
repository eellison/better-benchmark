"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: b209c9388414
Shape hash: f8c6ddfc
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
    def forward(self, view_949: "f32[32, 512, 12, 64]", addmm_2: "f32[16384, 384]", mm_201: "f32[16384, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_tensor: "f32[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_949, 2, 0, 6);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_default: "f32[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  _shape_param_1 = None

        # No stacktrace found for following nodes
        permute_default_1: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        reshape_default_2: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(mm_201, _shape_param_2);  mm_201 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor: "f32[32, 512, 384]" = torch.ops.aten.mul.Tensor(reshape_default_2, reshape_default);  reshape_default_2 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_2: "f32[32, 384, 512]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None
        return (clone_default, permute_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([32, 512, 12, 64], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    [32, 512, 384],  # _shape_param_0
    [32, 512, -1, 64],  # _shape_param_1
    [32, 512, 384],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
