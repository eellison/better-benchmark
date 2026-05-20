"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 2d02b82b3da4
Shape hash: d6847ab3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([384, 768], f32), T([384, 768], f32), T([32, 384, 512], f32), T([384, 1], f32), T([16384, 384], f32), T([54, 384], f32), T([384, 768], f32), S([32, 512, 384]), S([16384, 384]))"

class Repro(torch.nn.Module):
    def forward(self, primals_262: "f32[384, 768]", primals_264: "f32[384, 768]", convolution_23: "f32[32, 384, 512]", primals_268: "f32[384, 1]", addmm_79: "f32[16384, 384]", primals_271: "f32[54, 384]", primals_273: "f32[384, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default: "f32[768, 384]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_1: "f32[768, 384]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_tensor: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, primals_268);  convolution_23 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_79, _shape_param_0);  addmm_79 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_2: "f32[32, 512, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor: "f32[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute_default_2, reshape_default);  permute_default_2 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_default_3: "f32[384, 54]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        clone_default: "f32[32, 512, 384]" = torch.ops.aten.clone.default(mul_tensor, memory_format = torch.contiguous_format);  mul_tensor = None
        reshape_default_1: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_4: "f32[768, 384]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
        return (permute_default, permute_default_1, permute_default_3, reshape_default_1, permute_default_4)


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
