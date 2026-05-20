"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: fa72e65c62dd
Shape hash: 750e38fa
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1024, 128, 32], f32), T([128, 512], f32), T([32768, 128], f32), T([32768, 128], f32), T([128], f32), T([128, 512], f32), T([256, 128, 128], f32), T([128], f32), T([128, 512], f32), S([256, 4, 128, 32]), S([256, 128, 128]), S([32768, 128]), S([256, 128, 128]), S([256, 128, 128]), S([32768, 128]), S([32768, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_140: "f32[1024, 128, 32]", primals_22: "f32[128, 512]", mm_717: "f32[32768, 128]", mm_719: "f32[32768, 128]", primals_16: "f32[128]", primals_14: "f32[128, 512]", mul_794: "f32[256, 128, 128]", primals_12: "f32[128]", primals_10: "f32[128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_0);  bmm_140 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_default: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[32768, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[512, 128]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_default_2: "f32[128, 512]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(mm_717, _shape_param_3);  mm_717 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default_4: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(mm_719, _shape_param_4);  mm_719 = _shape_param_4 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_3, reshape_default_4);  reshape_default_3 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, primals_16);  add_tensor = primals_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_5: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_5);  mul_tensor = _shape_param_5 = None
        permute_default_3: "f32[512, 128]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_4: "f32[128, 512]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(mul_794, primals_12);  mul_794 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_6: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_6);  mul_tensor_1 = _shape_param_6 = None
        permute_default_5: "f32[512, 128]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_6: "f32[128, 512]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None
        return (reshape_default_2, permute_default_2, reshape_default_5, permute_default_4, reshape_default_6, permute_default_6)


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
