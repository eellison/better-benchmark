"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: bb3366af4995
Shape hash: 15445790
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([16, 16, 512, 64], f32, stride=(524288, 64, 1024, 1)), T([16, 16, 512, 64], f32, stride=(524288, 64, 1024, 1)), T([16, 16, 512, 64], f32, stride=(524288, 64, 1024, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([16, 512, 1024]), S([8192, 1024]), S([16, 512, 1024]), S([8192, 1024]), S([16, 512, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_258: "f32[16, 16, 512, 64]", getitem_259: "f32[16, 16, 512, 64]", getitem_260: "f32[16, 16, 512, 64]", primals_29: "f32[1024, 1024]", primals_27: "f32[1024, 1024]", primals_25: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_258, [0, 2, 1, 3]);  getitem_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_1: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_259, [0, 2, 1, 3]);  getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_2: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        clone_default: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_default_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_default_1: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        permute_default_5: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_default_6: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_default_2: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_4: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_4);  clone_default_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        reshape_default_5: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_default_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_7, [1, 0]);  permute_default_7 = None
        return (reshape_default_1, permute_default_4, reshape_default_3, permute_default_6, reshape_default_5, permute_default_8)


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
