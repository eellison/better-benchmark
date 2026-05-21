"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer
Pattern hash: 5cf2067b1ea9
Shape hash: c74518ed
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
_shapes_config = "(T([12000, 384], f16), T([12000, 384], f16), T([12000, 384], f16), S([8, 1500, 384]), S([8, 1500, -1, 64]), S([8, 1500, 384]), S([8, -1, 6, 64]), S([8, 1500, 384]), S([8, -1, 6, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_15: "f16[12000, 384]", mm_3: "f16[12000, 384]", addmm_16: "f16[12000, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        reshape_default: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_15, _shape_param_0);  addmm_15 = _shape_param_0 = None
        mul_tensor: "f16[8, 1500, 384]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None
        reshape_default_1: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        permute_default: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        clone_default: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default_2: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_3, _shape_param_2);  mm_3 = _shape_param_2 = None
        reshape_default_3: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        clone_default_1: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default_4: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_16, _shape_param_4);  addmm_16 = _shape_param_4 = None
        reshape_default_5: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        clone_default_2: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        return (clone_default, clone_default_1, clone_default_2)



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
