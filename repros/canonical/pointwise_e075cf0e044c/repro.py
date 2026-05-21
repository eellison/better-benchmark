"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_infer
Pattern hash: e075cf0e044c
Shape hash: eeff1187
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
_shapes_config = "(T([2048, 512], f16), T([2048, 512], f16), T([1, 2048, 512], f16), T([512, 512], f16), T([8, 2048, 64], f16), T([512, 512], f16), S([1, 2048, 512]), S([1, 2048, -1, 64]), S([1, 8, 2048, 64]), S([8, 2048, 64]), S([1, 2048, 512]), S([1, 2048, -1, 64]), S([1, 8, 64, 2048]), S([8, 64, 2048]), S([2048, 512]), S([1, 8, 2048, 64]), S([1, 2048, -1]), S([2048, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_36: "f16[2048, 512]", mm_37: "f16[2048, 512]", mul_29: "f16[1, 2048, 512]", arg55_1: "f16[512, 512]", bmm_11: "f16[8, 2048, 64]", arg47_1: "f16[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_36, _shape_param_0);  mm_36 = _shape_param_0 = None
        reshape_default_1: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_default: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        reshape_default_2: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_37, _shape_param_4);  mm_37 = _shape_param_4 = None
        reshape_default_4: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None
        permute_default_1: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_2: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default_1: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_6);  permute_default_2 = _shape_param_6 = None
        reshape_default_5: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_7);  expand_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_6: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_29, _shape_param_8);  mul_29 = _shape_param_8 = None
        permute_default_3: "f16[512, 512]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_7: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_11, _shape_param_9);  bmm_11 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_4: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 2, 1, 3]);  reshape_default_7 = None
        clone_default: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default_8: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_10);  clone_default = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_9: "f16[2048, 512]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_11);  reshape_default_8 = _shape_param_11 = None
        permute_default_5: "f16[512, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        return (reshape_default_2, reshape_default_5, reshape_default_6, permute_default_3, reshape_default_9, permute_default_5)



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
