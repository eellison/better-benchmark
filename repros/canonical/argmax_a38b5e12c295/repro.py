"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_inference
Pattern hash: a38b5e12c295
Shape hash: 0da1f411
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[8192, 2]", _shape_param_0, arg0_1: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        reshape_default: "f32[8, 1024, 2]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        iota_default: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:934 in forward, code: token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        iota_default_1: "i32[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:933 in forward, code: non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
        ne_scalar: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg0_1, 0);  arg0_1 = None
        convert_element_type_default: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:935 in forward, code: last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        mul_tensor: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(iota_default_1, convert_element_type_default);  iota_default_1 = convert_element_type_default = None
        argmax_default: "i64[8]" = torch.ops.aten.argmax.default(mul_tensor, -1);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        index_tensor: "f32[8, 2]" = torch.ops.aten.index.Tensor(reshape_default, [iota_default, argmax_default]);  reshape_default = iota_default = argmax_default = None
        return index_tensor


def _default_make_inputs():
    return [
    torch.randn([8192, 2], dtype=torch.float32, device='cuda'),
    [8, 1024, 2],  # _shape_param_0
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
