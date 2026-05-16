"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: ae103933839d
Shape hash: 52727beb
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant0: "i64[]", addmm_1: "f32[32768, 256]", _shape_param_0, arg2_1: "f32[8, 4096, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        lift_fresh_copy_default: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        convert_element_type_default: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_default, torch.float64);  lift_fresh_copy_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 4096, 256]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_0);  addmm_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_tensor: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(arg2_1, reshape_default);  arg2_1 = reshape_default = None
        return (convert_element_type_default, add_tensor)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    [8, 4096, 256],  # _shape_param_0
    torch.randn(16776960, dtype=torch.float32, device='cuda').as_strided([8, 4096, 256], [2097152, 512, 1]),  # arg2_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
