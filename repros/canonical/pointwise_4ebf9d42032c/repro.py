"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_training
Pattern hash: 4ebf9d42032c
Shape hash: 0998765f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_273: "f32[4096, 30522]", mm_142: "f32[4096, 256]", _shape_param_0, mul_556: "f32[8, 512, 256]", mm_144: "f32[4096, 256]", _shape_param_1, mm_146: "f32[4096, 256]", _shape_param_2, _shape_param_3, primals_10: "f32[256, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        permute_default: "f32[30522, 4096]" = torch.ops.aten.permute.default(view_273, [1, 0]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 512, 256]" = torch.ops.aten.reshape.default(mm_142, _shape_param_0);  mm_142 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 256]" = torch.ops.aten.add.Tensor(mul_556, reshape_default);  mul_556 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 512, 256]" = torch.ops.aten.reshape.default(mm_144, _shape_param_1);  mm_144 = _shape_param_1 = None
        add_tensor_1: "f32[8, 512, 256]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 256]" = torch.ops.aten.reshape.default(mm_146, _shape_param_2);  mm_146 = _shape_param_2 = None
        add_tensor_2: "f32[8, 512, 256]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        reshape_default_3: "f32[4096, 256]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default_1: "f32[128, 256]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_2: "f32[256, 128]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (permute_default, reshape_default_3, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([4096, 30522], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 256], dtype=torch.float32, device='cuda'),
    [8, 512, 256],  # _shape_param_0
    torch.randn([8, 512, 256], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 256], dtype=torch.float32, device='cuda'),
    [8, 512, 256],  # _shape_param_1
    torch.randn([4096, 256], dtype=torch.float32, device='cuda'),
    [8, 512, 256],  # _shape_param_2
    [4096, 256],  # _shape_param_3
    torch.randn([256, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
