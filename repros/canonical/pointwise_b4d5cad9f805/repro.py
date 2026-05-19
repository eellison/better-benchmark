"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_train
Pattern hash: b4d5cad9f805
Shape hash: 7e4611ca
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
    def forward(self, arg0_1: "i64[1, 128]", arg1_1: "f32[2050, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:96 in forward, code: position_ids = position_ids + self.offset
        add_tensor: "i64[1, 128]" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:102 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        reshape_default: "i64[128]" = torch.ops.aten.reshape.default(add_tensor, [-1]);  add_tensor = None
        index_tensor: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg1_1, [reshape_default]);  arg1_1 = reshape_default = None
        reshape_default_1: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randint(0, 128, [1, 128], dtype=torch.int64, device='cuda'),
    torch.randn([2050, 1024], dtype=torch.float32, device='cuda'),
    [1, 128, 1024],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
