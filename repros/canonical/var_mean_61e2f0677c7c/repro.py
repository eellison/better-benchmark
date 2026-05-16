"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_inference
Pattern hash: 61e2f0677c7c
Shape hash: 2a2fb38c
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[256008, 1024]", arg0_1: "i64[8, 128]", arg2_1: "f32[2050, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:50 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[8, 128, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:441 in forward, code: position_ids = torch.arange(
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:447 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:96 in forward, code: position_ids = position_ids + self.offset
        add_tensor: "i64[1, 128]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:102 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        reshape_default: "i64[128]" = torch.ops.aten.reshape.default(add_tensor, [-1]);  add_tensor = None
        index_tensor: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg2_1, [reshape_default]);  arg2_1 = reshape_default = None
        reshape_default_1: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:458 in forward, code: hidden_states = inputs_embeds + self.embed_positions(position_ids, past_key_values_length).to(
        add_tensor_1: "f32[8, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, reshape_default_1);  mul_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[8, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([256008, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 128], dtype=torch.int64, device='cuda'),
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
