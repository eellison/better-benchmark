"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=var_mean, ranges=[], reduction_ranges=[]
#   origins: ['aten.var_mean.correction']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[256008, 1024]", arg0_1: "i64[32, 128]", arg2_1: "f32[2050, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:49 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[32, 128, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:517 in forward, code: position_ids = torch.arange(
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:523 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:94 in forward, code: position_ids += self.offset
        add_tensor: "i64[1, 128]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        squeeze_dim: "i64[128]" = torch.ops.aten.squeeze.dim(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:101 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        unsqueeze_default_1: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dim, 0);  squeeze_dim = None
        reshape_default: "i64[128]" = torch.ops.aten.reshape.default(unsqueeze_default_1, [-1]);  unsqueeze_default_1 = None
        index_tensor: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg2_1, [reshape_default]);  arg2_1 = reshape_default = None
        reshape_default_1: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, [1, 128, 1024]);  index_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:532 in forward, code: hidden_states = inputs_embeds + self.embed_positions(position_ids, past_key_values_length).to(
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, reshape_default_1);  mul_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:330 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([256008, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128], dtype=torch.int64, device='cuda'),
    torch.randn([2050, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
