"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_infer
Pattern hash: f1c229bd204c
Shape hash: 457b5eaa
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([256008, 1024], f32), T([32, 128], i64, max=256008), T([2050, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([1, 128, 1024]), S([4096, 1024]), S([4096, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[256008, 1024]", arg0_1: "i64[32, 128]", arg2_1: "f32[2050, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024, 1024]", arg7_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:50 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[32, 128, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None

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
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, reshape_default_1);  mul_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  mul_tensor_1 = arg3_1 = None
        add_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg4_1);  mul_tensor_2 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default_2: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_3: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        return (reshape_default_2, permute_default, reshape_default_3, permute_default_1)


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
