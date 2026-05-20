"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_infer
Pattern hash: 8e8c3e1f9bf7
Shape hash: 24271ab7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([50265, 1024], f32), T([64, 256], i64, gen=Index(50265)), T([514, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([64, -1]), S([16384, 1024]), S([16384, 1024]), S([16384, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50265, 1024]", arg0_1: "i64[64, 256]", arg2_1: "f32[514, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024, 1024]", arg7_1: "f32[1024, 1024]", arg9_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:75 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[64, 256, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:55 in forward, code: position_ids = torch.arange(
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:57 in forward, code: ).expand(bsz, -1)
        expand_default: "i64[64, 256]" = torch.ops.aten.expand.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:61 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[64, 256]" = torch.ops.aten.add.Tensor(expand_default, 2);  expand_default = None
        embedding_default_1: "f32[64, 256, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:543 in forward, code: hidden_states = inputs_embeds + embed_pos
        add_tensor_1: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:546 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 256, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[64, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  mul_tensor_1 = arg3_1 = None
        add_tensor_3: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg4_1);  mul_tensor_2 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default: "f32[16384, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[16384, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f32[16384, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_3);  add_tensor_3 = _shape_param_3 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)


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
