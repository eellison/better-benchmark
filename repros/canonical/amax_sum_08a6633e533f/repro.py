"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_train
Pattern hash: 08a6633e533f
Shape hash: dc3bd322
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 1024], f32), T([512, 128, 128], f32), T([32, 1, 128, 128], f32), S([32, 128, 1024]), S([32, 128, -1, 64]), S([512, 128, 64]), S([32, 16, 128, 128]), S([512, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[4096, 1024]", bmm: "f32[512, 128, 128]", primals_10: "f32[32, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        reshape_default: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        reshape_default_1: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_default: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        reshape_default_3: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None
        add_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_3, primals_10);  reshape_default_3 = primals_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_default: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_tensor, full_default);  add_tensor = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        reshape_default_4: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(maximum_default, _shape_param_4);  maximum_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_default: "f32[512, 128, 1]" = torch.ops.aten.amax.default(reshape_default_4, [-1], True)
        sub_tensor: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default_4, amax_default);  reshape_default_4 = amax_default = None
        exp_default: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:226 in forward, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 128, 128]" = torch.ops.prims.inductor_random.default([512, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor);  gt_scalar = div_tensor = None
        mul_tensor_1: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        return (reshape_default_2, mul_tensor_1)


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
