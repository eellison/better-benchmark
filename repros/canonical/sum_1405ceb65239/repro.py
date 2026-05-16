"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_training
Pattern hash: 1405ceb65239
Shape hash: 8b022c72
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, gt: "b8[128, 128, 128]", bmm_3: "f32[128, 128, 128]", bmm: "f32[128, 128, 128]", _shape_param_0, primals_10: "f32[8, 1, 128, 128]", _shape_param_1, amax: "f32[128, 128, 1]", sum_1: "f32[128, 128, 1]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:226 in forward, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[128, 128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[128, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_3, mul_tensor);  bmm_3 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        reshape_default: "f32[8, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add_tensor: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, primals_10);  reshape_default = primals_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_default: "f32[8, 16, 128, 128]" = torch.ops.aten.maximum.default(add_tensor, full_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        reshape_default_1: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(maximum_default, _shape_param_1);  maximum_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        sub_tensor: "f32[128, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax);  reshape_default_1 = amax = None
        exp_default: "f32[128, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[128, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        mul_tensor_2: "f32[128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor);  mul_tensor_1 = None
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[128, 128, 128]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[128, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        reshape_default_2: "f32[8, 16, 128, 128]" = torch.ops.aten.reshape.default(fma_default, _shape_param_2);  fma_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        div_scalar: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Scalar(reshape_default_2, 2)
        eq_tensor: "b8[8, 16, 128, 128]" = torch.ops.aten.eq.Tensor(add_tensor, full_default)
        where_self: "f32[8, 16, 128, 128]" = torch.ops.aten.where.self(eq_tensor, div_scalar, reshape_default_2);  eq_tensor = div_scalar = reshape_default_2 = None
        lt_tensor: "b8[8, 16, 128, 128]" = torch.ops.aten.lt.Tensor(add_tensor, full_default);  add_tensor = full_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 16, 128, 128]" = torch.ops.aten.where.self(lt_tensor, full_default_1, where_self);  lt_tensor = full_default_1 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        reshape_default_3: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(where_self_1, _shape_param_3);  where_self_1 = _shape_param_3 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randint(0, 2, [128, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randn([128, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 128],  # _shape_param_0
    torch.randn([8, 1, 128, 128], dtype=torch.float32, device='cuda'),
    [128, 128, 128],  # _shape_param_1
    torch.randn([128, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 1], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 128],  # _shape_param_2
    [128, 128, 128],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
