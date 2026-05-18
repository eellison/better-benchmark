"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_training
Pattern hash: 3a3bd1800da3
Shape hash: 9a231908
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_default_333: "f32[4096, 1024]", _shape_param_0, _shape_param_1, bmm_4: "f32[128, 512, 512]", _shape_param_2, _shape_param_3, bmm_5: "f32[128, 512, 1024]", _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, inductor_seeds_default: "i64[99]", _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default: "f32[1, 4096, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_333, 0);  mm_default_333 = None
        reshape_default: "f32[512, 8, 1, 16, 64]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "f32[512, 8, 16, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 4, 2]);  reshape_default = None
        reshape_default_1: "f32[512, 8, 16, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default_2: "f32[8, 16, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_4, _shape_param_2);  bmm_4 = _shape_param_2 = None
        permute_default_1: "f32[8, 16, 512, 512, 1]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 2, 4, 3]);  reshape_default_2 = None
        reshape_default_3: "f32[8, 16, 512, 512]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_4: "f32[8, 16, 512, 1, 1024]" = torch.ops.aten.reshape.default(bmm_5, _shape_param_4);  bmm_5 = _shape_param_4 = None
        permute_default_2: "f32[8, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(reshape_default_4, [0, 1, 2, 4, 3]);  reshape_default_4 = None
        reshape_default_5: "f32[8, 16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_5);  permute_default_2 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        reshape_default_6: "f32[8, 16, 1024, 512]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_6);  reshape_default_5 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_tensor: "f32[8, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(reshape_default_6, 2, 1, 9223372036854775807);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        reshape_default_7: "f32[8, 16, 512, 1023]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_7);  slice_tensor = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.index.Tensor(reshape_default_7, [None, None, None, iota_default]);  reshape_default_7 = iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.add.Tensor(reshape_default_3, index_tensor);  reshape_default_3 = index_tensor = None
        add_tensor_1: "f32[8, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor, 0);  add_tensor = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, 1);  add_tensor_1 = None
        amax_default: "f32[8, 16, 512, 1]" = torch.ops.aten.amax.default(mul_tensor, [3], True)
        sub_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        mul_tensor_1: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[8, 16, 512, 512]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [3], True)
        div_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 16, 512, 512]" = torch.ops.prims.inductor_random.default([8, 16, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor);  gt_scalar = div_tensor = None
        mul_tensor_3: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_default_1: "f32[8, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 4);  mul_tensor_3 = None
        unsqueeze_default_2: "f32[512, 8, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, 4);  reshape_default_1 = None
        permute_default_3: "f32[1, 8, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_default_2, [4, 1, 2, 3, 0]);  unsqueeze_default_2 = None
        reshape_default_8: "f32[128, 512, 512]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_8);  unsqueeze_default_1 = _shape_param_8 = None
        permute_default_4: "f32[8, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default_3, [1, 2, 4, 3, 0]);  permute_default_3 = None
        reshape_default_9: "f32[128, 512, 64]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_9);  permute_default_4 = _shape_param_9 = None
        return (reshape_default_8, reshape_default_9)


def _default_make_inputs():
    return [
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [512, 8, 1, 16, 64],  # _shape_param_0
    [512, 8, 16, 64],  # _shape_param_1
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 16, 512, 1, 512],  # _shape_param_2
    [8, 16, 512, 512],  # _shape_param_3
    torch.randn([128, 512, 1024], dtype=torch.float32, device='cuda'),
    [8, 16, 512, 1, 1024],  # _shape_param_4
    [8, 16, 512, 1024],  # _shape_param_5
    [8, 16, 1024, 512],  # _shape_param_6
    [8, 16, 512, 1023],  # _shape_param_7
    torch.randint(0, 2, [99], dtype=torch.int64, device='cuda'),
    [128, 512, 512],  # _shape_param_8
    [128, 512, 64],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
