"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['16', '16', '512', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['16', '16', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_188: "f32[256, 512, 512]", bmm_189: "f32[256, 512, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:263 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default: "f32[16, 16, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_188, [16, 16, 512, 1, 512]);  bmm_188 = None
        permute_default: "f32[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 2, 4, 3]);  reshape_default = None
        reshape_default_1: "f32[16, 16, 512, 512]" = torch.ops.aten.reshape.default(permute_default, [16, 16, 512, 512]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:266 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_2: "f32[16, 16, 512, 1, 1024]" = torch.ops.aten.reshape.default(bmm_189, [16, 16, 512, 1, 1024]);  bmm_189 = None
        permute_default_1: "f32[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 2, 4, 3]);  reshape_default_2 = None
        reshape_default_3: "f32[16, 16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_1, [16, 16, 512, 1024]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:238 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        reshape_default_4: "f32[16, 16, 1024, 512]" = torch.ops.aten.reshape.default(reshape_default_3, [16, 16, 1024, 512]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:239 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_tensor: "f32[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(reshape_default_4, 2, 1, 9223372036854775807);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:240 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        reshape_default_5: "f32[16, 16, 512, 1023]" = torch.ops.aten.reshape.default(slice_tensor, [16, 16, 512, 1023]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:244 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_tensor: "f32[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(reshape_default_5, [None, None, None, iota_default]);  reshape_default_5 = iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:277 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_tensor: "f32[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(reshape_default_1, index_tensor);  reshape_default_1 = index_tensor = None
        add_tensor_1: "f32[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor, 0);  add_tensor = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, 1);  add_tensor_1 = None
        amax_default: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_tensor, [3], True)
        sub_tensor: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:286 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        mul_tensor_1: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [3], True)
        div_tensor: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:294 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_default: "f32[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(div_tensor, 4);  div_tensor = None
        reshape_default_6: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(unsqueeze_default, [256, 512, 512]);  unsqueeze_default = None
        return reshape_default_6


def _default_make_inputs():
    return [
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 512, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
