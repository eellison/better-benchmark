"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 81beb5066a80
Shape hash: f66ddf08
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
    def forward(self, bmm_540: "f32[256, 512, 512]", gt_2: "b8[16, 16, 512, 512]", div_1: "f32[16, 16, 512, 512]", full_default_5: "f32[16, 16, 512, 1023]", iota_2: "i64[512]", full_default_6: "f32[16, 16, 1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        reshape_default: "f32[16, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(bmm_540, _shape_param_0);  bmm_540 = _shape_param_0 = None
        squeeze_dim: "f32[16, 16, 512, 512]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        convert_element_type_default: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_tensor: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_dim, mul_tensor);  squeeze_dim = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        mul_tensor_2: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True)
        neg_default: "f32[16, 16, 512, 512]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_default: "f32[16, 16, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        mul_tensor_3: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(fma_default, 0.125);  fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_put_default: "f32[16, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota_2], mul_tensor_3, True);  full_default_5 = iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        reshape_default_1: "f32[16, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_default, _shape_param_1);  index_put_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_scatter_default: "f32[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_6, reshape_default_1, 2, 1, 9223372036854775807);  full_default_6 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        reshape_default_2: "f32[16, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_default, _shape_param_2);  slice_scatter_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_3: "f32[16, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default: "f32[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(reshape_default_3, [0, 1, 2, 4, 3]);  reshape_default_3 = None
        reshape_default_4: "f32[256, 512, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_4);  permute_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default_5: "f32[16, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_5);  mul_tensor_3 = _shape_param_5 = None
        permute_default_1: "f32[16, 16, 512, 1, 512]" = torch.ops.aten.permute.default(reshape_default_5, [0, 1, 2, 4, 3]);  reshape_default_5 = None
        reshape_default_6: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_6);  permute_default_1 = _shape_param_6 = None
        return (reshape_default_4, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [16, 16, 512, 512], dtype=torch.bool, device='cuda'),
    torch.randn([16, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16, 512, 1023], dtype=torch.float32, device='cuda'),
    torch.randint(0, 16, [512], dtype=torch.int64, device='cuda'),
    torch.randn([16, 16, 1024, 512], dtype=torch.float32, device='cuda'),
    [16, 16, 512, 512, 1],  # _shape_param_0
    [16, 16, 1023, 512],  # _shape_param_1
    [16, 16, 512, 1024],  # _shape_param_2
    [16, 16, 512, 1024, 1],  # _shape_param_3
    [256, 512, 1024],  # _shape_param_4
    [16, 16, 512, 512, 1],  # _shape_param_5
    [256, 512, 512],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
