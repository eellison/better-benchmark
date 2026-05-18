"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_default_4: "f32[8192, 1024]", mul_1132: "f32[512, 16, 1024]", mm_default_2: "f32[8192, 1024]", mm_default: "f32[8192, 1024]", gt: "b8[512, 16, 1024]", clone: "i64[512, 16]", full_default_1: "f32[]", mm_1: "f32[32000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:418 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None
        reshape_default: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        squeeze_dim: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None
        squeeze_dim_1: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim, 3);  squeeze_dim = None
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_1132, squeeze_dim_1);  mul_1132 = squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:417 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_1: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_2, 0);  mm_default_2 = None
        reshape_default_1: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_1);  unsqueeze_default_1 = _shape_param_1 = None
        squeeze_dim_2: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default_1, 4);  reshape_default_1 = None
        squeeze_dim_3: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim_2, 3);  squeeze_dim_2 = None
        add_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_tensor, squeeze_dim_3);  add_tensor = squeeze_dim_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:416 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_2: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        reshape_default_2: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_2, _shape_param_2);  unsqueeze_default_2 = _shape_param_2 = None
        squeeze_dim_4: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default_2, 4);  reshape_default_2 = None
        squeeze_dim_5: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim_4, 3);  squeeze_dim_4 = None
        add_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, squeeze_dim_5);  add_tensor_1 = squeeze_dim_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1308 in forward, code: output_h = self.dropout(word_emb_k)
        convert_element_type_default: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1307 in forward, code: word_emb_k = self.word_embedding(input_ids)
        eq_scalar: "b8[512, 16]" = torch.ops.aten.eq.Scalar(clone, -1)
        unsqueeze_default_3: "b8[512, 16, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[512, 16, 1024]" = torch.ops.aten.where.self(unsqueeze_default_3, full_default_1, mul_tensor_1);  unsqueeze_default_3 = full_default_1 = mul_tensor_1 = None
        full_default: "f32[32000, 1024]" = torch.ops.aten.full.default([32000, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32000, 1024]" = torch.ops.aten.index_put.default(full_default, [clone], where_self, True);  full_default = clone = where_self = None
        add_tensor_3: "f32[32000, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_default);  mm_1 = index_put_default = None
        return add_tensor_3


def _default_make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [512, 16, 1024], dtype=torch.bool, device='cuda'),
    torch.randint(0, 32000, [512, 16], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([32000, 1024], dtype=torch.float32, device='cuda'),
    [512, 16, 1024, 1, 1],  # _shape_param_0
    [512, 16, 1024, 1, 1],  # _shape_param_1
    [512, 16, 1024, 1, 1],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
