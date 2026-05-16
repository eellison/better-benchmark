"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_inference
Pattern hash: 130be803ccb2
Shape hash: 573b6ab7
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_default_4: "f32[4096, 1024]", _shape_param_0, _shape_param_1, arg351_1: "f32[16, 64]", _shape_param_2, mm_default_3: "f32[4096, 1024]", _shape_param_3, _shape_param_4, _shape_param_5, arg352_1: "f32[16, 64]", _shape_param_6, mm_default_1: "f32[8192, 1024]", _shape_param_7, _shape_param_8, _shape_param_9, add_252: "f32[512, 8, 1024]", _shape_param_10, arg349_1: "f32[1024, 16, 64]", _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[1, 4096, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None
        reshape_default: "f32[512, 8, 1, 16, 64]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "f32[512, 8, 16, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 4, 2]);  reshape_default = None
        reshape_default_1: "f32[512, 8, 16, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_tensor: "f32[512, 8, 16, 64]" = torch.ops.aten.add.Tensor(reshape_default_1, arg351_1);  arg351_1 = None
        unsqueeze_default_1: "f32[512, 8, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 4);  add_tensor = None
        permute_default_1: "f32[8, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [1, 2, 0, 4, 3]);  unsqueeze_default_1 = None
        permute_default_2: "f32[8, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 2, 4, 3]);  permute_default_1 = None
        reshape_default_2: "f32[128, 512, 64]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_2: "f32[1, 4096, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_3, 0);  mm_default_3 = None
        reshape_default_3: "f32[512, 8, 1, 16, 64]" = torch.ops.aten.reshape.default(unsqueeze_default_2, _shape_param_3);  unsqueeze_default_2 = _shape_param_3 = None
        permute_default_3: "f32[512, 8, 16, 64, 1]" = torch.ops.aten.permute.default(reshape_default_3, [0, 1, 3, 4, 2]);  reshape_default_3 = None
        reshape_default_4: "f32[512, 8, 16, 64]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_4);  permute_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_default_3: "f32[512, 8, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_4, 4);  reshape_default_4 = None
        permute_default_4: "f32[8, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default_3, [1, 2, 4, 0, 3]);  unsqueeze_default_3 = None
        permute_default_5: "f32[8, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_default_4, [0, 1, 4, 3, 2]);  permute_default_4 = None
        reshape_default_5: "f32[128, 64, 512]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_tensor_1: "f32[512, 8, 16, 64]" = torch.ops.aten.add.Tensor(reshape_default_1, arg352_1);  reshape_default_1 = arg352_1 = None
        unsqueeze_default_4: "f32[512, 8, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 4);  add_tensor_1 = None
        permute_default_6: "f32[8, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_4, [1, 2, 0, 4, 3]);  unsqueeze_default_4 = None
        permute_default_7: "f32[8, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default_6, [0, 1, 2, 4, 3]);  permute_default_6 = None
        reshape_default_6: "f32[128, 512, 64]" = torch.ops.aten.reshape.default(permute_default_7, _shape_param_6);  permute_default_7 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_5: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        reshape_default_7: "f32[1024, 8, 1, 16, 64]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_7);  unsqueeze_default_5 = _shape_param_7 = None
        permute_default_8: "f32[1024, 8, 16, 64, 1]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 3, 4, 2]);  reshape_default_7 = None
        reshape_default_8: "f32[1024, 8, 16, 64]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_8);  permute_default_8 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_default_6: "f32[1024, 8, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_8, 4);  reshape_default_8 = None
        permute_default_9: "f32[8, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_default_6, [1, 2, 4, 0, 3]);  unsqueeze_default_6 = None
        permute_default_10: "f32[8, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_default_9, [0, 1, 4, 3, 2]);  permute_default_9 = None
        reshape_default_9: "f32[128, 64, 1024]" = torch.ops.aten.reshape.default(permute_default_10, _shape_param_9);  permute_default_10 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_7: "f32[512, 8, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_252, 3);  add_252 = None
        unsqueeze_default_8: "f32[512, 8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 4);  unsqueeze_default_7 = None
        reshape_default_10: "f32[1, 4096, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_8, _shape_param_10);  unsqueeze_default_8 = _shape_param_10 = None
        squeeze_dim: "f32[4096, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_10, 0);  reshape_default_10 = None
        unsqueeze_default_9: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, 3);  arg349_1 = None
        unsqueeze_default_10: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 4);  unsqueeze_default_9 = None
        reshape_default_11: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_10, _shape_param_11);  unsqueeze_default_10 = _shape_param_11 = None
        squeeze_dim_1: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_11, 0);  reshape_default_11 = None
        return (reshape_default_2, reshape_default_5, reshape_default_6, reshape_default_9, squeeze_dim, squeeze_dim_1)


def _default_make_inputs():
    return [
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [512, 8, 1, 16, 64],  # _shape_param_0
    [512, 8, 16, 64],  # _shape_param_1
    torch.randn([16, 64], dtype=torch.float32, device='cuda'),
    [128, 512, 64],  # _shape_param_2
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [512, 8, 1, 16, 64],  # _shape_param_3
    [512, 8, 16, 64],  # _shape_param_4
    [128, 64, 512],  # _shape_param_5
    torch.randn([16, 64], dtype=torch.float32, device='cuda'),
    [128, 512, 64],  # _shape_param_6
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    [1024, 8, 1, 16, 64],  # _shape_param_7
    [1024, 8, 16, 64],  # _shape_param_8
    [128, 64, 1024],  # _shape_param_9
    torch.randn([512, 8, 1024], dtype=torch.float32, device='cuda'),
    [1, 4096, 1024],  # _shape_param_10
    torch.randn([1024, 16, 64], dtype=torch.float32, device='cuda'),
    [1, 1024, 1024],  # _shape_param_11
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
