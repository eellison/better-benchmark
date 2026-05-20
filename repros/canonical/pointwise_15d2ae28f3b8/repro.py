"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 15d2ae28f3b8
Shape hash: fe8271c7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 32000], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 64, 1024], f32), T([256, 512, 64], f32), T([256, 64, 1024], f32), T([256, 512, 64], f32), T([256, 64, 512], f32), T([256, 512, 64], f32), T([1024, 16, 64], f32), T([1024, 16, 64], f32), T([1024, 16, 64], f32), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 64, 1024, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([16, 16, 512, 64, 1]), S([16, 16, 64, 1024, 1]), S([16, 16, 512, 64, 1]), S([16, 16, 64, 512, 1]), S([16, 16, 512, 64, 1]), S([1024, 16, 16, 64, 1]), S([1, 16384, 1024]), S([512, 16, 16, 64, 1]), S([1, 8192, 1024]), S([1, 1024, 1024]), S([512, 16, 16, 64, 1]), S([1, 8192, 1024]), S([1, 1024, 1024]), S([512, 16, 16, 64, 1]), S([1, 8192, 1024]), S([1, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, view_917: "f32[8192, 32000]", bmm_196: "f32[256, 64, 1024]", bmm_211: "f32[256, 64, 1024]", bmm_226: "f32[256, 64, 1024]", bmm_241: "f32[256, 64, 1024]", bmm_256: "f32[256, 64, 1024]", bmm_271: "f32[256, 64, 1024]", bmm_286: "f32[256, 64, 1024]", bmm_301: "f32[256, 64, 1024]", bmm_316: "f32[256, 64, 1024]", bmm_331: "f32[256, 64, 1024]", bmm_346: "f32[256, 64, 1024]", bmm_361: "f32[256, 64, 1024]", bmm_376: "f32[256, 64, 1024]", bmm_391: "f32[256, 64, 1024]", bmm_406: "f32[256, 64, 1024]", bmm_421: "f32[256, 64, 1024]", bmm_436: "f32[256, 64, 1024]", bmm_451: "f32[256, 64, 1024]", bmm_466: "f32[256, 64, 1024]", bmm_481: "f32[256, 64, 1024]", bmm_496: "f32[256, 64, 1024]", bmm_511: "f32[256, 64, 1024]", bmm_526: "f32[256, 64, 1024]", bmm_539: "f32[256, 512, 64]", bmm_541: "f32[256, 64, 1024]", bmm_542: "f32[256, 512, 64]", bmm_543: "f32[256, 64, 512]", bmm_544: "f32[256, 512, 64]", primals_5: "f32[1024, 16, 64]", primals_4: "f32[1024, 16, 64]", primals_3: "f32[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        permute_default: "f32[32000, 8192]" = torch.ops.aten.permute.default(view_917, [1, 0]);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_196, _shape_param_0);  bmm_196 = _shape_param_0 = None
        permute_default_1: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 4, 3, 2]);  reshape_default = None
        permute_default_2: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_1, [3, 0, 1, 4, 2]);  permute_default_1 = None
        squeeze_dim: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_2, 4);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_1: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim, _shape_param_1);  squeeze_dim = _shape_param_1 = None
        permute_default_3: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 4, 2, 3]);  reshape_default_1 = None
        clone_default: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_2: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        squeeze_dim_1: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_3: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_211, _shape_param_3);  bmm_211 = _shape_param_3 = None
        permute_default_4: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 1, 4, 3, 2]);  reshape_default_3 = None
        permute_default_5: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_4, [3, 0, 1, 4, 2]);  permute_default_4 = None
        squeeze_dim_2: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_5, 4);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_4: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_2, _shape_param_4);  squeeze_dim_2 = _shape_param_4 = None
        permute_default_6: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 1, 4, 2, 3]);  reshape_default_4 = None
        clone_default_1: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        reshape_default_5: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        squeeze_dim_3: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_5, 0);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_6: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_226, _shape_param_6);  bmm_226 = _shape_param_6 = None
        permute_default_7: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 1, 4, 3, 2]);  reshape_default_6 = None
        permute_default_8: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_7, [3, 0, 1, 4, 2]);  permute_default_7 = None
        squeeze_dim_4: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_8, 4);  permute_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_7: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_4, _shape_param_7);  squeeze_dim_4 = _shape_param_7 = None
        permute_default_9: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 4, 2, 3]);  reshape_default_7 = None
        clone_default_2: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        reshape_default_8: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        squeeze_dim_5: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_8, 0);  reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_9: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_241, _shape_param_9);  bmm_241 = _shape_param_9 = None
        permute_default_10: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_9, [0, 1, 4, 3, 2]);  reshape_default_9 = None
        permute_default_11: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_10, [3, 0, 1, 4, 2]);  permute_default_10 = None
        squeeze_dim_6: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_11, 4);  permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_10: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_6, _shape_param_10);  squeeze_dim_6 = _shape_param_10 = None
        permute_default_12: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_10, [0, 1, 4, 2, 3]);  reshape_default_10 = None
        clone_default_3: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_12, memory_format = torch.contiguous_format);  permute_default_12 = None
        reshape_default_11: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_11);  clone_default_3 = _shape_param_11 = None
        squeeze_dim_7: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_11, 0);  reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_12: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_256, _shape_param_12);  bmm_256 = _shape_param_12 = None
        permute_default_13: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_12, [0, 1, 4, 3, 2]);  reshape_default_12 = None
        permute_default_14: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_13, [3, 0, 1, 4, 2]);  permute_default_13 = None
        squeeze_dim_8: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_14, 4);  permute_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_13: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_8, _shape_param_13);  squeeze_dim_8 = _shape_param_13 = None
        permute_default_15: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_13, [0, 1, 4, 2, 3]);  reshape_default_13 = None
        clone_default_4: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        reshape_default_14: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_14);  clone_default_4 = _shape_param_14 = None
        squeeze_dim_9: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_14, 0);  reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_15: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_271, _shape_param_15);  bmm_271 = _shape_param_15 = None
        permute_default_16: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_15, [0, 1, 4, 3, 2]);  reshape_default_15 = None
        permute_default_17: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_16, [3, 0, 1, 4, 2]);  permute_default_16 = None
        squeeze_dim_10: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_17, 4);  permute_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_16: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_10, _shape_param_16);  squeeze_dim_10 = _shape_param_16 = None
        permute_default_18: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_16, [0, 1, 4, 2, 3]);  reshape_default_16 = None
        clone_default_5: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_18, memory_format = torch.contiguous_format);  permute_default_18 = None
        reshape_default_17: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_17);  clone_default_5 = _shape_param_17 = None
        squeeze_dim_11: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_17, 0);  reshape_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_18: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_286, _shape_param_18);  bmm_286 = _shape_param_18 = None
        permute_default_19: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_18, [0, 1, 4, 3, 2]);  reshape_default_18 = None
        permute_default_20: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_19, [3, 0, 1, 4, 2]);  permute_default_19 = None
        squeeze_dim_12: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_20, 4);  permute_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_19: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_12, _shape_param_19);  squeeze_dim_12 = _shape_param_19 = None
        permute_default_21: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_19, [0, 1, 4, 2, 3]);  reshape_default_19 = None
        clone_default_6: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_21, memory_format = torch.contiguous_format);  permute_default_21 = None
        reshape_default_20: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_20);  clone_default_6 = _shape_param_20 = None
        squeeze_dim_13: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_20, 0);  reshape_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_21: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_301, _shape_param_21);  bmm_301 = _shape_param_21 = None
        permute_default_22: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_21, [0, 1, 4, 3, 2]);  reshape_default_21 = None
        permute_default_23: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_22, [3, 0, 1, 4, 2]);  permute_default_22 = None
        squeeze_dim_14: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_23, 4);  permute_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_22: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_14, _shape_param_22);  squeeze_dim_14 = _shape_param_22 = None
        permute_default_24: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_22, [0, 1, 4, 2, 3]);  reshape_default_22 = None
        clone_default_7: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_24, memory_format = torch.contiguous_format);  permute_default_24 = None
        reshape_default_23: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_7, _shape_param_23);  clone_default_7 = _shape_param_23 = None
        squeeze_dim_15: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_23, 0);  reshape_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_24: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_316, _shape_param_24);  bmm_316 = _shape_param_24 = None
        permute_default_25: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_24, [0, 1, 4, 3, 2]);  reshape_default_24 = None
        permute_default_26: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_25, [3, 0, 1, 4, 2]);  permute_default_25 = None
        squeeze_dim_16: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_26, 4);  permute_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_25: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_16, _shape_param_25);  squeeze_dim_16 = _shape_param_25 = None
        permute_default_27: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_25, [0, 1, 4, 2, 3]);  reshape_default_25 = None
        clone_default_8: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_27, memory_format = torch.contiguous_format);  permute_default_27 = None
        reshape_default_26: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_8, _shape_param_26);  clone_default_8 = _shape_param_26 = None
        squeeze_dim_17: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_26, 0);  reshape_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_27: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_331, _shape_param_27);  bmm_331 = _shape_param_27 = None
        permute_default_28: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_27, [0, 1, 4, 3, 2]);  reshape_default_27 = None
        permute_default_29: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_28, [3, 0, 1, 4, 2]);  permute_default_28 = None
        squeeze_dim_18: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_29, 4);  permute_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_28: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_18, _shape_param_28);  squeeze_dim_18 = _shape_param_28 = None
        permute_default_30: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_28, [0, 1, 4, 2, 3]);  reshape_default_28 = None
        clone_default_9: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_30, memory_format = torch.contiguous_format);  permute_default_30 = None
        reshape_default_29: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_9, _shape_param_29);  clone_default_9 = _shape_param_29 = None
        squeeze_dim_19: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_29, 0);  reshape_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_30: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_346, _shape_param_30);  bmm_346 = _shape_param_30 = None
        permute_default_31: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_30, [0, 1, 4, 3, 2]);  reshape_default_30 = None
        permute_default_32: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_31, [3, 0, 1, 4, 2]);  permute_default_31 = None
        squeeze_dim_20: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_32, 4);  permute_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_31: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_20, _shape_param_31);  squeeze_dim_20 = _shape_param_31 = None
        permute_default_33: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_31, [0, 1, 4, 2, 3]);  reshape_default_31 = None
        clone_default_10: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_33, memory_format = torch.contiguous_format);  permute_default_33 = None
        reshape_default_32: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_10, _shape_param_32);  clone_default_10 = _shape_param_32 = None
        squeeze_dim_21: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_32, 0);  reshape_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_33: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_361, _shape_param_33);  bmm_361 = _shape_param_33 = None
        permute_default_34: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_33, [0, 1, 4, 3, 2]);  reshape_default_33 = None
        permute_default_35: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_34, [3, 0, 1, 4, 2]);  permute_default_34 = None
        squeeze_dim_22: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_35, 4);  permute_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_34: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_22, _shape_param_34);  squeeze_dim_22 = _shape_param_34 = None
        permute_default_36: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_34, [0, 1, 4, 2, 3]);  reshape_default_34 = None
        clone_default_11: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_36, memory_format = torch.contiguous_format);  permute_default_36 = None
        reshape_default_35: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_11, _shape_param_35);  clone_default_11 = _shape_param_35 = None
        squeeze_dim_23: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_35, 0);  reshape_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_36: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_376, _shape_param_36);  bmm_376 = _shape_param_36 = None
        permute_default_37: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_36, [0, 1, 4, 3, 2]);  reshape_default_36 = None
        permute_default_38: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_37, [3, 0, 1, 4, 2]);  permute_default_37 = None
        squeeze_dim_24: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_38, 4);  permute_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_37: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_24, _shape_param_37);  squeeze_dim_24 = _shape_param_37 = None
        permute_default_39: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_37, [0, 1, 4, 2, 3]);  reshape_default_37 = None
        clone_default_12: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_39, memory_format = torch.contiguous_format);  permute_default_39 = None
        reshape_default_38: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_12, _shape_param_38);  clone_default_12 = _shape_param_38 = None
        squeeze_dim_25: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_38, 0);  reshape_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_39: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_391, _shape_param_39);  bmm_391 = _shape_param_39 = None
        permute_default_40: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_39, [0, 1, 4, 3, 2]);  reshape_default_39 = None
        permute_default_41: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_40, [3, 0, 1, 4, 2]);  permute_default_40 = None
        squeeze_dim_26: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_41, 4);  permute_default_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_40: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_26, _shape_param_40);  squeeze_dim_26 = _shape_param_40 = None
        permute_default_42: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_40, [0, 1, 4, 2, 3]);  reshape_default_40 = None
        clone_default_13: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_42, memory_format = torch.contiguous_format);  permute_default_42 = None
        reshape_default_41: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_13, _shape_param_41);  clone_default_13 = _shape_param_41 = None
        squeeze_dim_27: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_41, 0);  reshape_default_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_42: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_406, _shape_param_42);  bmm_406 = _shape_param_42 = None
        permute_default_43: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_42, [0, 1, 4, 3, 2]);  reshape_default_42 = None
        permute_default_44: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_43, [3, 0, 1, 4, 2]);  permute_default_43 = None
        squeeze_dim_28: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_44, 4);  permute_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_43: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_28, _shape_param_43);  squeeze_dim_28 = _shape_param_43 = None
        permute_default_45: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_43, [0, 1, 4, 2, 3]);  reshape_default_43 = None
        clone_default_14: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_45, memory_format = torch.contiguous_format);  permute_default_45 = None
        reshape_default_44: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_14, _shape_param_44);  clone_default_14 = _shape_param_44 = None
        squeeze_dim_29: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_44, 0);  reshape_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_45: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_421, _shape_param_45);  bmm_421 = _shape_param_45 = None
        permute_default_46: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_45, [0, 1, 4, 3, 2]);  reshape_default_45 = None
        permute_default_47: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_46, [3, 0, 1, 4, 2]);  permute_default_46 = None
        squeeze_dim_30: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_47, 4);  permute_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_46: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_30, _shape_param_46);  squeeze_dim_30 = _shape_param_46 = None
        permute_default_48: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_46, [0, 1, 4, 2, 3]);  reshape_default_46 = None
        clone_default_15: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_48, memory_format = torch.contiguous_format);  permute_default_48 = None
        reshape_default_47: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_15, _shape_param_47);  clone_default_15 = _shape_param_47 = None
        squeeze_dim_31: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_47, 0);  reshape_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_48: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_436, _shape_param_48);  bmm_436 = _shape_param_48 = None
        permute_default_49: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_48, [0, 1, 4, 3, 2]);  reshape_default_48 = None
        permute_default_50: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_49, [3, 0, 1, 4, 2]);  permute_default_49 = None
        squeeze_dim_32: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_50, 4);  permute_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_49: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_32, _shape_param_49);  squeeze_dim_32 = _shape_param_49 = None
        permute_default_51: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_49, [0, 1, 4, 2, 3]);  reshape_default_49 = None
        clone_default_16: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_51, memory_format = torch.contiguous_format);  permute_default_51 = None
        reshape_default_50: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_16, _shape_param_50);  clone_default_16 = _shape_param_50 = None
        squeeze_dim_33: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_50, 0);  reshape_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_51: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_451, _shape_param_51);  bmm_451 = _shape_param_51 = None
        permute_default_52: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_51, [0, 1, 4, 3, 2]);  reshape_default_51 = None
        permute_default_53: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_52, [3, 0, 1, 4, 2]);  permute_default_52 = None
        squeeze_dim_34: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_53, 4);  permute_default_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_52: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_34, _shape_param_52);  squeeze_dim_34 = _shape_param_52 = None
        permute_default_54: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_52, [0, 1, 4, 2, 3]);  reshape_default_52 = None
        clone_default_17: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_54, memory_format = torch.contiguous_format);  permute_default_54 = None
        reshape_default_53: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_17, _shape_param_53);  clone_default_17 = _shape_param_53 = None
        squeeze_dim_35: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_53, 0);  reshape_default_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_54: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_466, _shape_param_54);  bmm_466 = _shape_param_54 = None
        permute_default_55: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_54, [0, 1, 4, 3, 2]);  reshape_default_54 = None
        permute_default_56: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_55, [3, 0, 1, 4, 2]);  permute_default_55 = None
        squeeze_dim_36: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_56, 4);  permute_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_55: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_36, _shape_param_55);  squeeze_dim_36 = _shape_param_55 = None
        permute_default_57: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_55, [0, 1, 4, 2, 3]);  reshape_default_55 = None
        clone_default_18: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_57, memory_format = torch.contiguous_format);  permute_default_57 = None
        reshape_default_56: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_18, _shape_param_56);  clone_default_18 = _shape_param_56 = None
        squeeze_dim_37: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_56, 0);  reshape_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_57: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_481, _shape_param_57);  bmm_481 = _shape_param_57 = None
        permute_default_58: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_57, [0, 1, 4, 3, 2]);  reshape_default_57 = None
        permute_default_59: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_58, [3, 0, 1, 4, 2]);  permute_default_58 = None
        squeeze_dim_38: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_59, 4);  permute_default_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_58: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_38, _shape_param_58);  squeeze_dim_38 = _shape_param_58 = None
        permute_default_60: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_58, [0, 1, 4, 2, 3]);  reshape_default_58 = None
        clone_default_19: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_60, memory_format = torch.contiguous_format);  permute_default_60 = None
        reshape_default_59: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_19, _shape_param_59);  clone_default_19 = _shape_param_59 = None
        squeeze_dim_39: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_59, 0);  reshape_default_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_60: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_496, _shape_param_60);  bmm_496 = _shape_param_60 = None
        permute_default_61: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_60, [0, 1, 4, 3, 2]);  reshape_default_60 = None
        permute_default_62: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_61, [3, 0, 1, 4, 2]);  permute_default_61 = None
        squeeze_dim_40: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_62, 4);  permute_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_61: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_40, _shape_param_61);  squeeze_dim_40 = _shape_param_61 = None
        permute_default_63: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_61, [0, 1, 4, 2, 3]);  reshape_default_61 = None
        clone_default_20: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_63, memory_format = torch.contiguous_format);  permute_default_63 = None
        reshape_default_62: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_20, _shape_param_62);  clone_default_20 = _shape_param_62 = None
        squeeze_dim_41: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_62, 0);  reshape_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_63: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_511, _shape_param_63);  bmm_511 = _shape_param_63 = None
        permute_default_64: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_63, [0, 1, 4, 3, 2]);  reshape_default_63 = None
        permute_default_65: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_64, [3, 0, 1, 4, 2]);  permute_default_64 = None
        squeeze_dim_42: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_65, 4);  permute_default_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_64: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_42, _shape_param_64);  squeeze_dim_42 = _shape_param_64 = None
        permute_default_66: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_64, [0, 1, 4, 2, 3]);  reshape_default_64 = None
        clone_default_21: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_66, memory_format = torch.contiguous_format);  permute_default_66 = None
        reshape_default_65: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_21, _shape_param_65);  clone_default_21 = _shape_param_65 = None
        squeeze_dim_43: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_65, 0);  reshape_default_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_66: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_526, _shape_param_66);  bmm_526 = _shape_param_66 = None
        permute_default_67: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_66, [0, 1, 4, 3, 2]);  reshape_default_66 = None
        permute_default_68: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_67, [3, 0, 1, 4, 2]);  permute_default_67 = None
        squeeze_dim_44: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_68, 4);  permute_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_67: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_44, _shape_param_67);  squeeze_dim_44 = _shape_param_67 = None
        permute_default_69: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_67, [0, 1, 4, 2, 3]);  reshape_default_67 = None
        clone_default_22: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_69, memory_format = torch.contiguous_format);  permute_default_69 = None
        reshape_default_68: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_22, _shape_param_68);  clone_default_22 = _shape_param_68 = None
        squeeze_dim_45: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_68, 0);  reshape_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        reshape_default_69: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_539, _shape_param_69);  bmm_539 = _shape_param_69 = None
        permute_default_70: "f32[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(reshape_default_69, [4, 0, 1, 3, 2]);  reshape_default_69 = None
        permute_default_71: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_70, [4, 1, 2, 3, 0]);  permute_default_70 = None
        squeeze_dim_46: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_71, 4);  permute_default_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_70: "f32[16, 16, 64, 1024, 1]" = torch.ops.aten.reshape.default(bmm_541, _shape_param_70);  bmm_541 = _shape_param_70 = None
        permute_default_72: "f32[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_70, [0, 1, 4, 3, 2]);  reshape_default_70 = None
        reshape_default_71: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_542, _shape_param_71);  bmm_542 = _shape_param_71 = None
        permute_default_73: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(reshape_default_71, [0, 1, 2, 4, 3]);  reshape_default_71 = None
        permute_default_74: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_72, [3, 0, 1, 4, 2]);  permute_default_72 = None
        squeeze_dim_47: "f32[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_74, 4);  permute_default_74 = None
        permute_default_75: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_73, [2, 0, 1, 4, 3]);  permute_default_73 = None
        squeeze_dim_48: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_75, 4);  permute_default_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default_72: "f32[16, 16, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_543, _shape_param_72);  bmm_543 = _shape_param_72 = None
        permute_default_76: "f32[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(reshape_default_72, [0, 1, 4, 3, 2]);  reshape_default_72 = None
        reshape_default_73: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_544, _shape_param_73);  bmm_544 = _shape_param_73 = None
        permute_default_77: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(reshape_default_73, [0, 1, 2, 4, 3]);  reshape_default_73 = None
        permute_default_78: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_76, [3, 0, 1, 4, 2]);  permute_default_76 = None
        squeeze_dim_49: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_78, 4);  permute_default_78 = None
        permute_default_79: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_77, [2, 0, 1, 4, 3]);  permute_default_77 = None
        squeeze_dim_50: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_79, 4);  permute_default_79 = None
        add_tensor: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_dim_48, squeeze_dim_50);  squeeze_dim_48 = squeeze_dim_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        reshape_default_74: "f32[1024, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_47, _shape_param_74);  squeeze_dim_47 = _shape_param_74 = None
        permute_default_80: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_74, [0, 1, 4, 2, 3]);  reshape_default_74 = None
        clone_default_23: "f32[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_80, memory_format = torch.contiguous_format);  permute_default_80 = None
        reshape_default_75: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(clone_default_23, _shape_param_75);  clone_default_23 = _shape_param_75 = None
        squeeze_dim_51: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_75, 0);  reshape_default_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        reshape_default_76: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_46, _shape_param_76);  squeeze_dim_46 = _shape_param_76 = None
        permute_default_81: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_76, [0, 1, 4, 2, 3]);  reshape_default_76 = None
        clone_default_24: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_81, memory_format = torch.contiguous_format);  permute_default_81 = None
        reshape_default_77: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default_24, _shape_param_77);  clone_default_24 = _shape_param_77 = None
        squeeze_dim_52: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_77, 0);  reshape_default_77 = None
        unsqueeze_default: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, 3);  primals_5 = None
        unsqueeze_default_1: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        reshape_default_78: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_78);  unsqueeze_default_1 = _shape_param_78 = None
        permute_default_82: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(reshape_default_78, [0, 2, 1]);  reshape_default_78 = None
        squeeze_dim_53: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_82, 0);  permute_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        reshape_default_79: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_49, _shape_param_79);  squeeze_dim_49 = _shape_param_79 = None
        permute_default_83: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_79, [0, 1, 4, 2, 3]);  reshape_default_79 = None
        clone_default_25: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_83, memory_format = torch.contiguous_format);  permute_default_83 = None
        reshape_default_80: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default_25, _shape_param_80);  clone_default_25 = _shape_param_80 = None
        squeeze_dim_54: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_80, 0);  reshape_default_80 = None
        unsqueeze_default_2: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, 3);  primals_4 = None
        unsqueeze_default_3: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        reshape_default_81: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_81);  unsqueeze_default_3 = _shape_param_81 = None
        permute_default_84: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(reshape_default_81, [0, 2, 1]);  reshape_default_81 = None
        squeeze_dim_55: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_84, 0);  permute_default_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        reshape_default_82: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_82);  add_tensor = _shape_param_82 = None
        permute_default_85: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_82, [0, 1, 4, 2, 3]);  reshape_default_82 = None
        clone_default_26: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_85, memory_format = torch.contiguous_format);  permute_default_85 = None
        reshape_default_83: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default_26, _shape_param_83);  clone_default_26 = _shape_param_83 = None
        squeeze_dim_56: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_83, 0);  reshape_default_83 = None
        unsqueeze_default_4: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, 3);  primals_3 = None
        unsqueeze_default_5: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 4);  unsqueeze_default_4 = None
        reshape_default_84: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_84);  unsqueeze_default_5 = _shape_param_84 = None
        permute_default_86: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(reshape_default_84, [0, 2, 1]);  reshape_default_84 = None
        squeeze_dim_57: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_86, 0);  permute_default_86 = None
        return (permute_default, squeeze_dim_1, squeeze_dim_3, squeeze_dim_5, squeeze_dim_7, squeeze_dim_9, squeeze_dim_11, squeeze_dim_13, squeeze_dim_15, squeeze_dim_17, squeeze_dim_19, squeeze_dim_21, squeeze_dim_23, squeeze_dim_25, squeeze_dim_27, squeeze_dim_29, squeeze_dim_31, squeeze_dim_33, squeeze_dim_35, squeeze_dim_37, squeeze_dim_39, squeeze_dim_41, squeeze_dim_43, squeeze_dim_45, squeeze_dim_51, squeeze_dim_52, squeeze_dim_53, squeeze_dim_54, squeeze_dim_55, squeeze_dim_56, squeeze_dim_57)


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
