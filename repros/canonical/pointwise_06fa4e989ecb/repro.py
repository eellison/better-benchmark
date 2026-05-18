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
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1154 in relative_positional_encoding, code: fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.int64).float()
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_default: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota_default, torch.float32);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1114 in positional_embedding, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, 1);  convert_element_type_default = None
        permute_default: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1]);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1125 in relative_positional_encoding, code: freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.int64).float()
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_default_1: "f32[512]" = torch.ops.prims.convert_element_type.default(iota_default_1, torch.float32);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1126 in relative_positional_encoding, code: inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
        div_tensor: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_default_1, 1024);  convert_element_type_default_1 = None
        pow_scalar: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div_tensor);  div_tensor = None
        reciprocal_default: "f32[512]" = torch.ops.aten.reciprocal.default(pow_scalar);  pow_scalar = None
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1114 in positional_embedding, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        unsqueeze_default_1: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, 1);  mul_tensor = None
        permute_default_1: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_default_1, [1, 0]);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute_default, permute_default_1);  permute_default = permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1115 in positional_embedding, code: pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        sin_default: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_tensor_1)
        cos_default: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_tensor_1);  mul_tensor_1 = None
        cat_default: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin_default, cos_default], -1);  sin_default = cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1116 in positional_embedding, code: pos_emb = pos_emb[:, None, :]
        unsqueeze_default_2: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat_default, 1);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1119 in positional_embedding, code: pos_emb = pos_emb.expand(-1, bsz, -1)
        expand_default: "f32[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_default_2, _shape_param_0);  unsqueeze_default_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1335 in forward, code: pos_emb = pos_emb.to(output_h.device)
        device_put_default: "f32[1024, 16, 1024]" = torch.ops.prims.device_put.default(expand_default, device(type='cuda', index=0));  expand_default = None
        convert_element_type_default_2: "f32[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(device_put_default, torch.float32);  device_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:422 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_3: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_4: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 4);  unsqueeze_default_3 = None
        reshape_default: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_4, _shape_param_1);  unsqueeze_default_4 = _shape_param_1 = None
        squeeze_dim: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default, 0);  reshape_default = None
        unsqueeze_default_5: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_6: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 4);  unsqueeze_default_5 = None
        reshape_default_1: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_6, _shape_param_2);  unsqueeze_default_6 = _shape_param_2 = None
        squeeze_dim_1: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_1, 0);  reshape_default_1 = None
        unsqueeze_default_7: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_8: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 4);  unsqueeze_default_7 = None
        reshape_default_2: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_8, _shape_param_3);  unsqueeze_default_8 = _shape_param_3 = None
        squeeze_dim_2: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None
        unsqueeze_default_9: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_10: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 4);  unsqueeze_default_9 = None
        reshape_default_3: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_10, _shape_param_4);  unsqueeze_default_10 = _shape_param_4 = None
        squeeze_dim_3: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_3, 0);  reshape_default_3 = None
        unsqueeze_default_11: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_12: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 4);  unsqueeze_default_11 = None
        reshape_default_4: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_12, _shape_param_5);  unsqueeze_default_12 = _shape_param_5 = None
        squeeze_dim_4: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_4, 0);  reshape_default_4 = None
        unsqueeze_default_13: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_14: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 4);  unsqueeze_default_13 = None
        reshape_default_5: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_14, _shape_param_6);  unsqueeze_default_14 = _shape_param_6 = None
        squeeze_dim_5: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_5, 0);  reshape_default_5 = None
        unsqueeze_default_15: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_16: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 4);  unsqueeze_default_15 = None
        reshape_default_6: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_16, _shape_param_7);  unsqueeze_default_16 = _shape_param_7 = None
        squeeze_dim_6: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_6, 0);  reshape_default_6 = None
        unsqueeze_default_17: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_18: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 4);  unsqueeze_default_17 = None
        reshape_default_7: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_18, _shape_param_8);  unsqueeze_default_18 = _shape_param_8 = None
        squeeze_dim_7: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_7, 0);  reshape_default_7 = None
        unsqueeze_default_19: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_20: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 4);  unsqueeze_default_19 = None
        reshape_default_8: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_20, _shape_param_9);  unsqueeze_default_20 = _shape_param_9 = None
        squeeze_dim_8: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_8, 0);  reshape_default_8 = None
        unsqueeze_default_21: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_22: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 4);  unsqueeze_default_21 = None
        reshape_default_9: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_22, _shape_param_10);  unsqueeze_default_22 = _shape_param_10 = None
        squeeze_dim_9: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_9, 0);  reshape_default_9 = None
        unsqueeze_default_23: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_24: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 4);  unsqueeze_default_23 = None
        reshape_default_10: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_24, _shape_param_11);  unsqueeze_default_24 = _shape_param_11 = None
        squeeze_dim_10: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_10, 0);  reshape_default_10 = None
        unsqueeze_default_25: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_26: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 4);  unsqueeze_default_25 = None
        reshape_default_11: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_26, _shape_param_12);  unsqueeze_default_26 = _shape_param_12 = None
        squeeze_dim_11: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_11, 0);  reshape_default_11 = None
        unsqueeze_default_27: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_28: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 4);  unsqueeze_default_27 = None
        reshape_default_12: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_28, _shape_param_13);  unsqueeze_default_28 = _shape_param_13 = None
        squeeze_dim_12: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_12, 0);  reshape_default_12 = None
        unsqueeze_default_29: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_30: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 4);  unsqueeze_default_29 = None
        reshape_default_13: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_30, _shape_param_14);  unsqueeze_default_30 = _shape_param_14 = None
        squeeze_dim_13: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_13, 0);  reshape_default_13 = None
        unsqueeze_default_31: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_32: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 4);  unsqueeze_default_31 = None
        reshape_default_14: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_32, _shape_param_15);  unsqueeze_default_32 = _shape_param_15 = None
        squeeze_dim_14: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_14, 0);  reshape_default_14 = None
        unsqueeze_default_33: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_34: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_33, 4);  unsqueeze_default_33 = None
        reshape_default_15: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_34, _shape_param_16);  unsqueeze_default_34 = _shape_param_16 = None
        squeeze_dim_15: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_15, 0);  reshape_default_15 = None
        unsqueeze_default_35: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_36: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 4);  unsqueeze_default_35 = None
        reshape_default_16: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_36, _shape_param_17);  unsqueeze_default_36 = _shape_param_17 = None
        squeeze_dim_16: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_16, 0);  reshape_default_16 = None
        unsqueeze_default_37: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_38: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 4);  unsqueeze_default_37 = None
        reshape_default_17: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_38, _shape_param_18);  unsqueeze_default_38 = _shape_param_18 = None
        squeeze_dim_17: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_17, 0);  reshape_default_17 = None
        unsqueeze_default_39: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_40: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 4);  unsqueeze_default_39 = None
        reshape_default_18: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_40, _shape_param_19);  unsqueeze_default_40 = _shape_param_19 = None
        squeeze_dim_18: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_18, 0);  reshape_default_18 = None
        unsqueeze_default_41: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_42: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 4);  unsqueeze_default_41 = None
        reshape_default_19: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_42, _shape_param_20);  unsqueeze_default_42 = _shape_param_20 = None
        squeeze_dim_19: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_19, 0);  reshape_default_19 = None
        unsqueeze_default_43: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_44: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 4);  unsqueeze_default_43 = None
        reshape_default_20: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_44, _shape_param_21);  unsqueeze_default_44 = _shape_param_21 = None
        squeeze_dim_20: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_20, 0);  reshape_default_20 = None
        unsqueeze_default_45: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_46: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 4);  unsqueeze_default_45 = None
        reshape_default_21: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_46, _shape_param_22);  unsqueeze_default_46 = _shape_param_22 = None
        squeeze_dim_21: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_21, 0);  reshape_default_21 = None
        unsqueeze_default_47: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3)
        unsqueeze_default_48: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 4);  unsqueeze_default_47 = None
        reshape_default_22: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_48, _shape_param_23);  unsqueeze_default_48 = _shape_param_23 = None
        squeeze_dim_22: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_22, 0);  reshape_default_22 = None
        unsqueeze_default_49: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3);  convert_element_type_default_2 = None
        unsqueeze_default_50: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 4);  unsqueeze_default_49 = None
        reshape_default_23: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_50, _shape_param_24);  unsqueeze_default_50 = _shape_param_24 = None
        squeeze_dim_23: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_23, 0);  reshape_default_23 = None
        return (squeeze_dim, squeeze_dim_1, squeeze_dim_2, squeeze_dim_3, squeeze_dim_4, squeeze_dim_5, squeeze_dim_6, squeeze_dim_7, squeeze_dim_8, squeeze_dim_9, squeeze_dim_10, squeeze_dim_11, squeeze_dim_12, squeeze_dim_13, squeeze_dim_14, squeeze_dim_15, squeeze_dim_16, squeeze_dim_17, squeeze_dim_18, squeeze_dim_19, squeeze_dim_20, squeeze_dim_21, squeeze_dim_22, squeeze_dim_23)


def _default_make_inputs():
    return [
    [-1, 16, -1],  # _shape_param_0
    [1, 16384, 1024],  # _shape_param_1
    [1, 16384, 1024],  # _shape_param_2
    [1, 16384, 1024],  # _shape_param_3
    [1, 16384, 1024],  # _shape_param_4
    [1, 16384, 1024],  # _shape_param_5
    [1, 16384, 1024],  # _shape_param_6
    [1, 16384, 1024],  # _shape_param_7
    [1, 16384, 1024],  # _shape_param_8
    [1, 16384, 1024],  # _shape_param_9
    [1, 16384, 1024],  # _shape_param_10
    [1, 16384, 1024],  # _shape_param_11
    [1, 16384, 1024],  # _shape_param_12
    [1, 16384, 1024],  # _shape_param_13
    [1, 16384, 1024],  # _shape_param_14
    [1, 16384, 1024],  # _shape_param_15
    [1, 16384, 1024],  # _shape_param_16
    [1, 16384, 1024],  # _shape_param_17
    [1, 16384, 1024],  # _shape_param_18
    [1, 16384, 1024],  # _shape_param_19
    [1, 16384, 1024],  # _shape_param_20
    [1, 16384, 1024],  # _shape_param_21
    [1, 16384, 1024],  # _shape_param_22
    [1, 16384, 1024],  # _shape_param_23
    [1, 16384, 1024],  # _shape_param_24
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
