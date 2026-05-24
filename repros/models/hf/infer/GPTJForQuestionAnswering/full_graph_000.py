import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 128]", arg1_1: "f32[50400, 4096]", arg2_1: "f32[4096]", arg3_1: "f32[4096]", arg4_1: "f32[4096, 4096]", arg5_1: "f32[4096, 4096]", arg6_1: "f32[4096, 4096]", arg7_1: "f32[2048, 64]", arg8_1: "f32[4096, 4096]", arg9_1: "f32[16384, 4096]", arg10_1: "f32[16384]", arg11_1: "f32[4096, 16384]", arg12_1: "f32[4096]", arg13_1: "f32[4096]", arg14_1: "f32[4096]", arg15_1: "f32[4096, 4096]", arg16_1: "f32[4096, 4096]", arg17_1: "f32[4096, 4096]", arg18_1: "f32[2048, 64]", arg19_1: "f32[4096, 4096]", arg20_1: "f32[16384, 4096]", arg21_1: "f32[16384]", arg22_1: "f32[4096, 16384]", arg23_1: "f32[4096]", arg24_1: "f32[4096]", arg25_1: "f32[4096]", arg26_1: "f32[4096, 4096]", arg27_1: "f32[4096, 4096]", arg28_1: "f32[4096, 4096]", arg29_1: "f32[2048, 64]", arg30_1: "f32[4096, 4096]", arg31_1: "f32[16384, 4096]", arg32_1: "f32[16384]", arg33_1: "f32[4096, 16384]", arg34_1: "f32[4096]", arg35_1: "f32[4096]", arg36_1: "f32[4096]", arg37_1: "f32[4096, 4096]", arg38_1: "f32[4096, 4096]", arg39_1: "f32[4096, 4096]", arg40_1: "f32[2048, 64]", arg41_1: "f32[4096, 4096]", arg42_1: "f32[16384, 4096]", arg43_1: "f32[16384]", arg44_1: "f32[4096, 16384]", arg45_1: "f32[4096]", arg46_1: "f32[4096]", arg47_1: "f32[4096]", arg48_1: "f32[4096, 4096]", arg49_1: "f32[4096, 4096]", arg50_1: "f32[4096, 4096]", arg51_1: "f32[2048, 64]", arg52_1: "f32[4096, 4096]", arg53_1: "f32[16384, 4096]", arg54_1: "f32[16384]", arg55_1: "f32[4096, 16384]", arg56_1: "f32[4096]", arg57_1: "f32[4096]", arg58_1: "f32[4096]", arg59_1: "f32[4096, 4096]", arg60_1: "f32[4096, 4096]", arg61_1: "f32[4096, 4096]", arg62_1: "f32[2048, 64]", arg63_1: "f32[4096, 4096]", arg64_1: "f32[16384, 4096]", arg65_1: "f32[16384]", arg66_1: "f32[4096, 16384]", arg67_1: "f32[4096]", arg68_1: "f32[4096]", arg69_1: "f32[4096]", arg70_1: "f32[4096, 4096]", arg71_1: "f32[4096, 4096]", arg72_1: "f32[4096, 4096]", arg73_1: "f32[2048, 64]", arg74_1: "f32[4096, 4096]", arg75_1: "f32[16384, 4096]", arg76_1: "f32[16384]", arg77_1: "f32[4096, 16384]", arg78_1: "f32[4096]", arg79_1: "f32[4096]", arg80_1: "f32[4096]", arg81_1: "f32[4096, 4096]", arg82_1: "f32[4096, 4096]", arg83_1: "f32[4096, 4096]", arg84_1: "f32[2048, 64]", arg85_1: "f32[4096, 4096]", arg86_1: "f32[16384, 4096]", arg87_1: "f32[16384]", arg88_1: "f32[4096, 16384]", arg89_1: "f32[4096]", arg90_1: "f32[4096]", arg91_1: "f32[4096]", arg92_1: "f32[4096, 4096]", arg93_1: "f32[4096, 4096]", arg94_1: "f32[4096, 4096]", arg95_1: "f32[2048, 64]", arg96_1: "f32[4096, 4096]", arg97_1: "f32[16384, 4096]", arg98_1: "f32[16384]", arg99_1: "f32[4096, 16384]", arg100_1: "f32[4096]", arg101_1: "f32[4096]", arg102_1: "f32[4096]", arg103_1: "f32[4096, 4096]", arg104_1: "f32[4096, 4096]", arg105_1: "f32[4096, 4096]", arg106_1: "f32[2048, 64]", arg107_1: "f32[4096, 4096]", arg108_1: "f32[16384, 4096]", arg109_1: "f32[16384]", arg110_1: "f32[4096, 16384]", arg111_1: "f32[4096]", arg112_1: "f32[4096]", arg113_1: "f32[4096]", arg114_1: "f32[4096, 4096]", arg115_1: "f32[4096, 4096]", arg116_1: "f32[4096, 4096]", arg117_1: "f32[2048, 64]", arg118_1: "f32[4096, 4096]", arg119_1: "f32[16384, 4096]", arg120_1: "f32[16384]", arg121_1: "f32[4096, 16384]", arg122_1: "f32[4096]", arg123_1: "f32[4096]", arg124_1: "f32[4096]", arg125_1: "f32[4096, 4096]", arg126_1: "f32[4096, 4096]", arg127_1: "f32[4096, 4096]", arg128_1: "f32[2048, 64]", arg129_1: "f32[4096, 4096]", arg130_1: "f32[16384, 4096]", arg131_1: "f32[16384]", arg132_1: "f32[4096, 16384]", arg133_1: "f32[4096]", arg134_1: "f32[4096]", arg135_1: "f32[4096]", arg136_1: "f32[4096, 4096]", arg137_1: "f32[4096, 4096]", arg138_1: "f32[4096, 4096]", arg139_1: "f32[2048, 64]", arg140_1: "f32[4096, 4096]", arg141_1: "f32[16384, 4096]", arg142_1: "f32[16384]", arg143_1: "f32[4096, 16384]", arg144_1: "f32[4096]", arg145_1: "f32[4096]", arg146_1: "f32[4096]", arg147_1: "f32[4096, 4096]", arg148_1: "f32[4096, 4096]", arg149_1: "f32[4096, 4096]", arg150_1: "f32[2048, 64]", arg151_1: "f32[4096, 4096]", arg152_1: "f32[16384, 4096]", arg153_1: "f32[16384]", arg154_1: "f32[4096, 16384]", arg155_1: "f32[4096]", arg156_1: "f32[4096]", arg157_1: "f32[4096]", arg158_1: "f32[4096, 4096]", arg159_1: "f32[4096, 4096]", arg160_1: "f32[4096, 4096]", arg161_1: "f32[2048, 64]", arg162_1: "f32[4096, 4096]", arg163_1: "f32[16384, 4096]", arg164_1: "f32[16384]", arg165_1: "f32[4096, 16384]", arg166_1: "f32[4096]", arg167_1: "f32[4096]", arg168_1: "f32[4096]", arg169_1: "f32[4096, 4096]", arg170_1: "f32[4096, 4096]", arg171_1: "f32[4096, 4096]", arg172_1: "f32[2048, 64]", arg173_1: "f32[4096, 4096]", arg174_1: "f32[16384, 4096]", arg175_1: "f32[16384]", arg176_1: "f32[4096, 16384]", arg177_1: "f32[4096]", arg178_1: "f32[4096]", arg179_1: "f32[4096]", arg180_1: "f32[4096, 4096]", arg181_1: "f32[4096, 4096]", arg182_1: "f32[4096, 4096]", arg183_1: "f32[2048, 64]", arg184_1: "f32[4096, 4096]", arg185_1: "f32[16384, 4096]", arg186_1: "f32[16384]", arg187_1: "f32[4096, 16384]", arg188_1: "f32[4096]", arg189_1: "f32[4096]", arg190_1: "f32[4096]", arg191_1: "f32[4096, 4096]", arg192_1: "f32[4096, 4096]", arg193_1: "f32[4096, 4096]", arg194_1: "f32[2048, 64]", arg195_1: "f32[4096, 4096]", arg196_1: "f32[16384, 4096]", arg197_1: "f32[16384]", arg198_1: "f32[4096, 16384]", arg199_1: "f32[4096]", arg200_1: "f32[4096]", arg201_1: "f32[4096]", arg202_1: "f32[4096, 4096]", arg203_1: "f32[4096, 4096]", arg204_1: "f32[4096, 4096]", arg205_1: "f32[2048, 64]", arg206_1: "f32[4096, 4096]", arg207_1: "f32[16384, 4096]", arg208_1: "f32[16384]", arg209_1: "f32[4096, 16384]", arg210_1: "f32[4096]", arg211_1: "f32[4096]", arg212_1: "f32[4096]", arg213_1: "f32[4096, 4096]", arg214_1: "f32[4096, 4096]", arg215_1: "f32[4096, 4096]", arg216_1: "f32[2048, 64]", arg217_1: "f32[4096, 4096]", arg218_1: "f32[16384, 4096]", arg219_1: "f32[16384]", arg220_1: "f32[4096, 16384]", arg221_1: "f32[4096]", arg222_1: "f32[4096]", arg223_1: "f32[4096]", arg224_1: "f32[4096, 4096]", arg225_1: "f32[4096, 4096]", arg226_1: "f32[4096, 4096]", arg227_1: "f32[2048, 64]", arg228_1: "f32[4096, 4096]", arg229_1: "f32[16384, 4096]", arg230_1: "f32[16384]", arg231_1: "f32[4096, 16384]", arg232_1: "f32[4096]", arg233_1: "f32[4096]", arg234_1: "f32[4096]", arg235_1: "f32[4096, 4096]", arg236_1: "f32[4096, 4096]", arg237_1: "f32[4096, 4096]", arg238_1: "f32[2048, 64]", arg239_1: "f32[4096, 4096]", arg240_1: "f32[16384, 4096]", arg241_1: "f32[16384]", arg242_1: "f32[4096, 16384]", arg243_1: "f32[4096]", arg244_1: "f32[4096]", arg245_1: "f32[4096]", arg246_1: "f32[4096, 4096]", arg247_1: "f32[4096, 4096]", arg248_1: "f32[4096, 4096]", arg249_1: "f32[2048, 64]", arg250_1: "f32[4096, 4096]", arg251_1: "f32[16384, 4096]", arg252_1: "f32[16384]", arg253_1: "f32[4096, 16384]", arg254_1: "f32[4096]", arg255_1: "f32[4096]", arg256_1: "f32[4096]", arg257_1: "f32[4096, 4096]", arg258_1: "f32[4096, 4096]", arg259_1: "f32[4096, 4096]", arg260_1: "f32[2048, 64]", arg261_1: "f32[4096, 4096]", arg262_1: "f32[16384, 4096]", arg263_1: "f32[16384]", arg264_1: "f32[4096, 16384]", arg265_1: "f32[4096]", arg266_1: "f32[4096]", arg267_1: "f32[4096]", arg268_1: "f32[4096, 4096]", arg269_1: "f32[4096, 4096]", arg270_1: "f32[4096, 4096]", arg271_1: "f32[2048, 64]", arg272_1: "f32[4096, 4096]", arg273_1: "f32[16384, 4096]", arg274_1: "f32[16384]", arg275_1: "f32[4096, 16384]", arg276_1: "f32[4096]", arg277_1: "f32[4096]", arg278_1: "f32[4096]", arg279_1: "f32[4096, 4096]", arg280_1: "f32[4096, 4096]", arg281_1: "f32[4096, 4096]", arg282_1: "f32[2048, 64]", arg283_1: "f32[4096, 4096]", arg284_1: "f32[16384, 4096]", arg285_1: "f32[16384]", arg286_1: "f32[4096, 16384]", arg287_1: "f32[4096]", arg288_1: "f32[4096]", arg289_1: "f32[4096]", arg290_1: "f32[4096, 4096]", arg291_1: "f32[4096, 4096]", arg292_1: "f32[4096, 4096]", arg293_1: "f32[2048, 64]", arg294_1: "f32[4096, 4096]", arg295_1: "f32[16384, 4096]", arg296_1: "f32[16384]", arg297_1: "f32[4096, 16384]", arg298_1: "f32[4096]", arg299_1: "f32[4096]", arg300_1: "f32[4096]", arg301_1: "f32[4096, 4096]", arg302_1: "f32[4096, 4096]", arg303_1: "f32[4096, 4096]", arg304_1: "f32[2048, 64]", arg305_1: "f32[4096, 4096]", arg306_1: "f32[16384, 4096]", arg307_1: "f32[16384]", arg308_1: "f32[4096, 16384]", arg309_1: "f32[4096]", arg310_1: "f32[4096]", arg311_1: "f32[4096]", arg312_1: "f32[2, 4096]", arg313_1: "f32[2]", arg314_1: "i64[1]", arg315_1: "i64[1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "f32[]" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[1, 128, 4096]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(embedding, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg7_1, [1, 1, 1]);  arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:502 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:503 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_10: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_10, [1, 1, 64]);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = repeat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split = torch.ops.aten.split.Tensor(gather, 32, -1);  gather = None
        getitem_2: "f32[1, 128, 32]" = split[0]
        getitem_3: "f32[1, 128, 32]" = split[1];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  getitem_1 = None
        add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_4: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_4, [128, 4096])
        permute: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "f32[128, 4096]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm, [1, 128, 4096]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_6: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1, [1, 128, 16, 256]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_6: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_6, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_19: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_3, 2)
        unsqueeze_20: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 4);  unsqueeze_19 = None
        expand_4: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_20, [1, 128, 1, 32, 2]);  unsqueeze_20 = None
        clone_4: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_4, [1, 128, 1, 64]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_4: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_6, view_13);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_11: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None
        unsqueeze_21: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_10: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 9223372036854775807, 2);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_22: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_10, 4);  slice_10 = None
        cat_2: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_21, unsqueeze_22], -1);  unsqueeze_21 = unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_14: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_2, [1, 128, 16, 64]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_17: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_2, 2)
        unsqueeze_18: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
        expand_3: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_18, [1, 128, 1, 32, 2]);  unsqueeze_18 = None
        clone_3: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_3, [1, 128, 1, 64]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_5: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_14, view_12);  view_14 = view_12 = None
        add_6: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_7: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_6, 3, 64, 9223372036854775807);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_4: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_6, slice_7], -1);  add_6 = slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_5: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_4, [0, 2, 1, 3]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_2: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_4, [128, 4096])
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "f32[128, 4096]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 4096]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_7: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_3, [1, 128, 16, 256]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_4: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_7, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_13: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_3, 2);  getitem_3 = None
        unsqueeze_14: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
        expand_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_14, [1, 128, 1, 32, 2]);  unsqueeze_14 = None
        clone_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_2: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_4, view_10);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_9: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None
        unsqueeze_15: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_8: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 9223372036854775807, 2);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_16: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_8, 4);  slice_8 = None
        cat_1: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_15, unsqueeze_16], -1);  unsqueeze_15 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_11: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_1, [1, 128, 16, 64]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_11: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_2, 2);  getitem_2 = None
        unsqueeze_12: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        expand_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_12, [1, 128, 1, 32, 2]);  unsqueeze_12 = None
        clone_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_3: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_11, view_9);  view_11 = view_9 = None
        add_5: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_5: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_7, 3, 64, 9223372036854775807);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_3: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_5, slice_5], -1);  add_5 = slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_4: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_4: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_4, [128, 4096])
        permute_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "f32[128, 4096]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 4096]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_8: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_5, [1, 128, 16, 256]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_3: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[128]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_8: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_5: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[1, 129]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = None
        slice_3: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 129)
        slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 128);  cat = None
        sub_1: "i64[1, 128]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[1, 128]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[1, 1, 128, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[1, 1, 1, 128]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[1, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 128, 128]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_1, full_default_2);  expand = full_default_1 = full_default_2 = None

        # No stacktrace found for following nodes
        expand_default_27: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_5, permute_4, permute_3, expand_default_27, False, scale = 0.0625);  permute_5 = permute_4 = permute_3 = expand_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_143: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_27[0];  _scaled_dot_product_efficient_attention_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_7: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_143, [0, 2, 1, 3]);  getitem_143 = None
        clone_6: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_21: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_6, [1, 128, 4096]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_22: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_21, [128, 4096]);  view_21 = None
        permute_8: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_3: "f32[128, 4096]" = torch.ops.aten.mm.default(view_22, permute_8);  view_22 = permute_8 = None
        view_23: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 4096]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_24: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_4, [128, 4096]);  add_4 = None
        permute_9: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg10_1, view_24, permute_9);  arg10_1 = view_24 = permute_9 = None
        view_25: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_25, 0.5)
        pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_25, 3.0)
        mul_7: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_8: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_25, mul_7);  view_25 = mul_7 = None
        mul_8: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_9: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_6, add_9);  mul_6 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_26: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_9, [128, 16384]);  mul_9 = None
        permute_10: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg12_1, view_26, permute_10);  arg12_1 = view_26 = permute_10 = None
        view_27: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 4096]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_10: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_23, view_27);  view_23 = view_27 = None
        add_11: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_10, embedding);  add_10 = embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 128, 1]" = var_mean_1[0]
        getitem_5: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_2: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg18_1, [1, 1, 1]);  arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_23: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_3: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_23, [1, 1, 64]);  unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_1: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_2, 1, repeat_3);  repeat_2 = repeat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_1 = torch.ops.aten.split.Tensor(gather_1, 32, -1);  gather_1 = None
        getitem_6: "f32[1, 128, 32]" = split_1[0]
        getitem_7: "f32[1, 128, 32]" = split_1[1];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_4: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  getitem_5 = None
        add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, arg13_1);  mul_10 = arg13_1 = None
        add_13: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_11, arg14_1);  mul_11 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_28: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_13, [128, 4096])
        permute_11: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_4: "f32[128, 4096]" = torch.ops.aten.mm.default(view_28, permute_11);  view_28 = permute_11 = None
        view_29: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 4096]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_34: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_29, [1, 128, 16, 256]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_14: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_34, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_32: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_7, 2)
        unsqueeze_33: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 4);  unsqueeze_32 = None
        expand_12: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_33, [1, 128, 1, 32, 2]);  unsqueeze_33 = None
        clone_12: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_41: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_12, [1, 128, 1, 64]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_14: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_14, view_41);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_19: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_14, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_3: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_19);  slice_19 = None
        unsqueeze_34: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_3, 4);  neg_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_18: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 9223372036854775807, 2);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_35: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_18, 4);  slice_18 = None
        cat_6: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_34, unsqueeze_35], -1);  unsqueeze_34 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_42: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_6, [1, 128, 16, 64]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_30: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_6, 2)
        unsqueeze_31: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
        expand_11: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_31, [1, 128, 1, 32, 2]);  unsqueeze_31 = None
        clone_11: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_40: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_11, [1, 128, 1, 64]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_15: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_42, view_40);  view_42 = view_40 = None
        add_15: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_15: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_34, 3, 64, 9223372036854775807);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_8: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_15, slice_15], -1);  add_15 = slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_16: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_8, [0, 2, 1, 3]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_30: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_13, [128, 4096])
        permute_12: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_5: "f32[128, 4096]" = torch.ops.aten.mm.default(view_30, permute_12);  view_30 = permute_12 = None
        view_31: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 4096]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_35: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_31, [1, 128, 16, 256]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_12: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_35, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_26: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_7, 2);  getitem_7 = None
        unsqueeze_27: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
        expand_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_27, [1, 128, 1, 32, 2]);  unsqueeze_27 = None
        clone_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_38: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_12: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_12, view_38);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_17: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_12, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_2: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None
        unsqueeze_28: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_2, 4);  neg_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_16: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 9223372036854775807, 2);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_29: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_16, 4);  slice_16 = None
        cat_5: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_28, unsqueeze_29], -1);  unsqueeze_28 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_39: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_5, [1, 128, 16, 64]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_24: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_6, 2);  getitem_6 = None
        unsqueeze_25: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 4);  unsqueeze_24 = None
        expand_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_25, [1, 128, 1, 32, 2]);  unsqueeze_25 = None
        clone_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_37: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_13: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_39, view_37);  view_39 = view_37 = None
        add_14: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_13: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_35, 3, 64, 9223372036854775807);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_7: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_14, slice_13], -1);  add_14 = slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_15: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_32: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_13, [128, 4096])
        permute_13: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_6: "f32[128, 4096]" = torch.ops.aten.mm.default(view_32, permute_13);  view_32 = permute_13 = None
        view_33: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 4096]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_36: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_33, [1, 128, 16, 256]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_14: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None

        # No stacktrace found for following nodes
        expand_default_26: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_16, permute_15, permute_14, expand_default_26, False, scale = 0.0625);  permute_16 = permute_15 = permute_14 = expand_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_142: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_26[0];  _scaled_dot_product_efficient_attention_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_18: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        clone_14: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_49: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_14, [1, 128, 4096]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_50: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_49, [128, 4096]);  view_49 = None
        permute_19: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_7: "f32[128, 4096]" = torch.ops.aten.mm.default(view_50, permute_19);  view_50 = permute_19 = None
        view_51: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 4096]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_52: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_13, [128, 4096]);  add_13 = None
        permute_20: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_2: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg21_1, view_52, permute_20);  arg21_1 = view_52 = permute_20 = None
        view_53: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
        mul_17: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_53, mul_17);  view_53 = mul_17 = None
        mul_18: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_18: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_16, add_18);  mul_16 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_54: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_19, [128, 16384]);  mul_19 = None
        permute_21: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_3: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg23_1, view_54, permute_21);  arg23_1 = view_54 = permute_21 = None
        view_55: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 4096]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_19: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_51, view_55);  view_51 = view_55 = None
        add_20: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_19, add_11);  add_19 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_4: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg29_1, [1, 1, 1]);  arg29_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_36: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_5: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_36, [1, 1, 64]);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_2: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_4, 1, repeat_5);  repeat_4 = repeat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_2 = torch.ops.aten.split.Tensor(gather_2, 32, -1);  gather_2 = None
        getitem_10: "f32[1, 128, 32]" = split_2[0]
        getitem_11: "f32[1, 128, 32]" = split_2[1];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_6: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_20, getitem_9);  getitem_9 = None
        add_21: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_20: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
        mul_21: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, arg24_1);  mul_20 = arg24_1 = None
        add_22: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_21, arg25_1);  mul_21 = arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_56: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_22, [128, 4096])
        permute_22: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_8: "f32[128, 4096]" = torch.ops.aten.mm.default(view_56, permute_22);  view_56 = permute_22 = None
        view_57: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 4096]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_62: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_57, [1, 128, 16, 256]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_22: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_62, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_45: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_11, 2)
        unsqueeze_46: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 4);  unsqueeze_45 = None
        expand_20: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_46, [1, 128, 1, 32, 2]);  unsqueeze_46 = None
        clone_20: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_69: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_20, [1, 128, 1, 64]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_24: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_22, view_69);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_27: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_22, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_5: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_27);  slice_27 = None
        unsqueeze_47: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_5, 4);  neg_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_26: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 9223372036854775807, 2);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_48: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_26, 4);  slice_26 = None
        cat_10: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_47, unsqueeze_48], -1);  unsqueeze_47 = unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_70: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_10, [1, 128, 16, 64]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_43: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_10, 2)
        unsqueeze_44: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 4);  unsqueeze_43 = None
        expand_19: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_44, [1, 128, 1, 32, 2]);  unsqueeze_44 = None
        clone_19: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_68: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_19, [1, 128, 1, 64]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_25: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_70, view_68);  view_70 = view_68 = None
        add_24: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_23: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_62, 3, 64, 9223372036854775807);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_12: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_24, slice_23], -1);  add_24 = slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_27: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_12, [0, 2, 1, 3]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_58: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_22, [128, 4096])
        permute_23: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_9: "f32[128, 4096]" = torch.ops.aten.mm.default(view_58, permute_23);  view_58 = permute_23 = None
        view_59: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 4096]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_63: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_59, [1, 128, 16, 256]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_20: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_63, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_39: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_11, 2);  getitem_11 = None
        unsqueeze_40: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 4);  unsqueeze_39 = None
        expand_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_40, [1, 128, 1, 32, 2]);  unsqueeze_40 = None
        clone_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_66: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_22: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_20, view_66);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_25: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_20, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_4: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_25);  slice_25 = None
        unsqueeze_41: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_4, 4);  neg_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_24: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 9223372036854775807, 2);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_42: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_24, 4);  slice_24 = None
        cat_9: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_41, unsqueeze_42], -1);  unsqueeze_41 = unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_67: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_9, [1, 128, 16, 64]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_37: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_10, 2);  getitem_10 = None
        unsqueeze_38: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 4);  unsqueeze_37 = None
        expand_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_38, [1, 128, 1, 32, 2]);  unsqueeze_38 = None
        clone_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_65: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_23: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_67, view_65);  view_67 = view_65 = None
        add_23: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_21: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_63, 3, 64, 9223372036854775807);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_11: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_23, slice_21], -1);  add_23 = slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_26: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_60: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_22, [128, 4096])
        permute_24: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_10: "f32[128, 4096]" = torch.ops.aten.mm.default(view_60, permute_24);  view_60 = permute_24 = None
        view_61: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 4096]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_64: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_61, [1, 128, 16, 256]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_25: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # No stacktrace found for following nodes
        expand_default_25: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_27, permute_26, permute_25, expand_default_25, False, scale = 0.0625);  permute_27 = permute_26 = permute_25 = expand_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_141: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_25[0];  _scaled_dot_product_efficient_attention_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_29: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1, 3]);  getitem_141 = None
        clone_22: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_77: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_22, [1, 128, 4096]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_78: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_77, [128, 4096]);  view_77 = None
        permute_30: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_11: "f32[128, 4096]" = torch.ops.aten.mm.default(view_78, permute_30);  view_78 = permute_30 = None
        view_79: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 4096]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_80: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_22, [128, 4096]);  add_22 = None
        permute_31: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_4: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg32_1, view_80, permute_31);  arg32_1 = view_80 = permute_31 = None
        view_81: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        pow_3: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 3.0)
        mul_27: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_26: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_81, mul_27);  view_81 = mul_27 = None
        mul_28: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
        tanh_2: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_27: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_26, add_27);  mul_26 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_82: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_29, [128, 16384]);  mul_29 = None
        permute_32: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_5: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg34_1, view_82, permute_32);  arg34_1 = view_82 = permute_32 = None
        view_83: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 4096]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_28: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_79, view_83);  view_79 = view_83 = None
        add_29: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_28, add_20);  add_28 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_13: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_6: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg40_1, [1, 1, 1]);  arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_49: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_7: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_49, [1, 1, 64]);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_3: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_6, 1, repeat_7);  repeat_6 = repeat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_3 = torch.ops.aten.split.Tensor(gather_3, 32, -1);  gather_3 = None
        getitem_14: "f32[1, 128, 32]" = split_3[0]
        getitem_15: "f32[1, 128, 32]" = split_3[1];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_8: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_29, getitem_13);  getitem_13 = None
        add_30: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
        mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, arg35_1);  mul_30 = arg35_1 = None
        add_31: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_31, arg36_1);  mul_31 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_84: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_31, [128, 4096])
        permute_33: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_12: "f32[128, 4096]" = torch.ops.aten.mm.default(view_84, permute_33);  view_84 = permute_33 = None
        view_85: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 4096]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_90: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_85, [1, 128, 16, 256]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_30: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_90, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_58: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_15, 2)
        unsqueeze_59: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 4);  unsqueeze_58 = None
        expand_28: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_59, [1, 128, 1, 32, 2]);  unsqueeze_59 = None
        clone_28: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_97: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_28, [1, 128, 1, 64]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_34: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_30, view_97);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_35: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_30, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_7: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_35);  slice_35 = None
        unsqueeze_60: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_7, 4);  neg_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_34: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 9223372036854775807, 2);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_61: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_34, 4);  slice_34 = None
        cat_14: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_60, unsqueeze_61], -1);  unsqueeze_60 = unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_98: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_14, [1, 128, 16, 64]);  cat_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_56: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_14, 2)
        unsqueeze_57: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, 4);  unsqueeze_56 = None
        expand_27: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_57, [1, 128, 1, 32, 2]);  unsqueeze_57 = None
        clone_27: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_96: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_27, [1, 128, 1, 64]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_35: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_98, view_96);  view_98 = view_96 = None
        add_33: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_31: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_90, 3, 64, 9223372036854775807);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_16: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_33, slice_31], -1);  add_33 = slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_38: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_16, [0, 2, 1, 3]);  cat_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_86: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_31, [128, 4096])
        permute_34: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_13: "f32[128, 4096]" = torch.ops.aten.mm.default(view_86, permute_34);  view_86 = permute_34 = None
        view_87: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 4096]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_91: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_87, [1, 128, 16, 256]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_28: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_91, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_52: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_15, 2);  getitem_15 = None
        unsqueeze_53: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 4);  unsqueeze_52 = None
        expand_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_53, [1, 128, 1, 32, 2]);  unsqueeze_53 = None
        clone_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_94: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_32: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_28, view_94);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_33: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_28, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_6: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_33);  slice_33 = None
        unsqueeze_54: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_6, 4);  neg_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_32: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 9223372036854775807, 2);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_55: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_32, 4);  slice_32 = None
        cat_13: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_54, unsqueeze_55], -1);  unsqueeze_54 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_95: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_13, [1, 128, 16, 64]);  cat_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_50: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_14, 2);  getitem_14 = None
        unsqueeze_51: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 4);  unsqueeze_50 = None
        expand_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_51, [1, 128, 1, 32, 2]);  unsqueeze_51 = None
        clone_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_93: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_33: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_95, view_93);  view_95 = view_93 = None
        add_32: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_29: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_91, 3, 64, 9223372036854775807);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_15: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_32, slice_29], -1);  add_32 = slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_37: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_15, [0, 2, 1, 3]);  cat_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_88: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_31, [128, 4096])
        permute_35: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_14: "f32[128, 4096]" = torch.ops.aten.mm.default(view_88, permute_35);  view_88 = permute_35 = None
        view_89: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 4096]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_92: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_89, [1, 128, 16, 256]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_36: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # No stacktrace found for following nodes
        expand_default_24: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_37, permute_36, expand_default_24, False, scale = 0.0625);  permute_38 = permute_37 = permute_36 = expand_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_140: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_24[0];  _scaled_dot_product_efficient_attention_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_40: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None
        clone_30: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_105: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_30, [1, 128, 4096]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_105, [128, 4096]);  view_105 = None
        permute_41: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_15: "f32[128, 4096]" = torch.ops.aten.mm.default(view_106, permute_41);  view_106 = permute_41 = None
        view_107: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 4096]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_108: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_31, [128, 4096]);  add_31 = None
        permute_42: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_6: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg43_1, view_108, permute_42);  arg43_1 = view_108 = permute_42 = None
        view_109: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        pow_4: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
        mul_37: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_35: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_109, mul_37);  view_109 = mul_37 = None
        mul_38: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
        tanh_3: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_36: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_36);  mul_36 = add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_110: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_39, [128, 16384]);  mul_39 = None
        permute_43: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_7: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg45_1, view_110, permute_43);  arg45_1 = view_110 = permute_43 = None
        view_111: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 4096]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_37: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_107, view_111);  view_107 = view_111 = None
        add_38: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_37, add_29);  add_37 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 128, 1]" = var_mean_4[0]
        getitem_17: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_8: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg51_1, [1, 1, 1]);  arg51_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_62: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_9: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_62, [1, 1, 64]);  unsqueeze_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_4: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_8, 1, repeat_9);  repeat_8 = repeat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_4 = torch.ops.aten.split.Tensor(gather_4, 32, -1);  gather_4 = None
        getitem_18: "f32[1, 128, 32]" = split_4[0]
        getitem_19: "f32[1, 128, 32]" = split_4[1];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_10: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_38, getitem_17);  getitem_17 = None
        add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_40: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_4);  sub_10 = rsqrt_4 = None
        mul_41: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, arg46_1);  mul_40 = arg46_1 = None
        add_40: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_41, arg47_1);  mul_41 = arg47_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_112: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_40, [128, 4096])
        permute_44: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_16: "f32[128, 4096]" = torch.ops.aten.mm.default(view_112, permute_44);  view_112 = permute_44 = None
        view_113: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 4096]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_118: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_113, [1, 128, 16, 256]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_38: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_118, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_71: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_19, 2)
        unsqueeze_72: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 4);  unsqueeze_71 = None
        expand_36: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_72, [1, 128, 1, 32, 2]);  unsqueeze_72 = None
        clone_36: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_125: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_36, [1, 128, 1, 64]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_44: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_38, view_125);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_43: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_38, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_9: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_43);  slice_43 = None
        unsqueeze_73: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_9, 4);  neg_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_42: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 9223372036854775807, 2);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_74: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_42, 4);  slice_42 = None
        cat_18: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_73, unsqueeze_74], -1);  unsqueeze_73 = unsqueeze_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_126: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_18, [1, 128, 16, 64]);  cat_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_69: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_18, 2)
        unsqueeze_70: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 4);  unsqueeze_69 = None
        expand_35: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_70, [1, 128, 1, 32, 2]);  unsqueeze_70 = None
        clone_35: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_124: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_35, [1, 128, 1, 64]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_45: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_126, view_124);  view_126 = view_124 = None
        add_42: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_39: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_118, 3, 64, 9223372036854775807);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_20: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_42, slice_39], -1);  add_42 = slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_49: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_20, [0, 2, 1, 3]);  cat_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_114: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_40, [128, 4096])
        permute_45: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_17: "f32[128, 4096]" = torch.ops.aten.mm.default(view_114, permute_45);  view_114 = permute_45 = None
        view_115: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 4096]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_119: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_115, [1, 128, 16, 256]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_36: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_119, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_65: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_19, 2);  getitem_19 = None
        unsqueeze_66: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 4);  unsqueeze_65 = None
        expand_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_66, [1, 128, 1, 32, 2]);  unsqueeze_66 = None
        clone_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_122: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_42: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_36, view_122);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_41: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_36, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_8: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_41);  slice_41 = None
        unsqueeze_67: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_8, 4);  neg_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_40: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 9223372036854775807, 2);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_68: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_40, 4);  slice_40 = None
        cat_17: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_67, unsqueeze_68], -1);  unsqueeze_67 = unsqueeze_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_123: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_17, [1, 128, 16, 64]);  cat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_63: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_18, 2);  getitem_18 = None
        unsqueeze_64: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
        expand_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_64, [1, 128, 1, 32, 2]);  unsqueeze_64 = None
        clone_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_121: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_43: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_123, view_121);  view_123 = view_121 = None
        add_41: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_37: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_119, 3, 64, 9223372036854775807);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_19: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_41, slice_37], -1);  add_41 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_48: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_19, [0, 2, 1, 3]);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_116: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_40, [128, 4096])
        permute_46: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_18: "f32[128, 4096]" = torch.ops.aten.mm.default(view_116, permute_46);  view_116 = permute_46 = None
        view_117: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 4096]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_120: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_117, [1, 128, 16, 256]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_47: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # No stacktrace found for following nodes
        expand_default_23: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_49, permute_48, permute_47, expand_default_23, False, scale = 0.0625);  permute_49 = permute_48 = permute_47 = expand_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_139: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_23[0];  _scaled_dot_product_efficient_attention_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_51: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        clone_38: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_133: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_38, [1, 128, 4096]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_134: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_133, [128, 4096]);  view_133 = None
        permute_52: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_19: "f32[128, 4096]" = torch.ops.aten.mm.default(view_134, permute_52);  view_134 = permute_52 = None
        view_135: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 4096]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_136: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_40, [128, 4096]);  add_40 = None
        permute_53: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_8: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg54_1, view_136, permute_53);  arg54_1 = view_136 = permute_53 = None
        view_137: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        pow_5: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_137, 3.0)
        mul_47: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_44: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_137, mul_47);  view_137 = mul_47 = None
        mul_48: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_4: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_45: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_46, add_45);  mul_46 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_138: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_49, [128, 16384]);  mul_49 = None
        permute_54: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_9: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg56_1, view_138, permute_54);  arg56_1 = view_138 = permute_54 = None
        view_139: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 4096]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_46: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_135, view_139);  view_135 = view_139 = None
        add_47: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_46, add_38);  add_46 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_21: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_10: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg62_1, [1, 1, 1]);  arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_75: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_11: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_75, [1, 1, 64]);  unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_5: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_10, 1, repeat_11);  repeat_10 = repeat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_5 = torch.ops.aten.split.Tensor(gather_5, 32, -1);  gather_5 = None
        getitem_22: "f32[1, 128, 32]" = split_5[0]
        getitem_23: "f32[1, 128, 32]" = split_5[1];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_12: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_47, getitem_21);  getitem_21 = None
        add_48: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_50: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = rsqrt_5 = None
        mul_51: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, arg57_1);  mul_50 = arg57_1 = None
        add_49: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_51, arg58_1);  mul_51 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_140: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096])
        permute_55: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_20: "f32[128, 4096]" = torch.ops.aten.mm.default(view_140, permute_55);  view_140 = permute_55 = None
        view_141: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 4096]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_146: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_141, [1, 128, 16, 256]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_46: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_146, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_84: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_23, 2)
        unsqueeze_85: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 4);  unsqueeze_84 = None
        expand_44: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_85, [1, 128, 1, 32, 2]);  unsqueeze_85 = None
        clone_44: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_153: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_44, [1, 128, 1, 64]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_54: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_46, view_153);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_51: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_46, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_11: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_51);  slice_51 = None
        unsqueeze_86: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_11, 4);  neg_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_50: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 9223372036854775807, 2);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_87: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_50, 4);  slice_50 = None
        cat_22: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_86, unsqueeze_87], -1);  unsqueeze_86 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_154: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_22, [1, 128, 16, 64]);  cat_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_82: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_22, 2)
        unsqueeze_83: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 4);  unsqueeze_82 = None
        expand_43: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_83, [1, 128, 1, 32, 2]);  unsqueeze_83 = None
        clone_43: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_152: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_43, [1, 128, 1, 64]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_55: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_154, view_152);  view_154 = view_152 = None
        add_51: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_47: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_146, 3, 64, 9223372036854775807);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_24: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_51, slice_47], -1);  add_51 = slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_60: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_24, [0, 2, 1, 3]);  cat_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_142: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096])
        permute_56: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_21: "f32[128, 4096]" = torch.ops.aten.mm.default(view_142, permute_56);  view_142 = permute_56 = None
        view_143: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 4096]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_147: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_143, [1, 128, 16, 256]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_44: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_147, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_78: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_23, 2);  getitem_23 = None
        unsqueeze_79: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
        expand_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_79, [1, 128, 1, 32, 2]);  unsqueeze_79 = None
        clone_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_150: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_52: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_44, view_150);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_49: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_44, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_10: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_49);  slice_49 = None
        unsqueeze_80: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_10, 4);  neg_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_48: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 9223372036854775807, 2);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_81: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_48, 4);  slice_48 = None
        cat_21: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_80, unsqueeze_81], -1);  unsqueeze_80 = unsqueeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_151: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_21, [1, 128, 16, 64]);  cat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_76: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_22, 2);  getitem_22 = None
        unsqueeze_77: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
        expand_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_77, [1, 128, 1, 32, 2]);  unsqueeze_77 = None
        clone_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_149: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_53: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_151, view_149);  view_151 = view_149 = None
        add_50: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_45: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_147, 3, 64, 9223372036854775807);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_23: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_50, slice_45], -1);  add_50 = slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_59: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_23, [0, 2, 1, 3]);  cat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_144: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096])
        permute_57: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_22: "f32[128, 4096]" = torch.ops.aten.mm.default(view_144, permute_57);  view_144 = permute_57 = None
        view_145: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 4096]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_148: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_145, [1, 128, 16, 256]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_58: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # No stacktrace found for following nodes
        expand_default_22: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_60, permute_59, permute_58, expand_default_22, False, scale = 0.0625);  permute_60 = permute_59 = permute_58 = expand_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_138: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_22[0];  _scaled_dot_product_efficient_attention_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_62: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
        clone_46: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_161: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_46, [1, 128, 4096]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_162: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_161, [128, 4096]);  view_161 = None
        permute_63: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_23: "f32[128, 4096]" = torch.ops.aten.mm.default(view_162, permute_63);  view_162 = permute_63 = None
        view_163: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 4096]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_164: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096]);  add_49 = None
        permute_64: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_10: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg65_1, view_164, permute_64);  arg65_1 = view_164 = permute_64 = None
        view_165: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_165, 0.5)
        pow_6: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_165, 3.0)
        mul_57: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_53: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_165, mul_57);  view_165 = mul_57 = None
        mul_58: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_5: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_54: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_56, add_54);  mul_56 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_166: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_59, [128, 16384]);  mul_59 = None
        permute_65: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_11: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg67_1, view_166, permute_65);  arg67_1 = view_166 = permute_65 = None
        view_167: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 4096]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_55: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_163, view_167);  view_163 = view_167 = None
        add_56: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_55, add_47);  add_55 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_25: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_12: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg73_1, [1, 1, 1]);  arg73_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_88: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_13: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_88, [1, 1, 64]);  unsqueeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_6: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_12, 1, repeat_13);  repeat_12 = repeat_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_6 = torch.ops.aten.split.Tensor(gather_6, 32, -1);  gather_6 = None
        getitem_26: "f32[1, 128, 32]" = split_6[0]
        getitem_27: "f32[1, 128, 32]" = split_6[1];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_14: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_56, getitem_25);  getitem_25 = None
        add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_60: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_6);  sub_14 = rsqrt_6 = None
        mul_61: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, arg68_1);  mul_60 = arg68_1 = None
        add_58: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_61, arg69_1);  mul_61 = arg69_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_168: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_58, [128, 4096])
        permute_66: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_24: "f32[128, 4096]" = torch.ops.aten.mm.default(view_168, permute_66);  view_168 = permute_66 = None
        view_169: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 4096]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_174: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_169, [1, 128, 16, 256]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_54: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_174, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_97: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_27, 2)
        unsqueeze_98: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 4);  unsqueeze_97 = None
        expand_52: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_98, [1, 128, 1, 32, 2]);  unsqueeze_98 = None
        clone_52: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_181: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_52, [1, 128, 1, 64]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_64: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_54, view_181);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_59: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_54, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_13: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_59);  slice_59 = None
        unsqueeze_99: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_13, 4);  neg_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_58: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_54, 3, 0, 9223372036854775807, 2);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_100: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_58, 4);  slice_58 = None
        cat_26: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_99, unsqueeze_100], -1);  unsqueeze_99 = unsqueeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_182: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_26, [1, 128, 16, 64]);  cat_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_95: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_26, 2)
        unsqueeze_96: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 4);  unsqueeze_95 = None
        expand_51: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_96, [1, 128, 1, 32, 2]);  unsqueeze_96 = None
        clone_51: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_180: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_51, [1, 128, 1, 64]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_65: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_182, view_180);  view_182 = view_180 = None
        add_60: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_55: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_174, 3, 64, 9223372036854775807);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_28: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_60, slice_55], -1);  add_60 = slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_71: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_28, [0, 2, 1, 3]);  cat_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_170: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_58, [128, 4096])
        permute_67: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_25: "f32[128, 4096]" = torch.ops.aten.mm.default(view_170, permute_67);  view_170 = permute_67 = None
        view_171: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 4096]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_175: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_171, [1, 128, 16, 256]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_52: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_175, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_91: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_27, 2);  getitem_27 = None
        unsqueeze_92: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 4);  unsqueeze_91 = None
        expand_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_92, [1, 128, 1, 32, 2]);  unsqueeze_92 = None
        clone_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_178: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_62: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_52, view_178);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_57: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_52, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_12: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_57);  slice_57 = None
        unsqueeze_93: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_12, 4);  neg_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_56: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_52, 3, 0, 9223372036854775807, 2);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_94: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_56, 4);  slice_56 = None
        cat_25: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_93, unsqueeze_94], -1);  unsqueeze_93 = unsqueeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_179: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_25, [1, 128, 16, 64]);  cat_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_89: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_26, 2);  getitem_26 = None
        unsqueeze_90: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 4);  unsqueeze_89 = None
        expand_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_90, [1, 128, 1, 32, 2]);  unsqueeze_90 = None
        clone_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_177: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_63: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_179, view_177);  view_179 = view_177 = None
        add_59: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_53: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_175, 3, 64, 9223372036854775807);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_27: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_59, slice_53], -1);  add_59 = slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_70: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_27, [0, 2, 1, 3]);  cat_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_172: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_58, [128, 4096])
        permute_68: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_26: "f32[128, 4096]" = torch.ops.aten.mm.default(view_172, permute_68);  view_172 = permute_68 = None
        view_173: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 4096]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_176: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_173, [1, 128, 16, 256]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_69: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None

        # No stacktrace found for following nodes
        expand_default_21: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_70, permute_69, expand_default_21, False, scale = 0.0625);  permute_71 = permute_70 = permute_69 = expand_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_137: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_21[0];  _scaled_dot_product_efficient_attention_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_73: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
        clone_54: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_189: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_54, [1, 128, 4096]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_190: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_189, [128, 4096]);  view_189 = None
        permute_74: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_27: "f32[128, 4096]" = torch.ops.aten.mm.default(view_190, permute_74);  view_190 = permute_74 = None
        view_191: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 4096]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_192: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_58, [128, 4096]);  add_58 = None
        permute_75: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_12: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg76_1, view_192, permute_75);  arg76_1 = view_192 = permute_75 = None
        view_193: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_193, 0.5)
        pow_7: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_193, 3.0)
        mul_67: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_62: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_193, mul_67);  view_193 = mul_67 = None
        mul_68: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_6: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_63: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_66, add_63);  mul_66 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_194: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_69, [128, 16384]);  mul_69 = None
        permute_76: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_13: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg78_1, view_194, permute_76);  arg78_1 = view_194 = permute_76 = None
        view_195: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 4096]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_64: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_191, view_195);  view_191 = view_195 = None
        add_65: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_64, add_56);  add_64 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_29: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_14: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg84_1, [1, 1, 1]);  arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_101: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_15: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_101, [1, 1, 64]);  unsqueeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_7: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_14, 1, repeat_15);  repeat_14 = repeat_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_7 = torch.ops.aten.split.Tensor(gather_7, 32, -1);  gather_7 = None
        getitem_30: "f32[1, 128, 32]" = split_7[0]
        getitem_31: "f32[1, 128, 32]" = split_7[1];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_16: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_65, getitem_29);  getitem_29 = None
        add_66: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_70: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = rsqrt_7 = None
        mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, arg79_1);  mul_70 = arg79_1 = None
        add_67: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_71, arg80_1);  mul_71 = arg80_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_196: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_67, [128, 4096])
        permute_77: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_28: "f32[128, 4096]" = torch.ops.aten.mm.default(view_196, permute_77);  view_196 = permute_77 = None
        view_197: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 4096]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_202: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_197, [1, 128, 16, 256]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_62: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_202, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_110: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_31, 2)
        unsqueeze_111: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 4);  unsqueeze_110 = None
        expand_60: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_111, [1, 128, 1, 32, 2]);  unsqueeze_111 = None
        clone_60: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_209: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_60, [1, 128, 1, 64]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_74: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_62, view_209);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_67: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_62, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_15: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_67);  slice_67 = None
        unsqueeze_112: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_15, 4);  neg_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_66: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_62, 3, 0, 9223372036854775807, 2);  slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_113: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_66, 4);  slice_66 = None
        cat_30: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_112, unsqueeze_113], -1);  unsqueeze_112 = unsqueeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_210: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_30, [1, 128, 16, 64]);  cat_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_108: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_30, 2)
        unsqueeze_109: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 4);  unsqueeze_108 = None
        expand_59: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_109, [1, 128, 1, 32, 2]);  unsqueeze_109 = None
        clone_59: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_208: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_59, [1, 128, 1, 64]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_75: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_210, view_208);  view_210 = view_208 = None
        add_69: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_63: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_202, 3, 64, 9223372036854775807);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_32: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_69, slice_63], -1);  add_69 = slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_82: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_32, [0, 2, 1, 3]);  cat_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_198: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_67, [128, 4096])
        permute_78: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_29: "f32[128, 4096]" = torch.ops.aten.mm.default(view_198, permute_78);  view_198 = permute_78 = None
        view_199: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 4096]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_203: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_199, [1, 128, 16, 256]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_60: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_203, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_104: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_31, 2);  getitem_31 = None
        unsqueeze_105: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 4);  unsqueeze_104 = None
        expand_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_105, [1, 128, 1, 32, 2]);  unsqueeze_105 = None
        clone_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_206: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_72: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_60, view_206);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_65: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_60, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_14: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_65);  slice_65 = None
        unsqueeze_106: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_14, 4);  neg_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_64: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_60, 3, 0, 9223372036854775807, 2);  slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_107: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_64, 4);  slice_64 = None
        cat_29: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_106, unsqueeze_107], -1);  unsqueeze_106 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_207: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_29, [1, 128, 16, 64]);  cat_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_102: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_30, 2);  getitem_30 = None
        unsqueeze_103: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 4);  unsqueeze_102 = None
        expand_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_103, [1, 128, 1, 32, 2]);  unsqueeze_103 = None
        clone_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_205: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_73: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_207, view_205);  view_207 = view_205 = None
        add_68: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_61: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_203, 3, 64, 9223372036854775807);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_31: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_68, slice_61], -1);  add_68 = slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_81: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_31, [0, 2, 1, 3]);  cat_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_200: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_67, [128, 4096])
        permute_79: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_30: "f32[128, 4096]" = torch.ops.aten.mm.default(view_200, permute_79);  view_200 = permute_79 = None
        view_201: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 4096]);  mm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_204: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_201, [1, 128, 16, 256]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_80: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # No stacktrace found for following nodes
        expand_default_20: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_82, permute_81, permute_80, expand_default_20, False, scale = 0.0625);  permute_82 = permute_81 = permute_80 = expand_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_136: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_20[0];  _scaled_dot_product_efficient_attention_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_84: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3]);  getitem_136 = None
        clone_62: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_217: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_62, [1, 128, 4096]);  clone_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_218: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_217, [128, 4096]);  view_217 = None
        permute_85: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_31: "f32[128, 4096]" = torch.ops.aten.mm.default(view_218, permute_85);  view_218 = permute_85 = None
        view_219: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 4096]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_220: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_67, [128, 4096]);  add_67 = None
        permute_86: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_14: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg87_1, view_220, permute_86);  arg87_1 = view_220 = permute_86 = None
        view_221: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_221, 0.5)
        pow_8: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_221, 3.0)
        mul_77: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_71: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_221, mul_77);  view_221 = mul_77 = None
        mul_78: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_7: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_72: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_72);  mul_76 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_222: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_79, [128, 16384]);  mul_79 = None
        permute_87: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_15: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg89_1, view_222, permute_87);  arg89_1 = view_222 = permute_87 = None
        view_223: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 4096]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_73: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_219, view_223);  view_219 = view_223 = None
        add_74: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_73, add_65);  add_73 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1]" = var_mean_8[0]
        getitem_33: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_16: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg95_1, [1, 1, 1]);  arg95_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_114: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_17: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_114, [1, 1, 64]);  unsqueeze_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_8: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_16, 1, repeat_17);  repeat_16 = repeat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_8 = torch.ops.aten.split.Tensor(gather_8, 32, -1);  gather_8 = None
        getitem_34: "f32[1, 128, 32]" = split_8[0]
        getitem_35: "f32[1, 128, 32]" = split_8[1];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_18: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_74, getitem_33);  getitem_33 = None
        add_75: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_80: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_8);  sub_18 = rsqrt_8 = None
        mul_81: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, arg90_1);  mul_80 = arg90_1 = None
        add_76: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_81, arg91_1);  mul_81 = arg91_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_224: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_76, [128, 4096])
        permute_88: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_32: "f32[128, 4096]" = torch.ops.aten.mm.default(view_224, permute_88);  view_224 = permute_88 = None
        view_225: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 4096]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_230: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_225, [1, 128, 16, 256]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_70: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_230, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_123: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_35, 2)
        unsqueeze_124: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 4);  unsqueeze_123 = None
        expand_68: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_124, [1, 128, 1, 32, 2]);  unsqueeze_124 = None
        clone_68: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_237: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_68, [1, 128, 1, 64]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_84: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_70, view_237);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_75: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_70, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_17: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_75);  slice_75 = None
        unsqueeze_125: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_17, 4);  neg_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_74: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_70, 3, 0, 9223372036854775807, 2);  slice_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_126: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_74, 4);  slice_74 = None
        cat_34: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_125, unsqueeze_126], -1);  unsqueeze_125 = unsqueeze_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_238: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_34, [1, 128, 16, 64]);  cat_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_121: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_34, 2)
        unsqueeze_122: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 4);  unsqueeze_121 = None
        expand_67: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_122, [1, 128, 1, 32, 2]);  unsqueeze_122 = None
        clone_67: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_236: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_67, [1, 128, 1, 64]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_85: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_238, view_236);  view_238 = view_236 = None
        add_78: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_71: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_230, 3, 64, 9223372036854775807);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_36: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_78, slice_71], -1);  add_78 = slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_93: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_36, [0, 2, 1, 3]);  cat_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_226: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_76, [128, 4096])
        permute_89: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_33: "f32[128, 4096]" = torch.ops.aten.mm.default(view_226, permute_89);  view_226 = permute_89 = None
        view_227: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 4096]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_231: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_227, [1, 128, 16, 256]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_68: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_231, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_117: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_35, 2);  getitem_35 = None
        unsqueeze_118: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 4);  unsqueeze_117 = None
        expand_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_118, [1, 128, 1, 32, 2]);  unsqueeze_118 = None
        clone_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_234: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_82: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_68, view_234);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_73: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_68, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_16: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_73);  slice_73 = None
        unsqueeze_119: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_16, 4);  neg_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_72: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_68, 3, 0, 9223372036854775807, 2);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_120: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_72, 4);  slice_72 = None
        cat_33: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_119, unsqueeze_120], -1);  unsqueeze_119 = unsqueeze_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_235: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_33, [1, 128, 16, 64]);  cat_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_115: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_34, 2);  getitem_34 = None
        unsqueeze_116: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 4);  unsqueeze_115 = None
        expand_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_116, [1, 128, 1, 32, 2]);  unsqueeze_116 = None
        clone_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_233: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_83: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_235, view_233);  view_235 = view_233 = None
        add_77: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_69: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_231, 3, 64, 9223372036854775807);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_35: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_77, slice_69], -1);  add_77 = slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_92: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_35, [0, 2, 1, 3]);  cat_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_228: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_76, [128, 4096])
        permute_90: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_34: "f32[128, 4096]" = torch.ops.aten.mm.default(view_228, permute_90);  view_228 = permute_90 = None
        view_229: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 4096]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_232: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_229, [1, 128, 16, 256]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_91: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # No stacktrace found for following nodes
        expand_default_19: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_93, permute_92, permute_91, expand_default_19, False, scale = 0.0625);  permute_93 = permute_92 = permute_91 = expand_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_135: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_19[0];  _scaled_dot_product_efficient_attention_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_95: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
        clone_70: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_245: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_70, [1, 128, 4096]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_246: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_245, [128, 4096]);  view_245 = None
        permute_96: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_35: "f32[128, 4096]" = torch.ops.aten.mm.default(view_246, permute_96);  view_246 = permute_96 = None
        view_247: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 4096]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_248: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_76, [128, 4096]);  add_76 = None
        permute_97: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_16: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg98_1, view_248, permute_97);  arg98_1 = view_248 = permute_97 = None
        view_249: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
        pow_9: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_249, 3.0)
        mul_87: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_80: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_249, mul_87);  view_249 = mul_87 = None
        mul_88: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_81: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_86, add_81);  mul_86 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_250: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_89, [128, 16384]);  mul_89 = None
        permute_98: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_17: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg100_1, view_250, permute_98);  arg100_1 = view_250 = permute_98 = None
        view_251: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 4096]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_82: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_247, view_251);  view_247 = view_251 = None
        add_83: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_82, add_74);  add_82 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 128, 1]" = var_mean_9[0]
        getitem_37: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_18: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg106_1, [1, 1, 1]);  arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_127: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_19: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_127, [1, 1, 64]);  unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_9: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_18, 1, repeat_19);  repeat_18 = repeat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_9 = torch.ops.aten.split.Tensor(gather_9, 32, -1);  gather_9 = None
        getitem_38: "f32[1, 128, 32]" = split_9[0]
        getitem_39: "f32[1, 128, 32]" = split_9[1];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_20: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_83, getitem_37);  getitem_37 = None
        add_84: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_90: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = rsqrt_9 = None
        mul_91: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, arg101_1);  mul_90 = arg101_1 = None
        add_85: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_91, arg102_1);  mul_91 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_252: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_85, [128, 4096])
        permute_99: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_36: "f32[128, 4096]" = torch.ops.aten.mm.default(view_252, permute_99);  view_252 = permute_99 = None
        view_253: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 4096]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_258: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_253, [1, 128, 16, 256]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_78: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_258, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_136: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_39, 2)
        unsqueeze_137: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 4);  unsqueeze_136 = None
        expand_76: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_137, [1, 128, 1, 32, 2]);  unsqueeze_137 = None
        clone_76: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_265: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_76, [1, 128, 1, 64]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_94: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_78, view_265);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_83: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_78, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_19: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_83);  slice_83 = None
        unsqueeze_138: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_19, 4);  neg_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_82: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_78, 3, 0, 9223372036854775807, 2);  slice_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_139: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_82, 4);  slice_82 = None
        cat_38: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_138, unsqueeze_139], -1);  unsqueeze_138 = unsqueeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_266: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_38, [1, 128, 16, 64]);  cat_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_134: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_38, 2)
        unsqueeze_135: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 4);  unsqueeze_134 = None
        expand_75: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_135, [1, 128, 1, 32, 2]);  unsqueeze_135 = None
        clone_75: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_264: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_75, [1, 128, 1, 64]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_95: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_266, view_264);  view_266 = view_264 = None
        add_87: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_79: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_258, 3, 64, 9223372036854775807);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_40: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_87, slice_79], -1);  add_87 = slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_104: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_40, [0, 2, 1, 3]);  cat_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_254: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_85, [128, 4096])
        permute_100: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_37: "f32[128, 4096]" = torch.ops.aten.mm.default(view_254, permute_100);  view_254 = permute_100 = None
        view_255: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 4096]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_259: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_255, [1, 128, 16, 256]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_76: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_259, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_130: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_39, 2);  getitem_39 = None
        unsqueeze_131: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
        expand_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_131, [1, 128, 1, 32, 2]);  unsqueeze_131 = None
        clone_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_262: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_92: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_76, view_262);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_81: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_76, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_18: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_81);  slice_81 = None
        unsqueeze_132: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_18, 4);  neg_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_80: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_76, 3, 0, 9223372036854775807, 2);  slice_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_133: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_80, 4);  slice_80 = None
        cat_37: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_132, unsqueeze_133], -1);  unsqueeze_132 = unsqueeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_263: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_37, [1, 128, 16, 64]);  cat_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_128: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_38, 2);  getitem_38 = None
        unsqueeze_129: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 4);  unsqueeze_128 = None
        expand_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_129, [1, 128, 1, 32, 2]);  unsqueeze_129 = None
        clone_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_261: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_93: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_263, view_261);  view_263 = view_261 = None
        add_86: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_77: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_259, 3, 64, 9223372036854775807);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_39: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_86, slice_77], -1);  add_86 = slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_103: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_39, [0, 2, 1, 3]);  cat_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_256: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_85, [128, 4096])
        permute_101: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_38: "f32[128, 4096]" = torch.ops.aten.mm.default(view_256, permute_101);  view_256 = permute_101 = None
        view_257: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 4096]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_260: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_257, [1, 128, 16, 256]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_102: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None

        # No stacktrace found for following nodes
        expand_default_18: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_104, permute_103, permute_102, expand_default_18, False, scale = 0.0625);  permute_104 = permute_103 = permute_102 = expand_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_134: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_18[0];  _scaled_dot_product_efficient_attention_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_106: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        clone_78: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_273: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_78, [1, 128, 4096]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_274: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_273, [128, 4096]);  view_273 = None
        permute_107: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_39: "f32[128, 4096]" = torch.ops.aten.mm.default(view_274, permute_107);  view_274 = permute_107 = None
        view_275: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 4096]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_276: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_85, [128, 4096]);  add_85 = None
        permute_108: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_18: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg109_1, view_276, permute_108);  arg109_1 = view_276 = permute_108 = None
        view_277: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        pow_10: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_277, 3.0)
        mul_97: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_89: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_277, mul_97);  view_277 = mul_97 = None
        mul_98: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_9: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_90: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_96, add_90);  mul_96 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_278: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_99, [128, 16384]);  mul_99 = None
        permute_109: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_19: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg111_1, view_278, permute_109);  arg111_1 = view_278 = permute_109 = None
        view_279: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 4096]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_91: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_275, view_279);  view_275 = view_279 = None
        add_92: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_91, add_83);  add_91 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_41: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_20: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg117_1, [1, 1, 1]);  arg117_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_140: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_21: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_140, [1, 1, 64]);  unsqueeze_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_10: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_20, 1, repeat_21);  repeat_20 = repeat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_10 = torch.ops.aten.split.Tensor(gather_10, 32, -1);  gather_10 = None
        getitem_42: "f32[1, 128, 32]" = split_10[0]
        getitem_43: "f32[1, 128, 32]" = split_10[1];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_22: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_92, getitem_41);  getitem_41 = None
        add_93: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_100: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_10);  sub_22 = rsqrt_10 = None
        mul_101: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, arg112_1);  mul_100 = arg112_1 = None
        add_94: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_101, arg113_1);  mul_101 = arg113_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_280: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_94, [128, 4096])
        permute_110: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_40: "f32[128, 4096]" = torch.ops.aten.mm.default(view_280, permute_110);  view_280 = permute_110 = None
        view_281: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 4096]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_286: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_281, [1, 128, 16, 256]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_86: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_286, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_149: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_43, 2)
        unsqueeze_150: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 4);  unsqueeze_149 = None
        expand_84: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_150, [1, 128, 1, 32, 2]);  unsqueeze_150 = None
        clone_84: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_293: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_84, [1, 128, 1, 64]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_104: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_86, view_293);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_91: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_86, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_21: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_91);  slice_91 = None
        unsqueeze_151: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_21, 4);  neg_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_90: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_86, 3, 0, 9223372036854775807, 2);  slice_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_152: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_90, 4);  slice_90 = None
        cat_42: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_151, unsqueeze_152], -1);  unsqueeze_151 = unsqueeze_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_294: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_42, [1, 128, 16, 64]);  cat_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_147: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_42, 2)
        unsqueeze_148: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 4);  unsqueeze_147 = None
        expand_83: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_148, [1, 128, 1, 32, 2]);  unsqueeze_148 = None
        clone_83: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_292: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_83, [1, 128, 1, 64]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_105: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_294, view_292);  view_294 = view_292 = None
        add_96: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_87: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_286, 3, 64, 9223372036854775807);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_44: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_96, slice_87], -1);  add_96 = slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_115: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_44, [0, 2, 1, 3]);  cat_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_282: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_94, [128, 4096])
        permute_111: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_41: "f32[128, 4096]" = torch.ops.aten.mm.default(view_282, permute_111);  view_282 = permute_111 = None
        view_283: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 4096]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_287: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_283, [1, 128, 16, 256]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_84: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_287, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_143: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_43, 2);  getitem_43 = None
        unsqueeze_144: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 4);  unsqueeze_143 = None
        expand_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_144, [1, 128, 1, 32, 2]);  unsqueeze_144 = None
        clone_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_290: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_102: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_84, view_290);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_89: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_84, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_20: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_89);  slice_89 = None
        unsqueeze_145: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_20, 4);  neg_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_88: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_84, 3, 0, 9223372036854775807, 2);  slice_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_146: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_88, 4);  slice_88 = None
        cat_41: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_145, unsqueeze_146], -1);  unsqueeze_145 = unsqueeze_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_291: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_41, [1, 128, 16, 64]);  cat_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_141: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_42, 2);  getitem_42 = None
        unsqueeze_142: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 4);  unsqueeze_141 = None
        expand_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_142, [1, 128, 1, 32, 2]);  unsqueeze_142 = None
        clone_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_289: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_103: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_291, view_289);  view_291 = view_289 = None
        add_95: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_85: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_287, 3, 64, 9223372036854775807);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_43: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_95, slice_85], -1);  add_95 = slice_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_114: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_43, [0, 2, 1, 3]);  cat_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_284: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_94, [128, 4096])
        permute_112: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        mm_42: "f32[128, 4096]" = torch.ops.aten.mm.default(view_284, permute_112);  view_284 = permute_112 = None
        view_285: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 4096]);  mm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_288: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_285, [1, 128, 16, 256]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_113: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # No stacktrace found for following nodes
        expand_default_17: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_115, permute_114, permute_113, expand_default_17, False, scale = 0.0625);  permute_115 = permute_114 = permute_113 = expand_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_133: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_17[0];  _scaled_dot_product_efficient_attention_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_117: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        clone_86: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_301: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_86, [1, 128, 4096]);  clone_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_302: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_301, [128, 4096]);  view_301 = None
        permute_118: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_43: "f32[128, 4096]" = torch.ops.aten.mm.default(view_302, permute_118);  view_302 = permute_118 = None
        view_303: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 4096]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_304: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_94, [128, 4096]);  add_94 = None
        permute_119: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_20: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg120_1, view_304, permute_119);  arg120_1 = view_304 = permute_119 = None
        view_305: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        pow_11: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_107: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_98: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_305, mul_107);  view_305 = mul_107 = None
        mul_108: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
        tanh_10: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_99: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_106, add_99);  mul_106 = add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_306: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_109, [128, 16384]);  mul_109 = None
        permute_120: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_21: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg122_1, view_306, permute_120);  arg122_1 = view_306 = permute_120 = None
        view_307: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 4096]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_100: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_303, view_307);  view_303 = view_307 = None
        add_101: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_100, add_92);  add_100 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_45: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_22: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg128_1, [1, 1, 1]);  arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_153: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_23: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_153, [1, 1, 64]);  unsqueeze_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_11: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_22, 1, repeat_23);  repeat_22 = repeat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_11 = torch.ops.aten.split.Tensor(gather_11, 32, -1);  gather_11 = None
        getitem_46: "f32[1, 128, 32]" = split_11[0]
        getitem_47: "f32[1, 128, 32]" = split_11[1];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_24: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_101, getitem_45);  getitem_45 = None
        add_102: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_110: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = rsqrt_11 = None
        mul_111: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, arg123_1);  mul_110 = arg123_1 = None
        add_103: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_111, arg124_1);  mul_111 = arg124_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_308: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_103, [128, 4096])
        permute_121: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        mm_44: "f32[128, 4096]" = torch.ops.aten.mm.default(view_308, permute_121);  view_308 = permute_121 = None
        view_309: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 4096]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_314: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_309, [1, 128, 16, 256]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_94: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_314, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_162: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_47, 2)
        unsqueeze_163: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 4);  unsqueeze_162 = None
        expand_92: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_163, [1, 128, 1, 32, 2]);  unsqueeze_163 = None
        clone_92: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_321: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_92, [1, 128, 1, 64]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_114: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_94, view_321);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_99: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_94, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_23: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_99);  slice_99 = None
        unsqueeze_164: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_23, 4);  neg_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_98: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_94, 3, 0, 9223372036854775807, 2);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_165: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_98, 4);  slice_98 = None
        cat_46: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_164, unsqueeze_165], -1);  unsqueeze_164 = unsqueeze_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_322: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_46, [1, 128, 16, 64]);  cat_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_160: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_46, 2)
        unsqueeze_161: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 4);  unsqueeze_160 = None
        expand_91: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_161, [1, 128, 1, 32, 2]);  unsqueeze_161 = None
        clone_91: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_320: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_91, [1, 128, 1, 64]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_115: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_322, view_320);  view_322 = view_320 = None
        add_105: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_95: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_314, 3, 64, 9223372036854775807);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_48: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_105, slice_95], -1);  add_105 = slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_126: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_48, [0, 2, 1, 3]);  cat_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_310: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_103, [128, 4096])
        permute_122: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_45: "f32[128, 4096]" = torch.ops.aten.mm.default(view_310, permute_122);  view_310 = permute_122 = None
        view_311: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 4096]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_315: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_311, [1, 128, 16, 256]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_92: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_315, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_156: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_47, 2);  getitem_47 = None
        unsqueeze_157: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 4);  unsqueeze_156 = None
        expand_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_157, [1, 128, 1, 32, 2]);  unsqueeze_157 = None
        clone_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_318: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_112: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_92, view_318);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_97: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_92, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_22: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_97);  slice_97 = None
        unsqueeze_158: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_22, 4);  neg_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_96: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_92, 3, 0, 9223372036854775807, 2);  slice_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_159: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_96, 4);  slice_96 = None
        cat_45: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_158, unsqueeze_159], -1);  unsqueeze_158 = unsqueeze_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_319: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_45, [1, 128, 16, 64]);  cat_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_154: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_46, 2);  getitem_46 = None
        unsqueeze_155: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 4);  unsqueeze_154 = None
        expand_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_155, [1, 128, 1, 32, 2]);  unsqueeze_155 = None
        clone_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_317: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_113: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_319, view_317);  view_319 = view_317 = None
        add_104: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_93: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_315, 3, 64, 9223372036854775807);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_47: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_104, slice_93], -1);  add_104 = slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_125: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_47, [0, 2, 1, 3]);  cat_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_312: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_103, [128, 4096])
        permute_123: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_46: "f32[128, 4096]" = torch.ops.aten.mm.default(view_312, permute_123);  view_312 = permute_123 = None
        view_313: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 4096]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_316: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_313, [1, 128, 16, 256]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_124: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # No stacktrace found for following nodes
        expand_default_16: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_126, permute_125, permute_124, expand_default_16, False, scale = 0.0625);  permute_126 = permute_125 = permute_124 = expand_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_132: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_16[0];  _scaled_dot_product_efficient_attention_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_128: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None
        clone_94: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_329: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_94, [1, 128, 4096]);  clone_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_330: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_329, [128, 4096]);  view_329 = None
        permute_129: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_47: "f32[128, 4096]" = torch.ops.aten.mm.default(view_330, permute_129);  view_330 = permute_129 = None
        view_331: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 4096]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_332: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_103, [128, 4096]);  add_103 = None
        permute_130: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_22: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg131_1, view_332, permute_130);  arg131_1 = view_332 = permute_130 = None
        view_333: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_333, 0.5)
        pow_12: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_333, 3.0)
        mul_117: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_107: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_333, mul_117);  view_333 = mul_117 = None
        mul_118: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
        tanh_11: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_108: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_116, add_108);  mul_116 = add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_334: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_119, [128, 16384]);  mul_119 = None
        permute_131: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_23: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg133_1, view_334, permute_131);  arg133_1 = view_334 = permute_131 = None
        view_335: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 4096]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_109: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_331, view_335);  view_331 = view_335 = None
        add_110: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_109, add_101);  add_109 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_49: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_24: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg139_1, [1, 1, 1]);  arg139_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_166: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_25: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_166, [1, 1, 64]);  unsqueeze_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_12: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_24, 1, repeat_25);  repeat_24 = repeat_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_12 = torch.ops.aten.split.Tensor(gather_12, 32, -1);  gather_12 = None
        getitem_50: "f32[1, 128, 32]" = split_12[0]
        getitem_51: "f32[1, 128, 32]" = split_12[1];  split_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_26: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_110, getitem_49);  getitem_49 = None
        add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_12);  sub_26 = rsqrt_12 = None
        mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, arg134_1);  mul_120 = arg134_1 = None
        add_112: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_121, arg135_1);  mul_121 = arg135_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_336: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_112, [128, 4096])
        permute_132: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_48: "f32[128, 4096]" = torch.ops.aten.mm.default(view_336, permute_132);  view_336 = permute_132 = None
        view_337: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 4096]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_342: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_337, [1, 128, 16, 256]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_102: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_342, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_175: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_51, 2)
        unsqueeze_176: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 4);  unsqueeze_175 = None
        expand_100: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_176, [1, 128, 1, 32, 2]);  unsqueeze_176 = None
        clone_100: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_349: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_100, [1, 128, 1, 64]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_124: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_102, view_349);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_107: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_102, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_25: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_107);  slice_107 = None
        unsqueeze_177: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_25, 4);  neg_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_106: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_102, 3, 0, 9223372036854775807, 2);  slice_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_178: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_106, 4);  slice_106 = None
        cat_50: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_177, unsqueeze_178], -1);  unsqueeze_177 = unsqueeze_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_350: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_50, [1, 128, 16, 64]);  cat_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_173: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_50, 2)
        unsqueeze_174: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 4);  unsqueeze_173 = None
        expand_99: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_174, [1, 128, 1, 32, 2]);  unsqueeze_174 = None
        clone_99: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_348: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_99, [1, 128, 1, 64]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_125: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_350, view_348);  view_350 = view_348 = None
        add_114: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_103: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_342, 3, 64, 9223372036854775807);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_52: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_114, slice_103], -1);  add_114 = slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_137: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_52, [0, 2, 1, 3]);  cat_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_338: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_112, [128, 4096])
        permute_133: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_49: "f32[128, 4096]" = torch.ops.aten.mm.default(view_338, permute_133);  view_338 = permute_133 = None
        view_339: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 4096]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_343: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_339, [1, 128, 16, 256]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_100: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_343, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_169: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_51, 2);  getitem_51 = None
        unsqueeze_170: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 4);  unsqueeze_169 = None
        expand_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_170, [1, 128, 1, 32, 2]);  unsqueeze_170 = None
        clone_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_346: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_122: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_100, view_346);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_105: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_100, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_24: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_105);  slice_105 = None
        unsqueeze_171: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_24, 4);  neg_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_104: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_100, 3, 0, 9223372036854775807, 2);  slice_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_172: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_104, 4);  slice_104 = None
        cat_49: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_171, unsqueeze_172], -1);  unsqueeze_171 = unsqueeze_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_347: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_49, [1, 128, 16, 64]);  cat_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_167: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_50, 2);  getitem_50 = None
        unsqueeze_168: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
        expand_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_168, [1, 128, 1, 32, 2]);  unsqueeze_168 = None
        clone_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_345: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_123: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_347, view_345);  view_347 = view_345 = None
        add_113: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_101: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_343, 3, 64, 9223372036854775807);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_51: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_113, slice_101], -1);  add_113 = slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_136: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_51, [0, 2, 1, 3]);  cat_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_340: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_112, [128, 4096])
        permute_134: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        mm_50: "f32[128, 4096]" = torch.ops.aten.mm.default(view_340, permute_134);  view_340 = permute_134 = None
        view_341: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 4096]);  mm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_344: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_341, [1, 128, 16, 256]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_135: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # No stacktrace found for following nodes
        expand_default_15: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_137, permute_136, permute_135, expand_default_15, False, scale = 0.0625);  permute_137 = permute_136 = permute_135 = expand_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_131: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_139: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3]);  getitem_131 = None
        clone_102: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_357: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_102, [1, 128, 4096]);  clone_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_358: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_357, [128, 4096]);  view_357 = None
        permute_140: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_51: "f32[128, 4096]" = torch.ops.aten.mm.default(view_358, permute_140);  view_358 = permute_140 = None
        view_359: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 4096]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_360: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_112, [128, 4096]);  add_112 = None
        permute_141: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_24: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg142_1, view_360, permute_141);  arg142_1 = view_360 = permute_141 = None
        view_361: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_126: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_361, 0.5)
        pow_13: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_361, 3.0)
        mul_127: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_116: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_361, mul_127);  view_361 = mul_127 = None
        mul_128: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_116, 0.7978845608028654);  add_116 = None
        tanh_12: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_117: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_129: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_126, add_117);  mul_126 = add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_362: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_129, [128, 16384]);  mul_129 = None
        permute_142: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_25: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg144_1, view_362, permute_142);  arg144_1 = view_362 = permute_142 = None
        view_363: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 4096]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_118: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_359, view_363);  view_359 = view_363 = None
        add_119: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_118, add_110);  add_118 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 128, 1]" = var_mean_13[0]
        getitem_53: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_26: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg150_1, [1, 1, 1]);  arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_179: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_27: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_179, [1, 1, 64]);  unsqueeze_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_13: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_26, 1, repeat_27);  repeat_26 = repeat_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_13 = torch.ops.aten.split.Tensor(gather_13, 32, -1);  gather_13 = None
        getitem_54: "f32[1, 128, 32]" = split_13[0]
        getitem_55: "f32[1, 128, 32]" = split_13[1];  split_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_28: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_119, getitem_53);  getitem_53 = None
        add_120: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_130: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = rsqrt_13 = None
        mul_131: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, arg145_1);  mul_130 = arg145_1 = None
        add_121: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_131, arg146_1);  mul_131 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_364: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096])
        permute_143: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_52: "f32[128, 4096]" = torch.ops.aten.mm.default(view_364, permute_143);  view_364 = permute_143 = None
        view_365: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 4096]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_370: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_365, [1, 128, 16, 256]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_110: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_370, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_188: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_55, 2)
        unsqueeze_189: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 4);  unsqueeze_188 = None
        expand_108: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_189, [1, 128, 1, 32, 2]);  unsqueeze_189 = None
        clone_108: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_377: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_108, [1, 128, 1, 64]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_134: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_110, view_377);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_115: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_110, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_27: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_115);  slice_115 = None
        unsqueeze_190: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_27, 4);  neg_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_114: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_110, 3, 0, 9223372036854775807, 2);  slice_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_191: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_114, 4);  slice_114 = None
        cat_54: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_190, unsqueeze_191], -1);  unsqueeze_190 = unsqueeze_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_378: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_54, [1, 128, 16, 64]);  cat_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_186: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_54, 2)
        unsqueeze_187: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 4);  unsqueeze_186 = None
        expand_107: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_187, [1, 128, 1, 32, 2]);  unsqueeze_187 = None
        clone_107: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_376: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_107, [1, 128, 1, 64]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_135: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_378, view_376);  view_378 = view_376 = None
        add_123: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_111: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_370, 3, 64, 9223372036854775807);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_56: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_123, slice_111], -1);  add_123 = slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_148: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_56, [0, 2, 1, 3]);  cat_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_366: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096])
        permute_144: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        mm_53: "f32[128, 4096]" = torch.ops.aten.mm.default(view_366, permute_144);  view_366 = permute_144 = None
        view_367: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 4096]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_371: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_367, [1, 128, 16, 256]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_108: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_371, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_182: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_55, 2);  getitem_55 = None
        unsqueeze_183: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 4);  unsqueeze_182 = None
        expand_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_183, [1, 128, 1, 32, 2]);  unsqueeze_183 = None
        clone_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_374: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_132: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_108, view_374);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_113: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_108, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_26: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_113);  slice_113 = None
        unsqueeze_184: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_26, 4);  neg_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_112: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_108, 3, 0, 9223372036854775807, 2);  slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_185: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_112, 4);  slice_112 = None
        cat_53: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_184, unsqueeze_185], -1);  unsqueeze_184 = unsqueeze_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_375: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_53, [1, 128, 16, 64]);  cat_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_180: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_54, 2);  getitem_54 = None
        unsqueeze_181: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
        expand_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_181, [1, 128, 1, 32, 2]);  unsqueeze_181 = None
        clone_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_373: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_133: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_375, view_373);  view_375 = view_373 = None
        add_122: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_109: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_371, 3, 64, 9223372036854775807);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_55: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_122, slice_109], -1);  add_122 = slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_147: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_55, [0, 2, 1, 3]);  cat_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_368: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096])
        permute_145: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_54: "f32[128, 4096]" = torch.ops.aten.mm.default(view_368, permute_145);  view_368 = permute_145 = None
        view_369: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 4096]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_372: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_369, [1, 128, 16, 256]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_146: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # No stacktrace found for following nodes
        expand_default_14: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_148, permute_147, permute_146, expand_default_14, False, scale = 0.0625);  permute_148 = permute_147 = permute_146 = expand_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_130: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_150: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3]);  getitem_130 = None
        clone_110: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_385: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_110, [1, 128, 4096]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_386: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_385, [128, 4096]);  view_385 = None
        permute_151: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_55: "f32[128, 4096]" = torch.ops.aten.mm.default(view_386, permute_151);  view_386 = permute_151 = None
        view_387: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 4096]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_388: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096]);  add_121 = None
        permute_152: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_26: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg153_1, view_388, permute_152);  arg153_1 = view_388 = permute_152 = None
        view_389: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_136: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        pow_14: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
        mul_137: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_125: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_389, mul_137);  view_389 = mul_137 = None
        mul_138: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_125, 0.7978845608028654);  add_125 = None
        tanh_13: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
        add_126: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_139: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_136, add_126);  mul_136 = add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_390: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_139, [128, 16384]);  mul_139 = None
        permute_153: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_27: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg155_1, view_390, permute_153);  arg155_1 = view_390 = permute_153 = None
        view_391: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 4096]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_127: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_387, view_391);  view_387 = view_391 = None
        add_128: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_127, add_119);  add_127 = add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 128, 1]" = var_mean_14[0]
        getitem_57: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_28: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg161_1, [1, 1, 1]);  arg161_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_192: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_29: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_192, [1, 1, 64]);  unsqueeze_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_14: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_28, 1, repeat_29);  repeat_28 = repeat_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_14 = torch.ops.aten.split.Tensor(gather_14, 32, -1);  gather_14 = None
        getitem_58: "f32[1, 128, 32]" = split_14[0]
        getitem_59: "f32[1, 128, 32]" = split_14[1];  split_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_30: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_128, getitem_57);  getitem_57 = None
        add_129: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_140: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_14);  sub_30 = rsqrt_14 = None
        mul_141: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, arg156_1);  mul_140 = arg156_1 = None
        add_130: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_141, arg157_1);  mul_141 = arg157_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_392: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_130, [128, 4096])
        permute_154: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        mm_56: "f32[128, 4096]" = torch.ops.aten.mm.default(view_392, permute_154);  view_392 = permute_154 = None
        view_393: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 4096]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_398: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_393, [1, 128, 16, 256]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_118: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_398, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_201: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_59, 2)
        unsqueeze_202: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 4);  unsqueeze_201 = None
        expand_116: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_202, [1, 128, 1, 32, 2]);  unsqueeze_202 = None
        clone_116: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_405: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_116, [1, 128, 1, 64]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_144: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_118, view_405);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_123: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_118, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_29: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_123);  slice_123 = None
        unsqueeze_203: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_29, 4);  neg_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_122: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_118, 3, 0, 9223372036854775807, 2);  slice_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_204: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_122, 4);  slice_122 = None
        cat_58: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_203, unsqueeze_204], -1);  unsqueeze_203 = unsqueeze_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_406: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_58, [1, 128, 16, 64]);  cat_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_199: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_58, 2)
        unsqueeze_200: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 4);  unsqueeze_199 = None
        expand_115: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_200, [1, 128, 1, 32, 2]);  unsqueeze_200 = None
        clone_115: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_404: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_115, [1, 128, 1, 64]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_145: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_406, view_404);  view_406 = view_404 = None
        add_132: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_119: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_398, 3, 64, 9223372036854775807);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_60: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_132, slice_119], -1);  add_132 = slice_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_159: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_60, [0, 2, 1, 3]);  cat_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_394: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_130, [128, 4096])
        permute_155: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_57: "f32[128, 4096]" = torch.ops.aten.mm.default(view_394, permute_155);  view_394 = permute_155 = None
        view_395: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 4096]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_399: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_395, [1, 128, 16, 256]);  view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_116: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_399, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_195: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_59, 2);  getitem_59 = None
        unsqueeze_196: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 4);  unsqueeze_195 = None
        expand_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_196, [1, 128, 1, 32, 2]);  unsqueeze_196 = None
        clone_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_402: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_142: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_116, view_402);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_121: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_116, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_28: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_121);  slice_121 = None
        unsqueeze_197: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_28, 4);  neg_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_120: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_116, 3, 0, 9223372036854775807, 2);  slice_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_198: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_120, 4);  slice_120 = None
        cat_57: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_197, unsqueeze_198], -1);  unsqueeze_197 = unsqueeze_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_403: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_57, [1, 128, 16, 64]);  cat_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_193: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_58, 2);  getitem_58 = None
        unsqueeze_194: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 4);  unsqueeze_193 = None
        expand_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_194, [1, 128, 1, 32, 2]);  unsqueeze_194 = None
        clone_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_401: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_143: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_403, view_401);  view_403 = view_401 = None
        add_131: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_117: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_399, 3, 64, 9223372036854775807);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_59: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_131, slice_117], -1);  add_131 = slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_158: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_59, [0, 2, 1, 3]);  cat_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_396: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_130, [128, 4096])
        permute_156: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_58: "f32[128, 4096]" = torch.ops.aten.mm.default(view_396, permute_156);  view_396 = permute_156 = None
        view_397: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 4096]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_400: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_397, [1, 128, 16, 256]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_157: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # No stacktrace found for following nodes
        expand_default_13: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_159, permute_158, permute_157, expand_default_13, False, scale = 0.0625);  permute_159 = permute_158 = permute_157 = expand_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_129: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_161: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1, 3]);  getitem_129 = None
        clone_118: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_413: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_118, [1, 128, 4096]);  clone_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_414: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_413, [128, 4096]);  view_413 = None
        permute_162: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_59: "f32[128, 4096]" = torch.ops.aten.mm.default(view_414, permute_162);  view_414 = permute_162 = None
        view_415: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 4096]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_416: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_130, [128, 4096]);  add_130 = None
        permute_163: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_28: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg164_1, view_416, permute_163);  arg164_1 = view_416 = permute_163 = None
        view_417: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_146: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
        pow_15: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 3.0)
        mul_147: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_134: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_417, mul_147);  view_417 = mul_147 = None
        mul_148: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
        tanh_14: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
        add_135: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_149: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_146, add_135);  mul_146 = add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_418: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_149, [128, 16384]);  mul_149 = None
        permute_164: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_29: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg166_1, view_418, permute_164);  arg166_1 = view_418 = permute_164 = None
        view_419: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 4096]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_136: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_415, view_419);  view_415 = view_419 = None
        add_137: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_136, add_128);  add_136 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 128, 1]" = var_mean_15[0]
        getitem_61: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_30: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg172_1, [1, 1, 1]);  arg172_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_205: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_31: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_205, [1, 1, 64]);  unsqueeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_15: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_30, 1, repeat_31);  repeat_30 = repeat_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_15 = torch.ops.aten.split.Tensor(gather_15, 32, -1);  gather_15 = None
        getitem_62: "f32[1, 128, 32]" = split_15[0]
        getitem_63: "f32[1, 128, 32]" = split_15[1];  split_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_32: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_137, getitem_61);  getitem_61 = None
        add_138: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_150: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = rsqrt_15 = None
        mul_151: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, arg167_1);  mul_150 = arg167_1 = None
        add_139: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_151, arg168_1);  mul_151 = arg168_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_420: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_139, [128, 4096])
        permute_165: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_60: "f32[128, 4096]" = torch.ops.aten.mm.default(view_420, permute_165);  view_420 = permute_165 = None
        view_421: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 4096]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_426: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_421, [1, 128, 16, 256]);  view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_126: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_426, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_214: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_63, 2)
        unsqueeze_215: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 4);  unsqueeze_214 = None
        expand_124: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_215, [1, 128, 1, 32, 2]);  unsqueeze_215 = None
        clone_124: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_433: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_124, [1, 128, 1, 64]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_154: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_126, view_433);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_131: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_126, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_31: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_131);  slice_131 = None
        unsqueeze_216: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_31, 4);  neg_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_130: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_126, 3, 0, 9223372036854775807, 2);  slice_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_217: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_130, 4);  slice_130 = None
        cat_62: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_216, unsqueeze_217], -1);  unsqueeze_216 = unsqueeze_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_434: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_62, [1, 128, 16, 64]);  cat_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_212: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_62, 2)
        unsqueeze_213: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 4);  unsqueeze_212 = None
        expand_123: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_213, [1, 128, 1, 32, 2]);  unsqueeze_213 = None
        clone_123: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_432: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_123, [1, 128, 1, 64]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_155: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_434, view_432);  view_434 = view_432 = None
        add_141: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_127: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_426, 3, 64, 9223372036854775807);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_64: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_141, slice_127], -1);  add_141 = slice_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_170: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_64, [0, 2, 1, 3]);  cat_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_422: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_139, [128, 4096])
        permute_166: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        mm_61: "f32[128, 4096]" = torch.ops.aten.mm.default(view_422, permute_166);  view_422 = permute_166 = None
        view_423: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 4096]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_427: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_423, [1, 128, 16, 256]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_124: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_427, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_208: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_63, 2);  getitem_63 = None
        unsqueeze_209: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 4);  unsqueeze_208 = None
        expand_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_209, [1, 128, 1, 32, 2]);  unsqueeze_209 = None
        clone_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_430: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_152: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_124, view_430);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_129: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_124, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_30: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_129);  slice_129 = None
        unsqueeze_210: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_30, 4);  neg_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_128: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_124, 3, 0, 9223372036854775807, 2);  slice_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_211: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_128, 4);  slice_128 = None
        cat_61: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_210, unsqueeze_211], -1);  unsqueeze_210 = unsqueeze_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_431: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_61, [1, 128, 16, 64]);  cat_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_206: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_62, 2);  getitem_62 = None
        unsqueeze_207: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 4);  unsqueeze_206 = None
        expand_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_207, [1, 128, 1, 32, 2]);  unsqueeze_207 = None
        clone_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_429: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_153: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_431, view_429);  view_431 = view_429 = None
        add_140: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_125: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_427, 3, 64, 9223372036854775807);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_63: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_140, slice_125], -1);  add_140 = slice_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_169: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_63, [0, 2, 1, 3]);  cat_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_424: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_139, [128, 4096])
        permute_167: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_62: "f32[128, 4096]" = torch.ops.aten.mm.default(view_424, permute_167);  view_424 = permute_167 = None
        view_425: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 4096]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_428: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_425, [1, 128, 16, 256]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_168: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None

        # No stacktrace found for following nodes
        expand_default_12: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_170, permute_169, permute_168, expand_default_12, False, scale = 0.0625);  permute_170 = permute_169 = permute_168 = expand_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_128: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_172: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None
        clone_126: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_441: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_126, [1, 128, 4096]);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_442: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_441, [128, 4096]);  view_441 = None
        permute_173: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_63: "f32[128, 4096]" = torch.ops.aten.mm.default(view_442, permute_173);  view_442 = permute_173 = None
        view_443: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 4096]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_444: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_139, [128, 4096]);  add_139 = None
        permute_174: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_30: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg175_1, view_444, permute_174);  arg175_1 = view_444 = permute_174 = None
        view_445: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_445, 0.5)
        pow_16: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_445, 3.0)
        mul_157: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_143: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_445, mul_157);  view_445 = mul_157 = None
        mul_158: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_143, 0.7978845608028654);  add_143 = None
        tanh_15: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_144: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_159: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_156, add_144);  mul_156 = add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_446: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_159, [128, 16384]);  mul_159 = None
        permute_175: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_31: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg177_1, view_446, permute_175);  arg177_1 = view_446 = permute_175 = None
        view_447: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 4096]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_145: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_443, view_447);  view_443 = view_447 = None
        add_146: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_145, add_137);  add_145 = add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 128, 1]" = var_mean_16[0]
        getitem_65: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_32: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg183_1, [1, 1, 1]);  arg183_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_218: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_33: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_218, [1, 1, 64]);  unsqueeze_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_16: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_32, 1, repeat_33);  repeat_32 = repeat_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_16 = torch.ops.aten.split.Tensor(gather_16, 32, -1);  gather_16 = None
        getitem_66: "f32[1, 128, 32]" = split_16[0]
        getitem_67: "f32[1, 128, 32]" = split_16[1];  split_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_34: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_146, getitem_65);  getitem_65 = None
        add_147: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_160: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_16);  sub_34 = rsqrt_16 = None
        mul_161: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, arg178_1);  mul_160 = arg178_1 = None
        add_148: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_161, arg179_1);  mul_161 = arg179_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_448: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_148, [128, 4096])
        permute_176: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_64: "f32[128, 4096]" = torch.ops.aten.mm.default(view_448, permute_176);  view_448 = permute_176 = None
        view_449: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 4096]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_454: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_449, [1, 128, 16, 256]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_134: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_454, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_227: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_67, 2)
        unsqueeze_228: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 4);  unsqueeze_227 = None
        expand_132: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_228, [1, 128, 1, 32, 2]);  unsqueeze_228 = None
        clone_132: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
        view_461: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_132, [1, 128, 1, 64]);  clone_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_164: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_134, view_461);  view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_139: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_134, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_33: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_139);  slice_139 = None
        unsqueeze_229: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_33, 4);  neg_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_138: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_134, 3, 0, 9223372036854775807, 2);  slice_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_230: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_138, 4);  slice_138 = None
        cat_66: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_229, unsqueeze_230], -1);  unsqueeze_229 = unsqueeze_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_462: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_66, [1, 128, 16, 64]);  cat_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_225: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_66, 2)
        unsqueeze_226: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 4);  unsqueeze_225 = None
        expand_131: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_226, [1, 128, 1, 32, 2]);  unsqueeze_226 = None
        clone_131: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_460: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_131, [1, 128, 1, 64]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_165: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_462, view_460);  view_462 = view_460 = None
        add_150: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_135: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_454, 3, 64, 9223372036854775807);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_68: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_150, slice_135], -1);  add_150 = slice_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_181: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_68, [0, 2, 1, 3]);  cat_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_450: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_148, [128, 4096])
        permute_177: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_65: "f32[128, 4096]" = torch.ops.aten.mm.default(view_450, permute_177);  view_450 = permute_177 = None
        view_451: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 4096]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_455: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_451, [1, 128, 16, 256]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_132: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_455, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_221: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_67, 2);  getitem_67 = None
        unsqueeze_222: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 4);  unsqueeze_221 = None
        expand_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_222, [1, 128, 1, 32, 2]);  unsqueeze_222 = None
        clone_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_458: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_162: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_132, view_458);  view_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_137: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_132, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_32: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_137);  slice_137 = None
        unsqueeze_223: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_32, 4);  neg_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_136: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_132, 3, 0, 9223372036854775807, 2);  slice_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_224: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_136, 4);  slice_136 = None
        cat_65: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_223, unsqueeze_224], -1);  unsqueeze_223 = unsqueeze_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_459: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_65, [1, 128, 16, 64]);  cat_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_219: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_66, 2);  getitem_66 = None
        unsqueeze_220: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 4);  unsqueeze_219 = None
        expand_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_220, [1, 128, 1, 32, 2]);  unsqueeze_220 = None
        clone_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_457: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_163: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_459, view_457);  view_459 = view_457 = None
        add_149: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_133: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_455, 3, 64, 9223372036854775807);  view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_67: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_149, slice_133], -1);  add_149 = slice_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_180: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_67, [0, 2, 1, 3]);  cat_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_452: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_148, [128, 4096])
        permute_178: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_66: "f32[128, 4096]" = torch.ops.aten.mm.default(view_452, permute_178);  view_452 = permute_178 = None
        view_453: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 4096]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_456: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_453, [1, 128, 16, 256]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_179: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None

        # No stacktrace found for following nodes
        expand_default_11: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_181, permute_180, permute_179, expand_default_11, False, scale = 0.0625);  permute_181 = permute_180 = permute_179 = expand_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_127: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_183: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None
        clone_134: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_469: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_134, [1, 128, 4096]);  clone_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_470: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_469, [128, 4096]);  view_469 = None
        permute_184: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_67: "f32[128, 4096]" = torch.ops.aten.mm.default(view_470, permute_184);  view_470 = permute_184 = None
        view_471: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 4096]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_472: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_148, [128, 4096]);  add_148 = None
        permute_185: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_32: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg186_1, view_472, permute_185);  arg186_1 = view_472 = permute_185 = None
        view_473: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_166: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_473, 0.5)
        pow_17: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_473, 3.0)
        mul_167: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_152: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_473, mul_167);  view_473 = mul_167 = None
        mul_168: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_152, 0.7978845608028654);  add_152 = None
        tanh_16: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
        add_153: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_169: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_166, add_153);  mul_166 = add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_474: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_169, [128, 16384]);  mul_169 = None
        permute_186: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_33: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg188_1, view_474, permute_186);  arg188_1 = view_474 = permute_186 = None
        view_475: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 4096]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_154: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_471, view_475);  view_471 = view_475 = None
        add_155: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_154, add_146);  add_154 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
        getitem_68: "f32[1, 128, 1]" = var_mean_17[0]
        getitem_69: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_34: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg194_1, [1, 1, 1]);  arg194_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_231: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_35: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_231, [1, 1, 64]);  unsqueeze_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_17: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_34, 1, repeat_35);  repeat_34 = repeat_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_17 = torch.ops.aten.split.Tensor(gather_17, 32, -1);  gather_17 = None
        getitem_70: "f32[1, 128, 32]" = split_17[0]
        getitem_71: "f32[1, 128, 32]" = split_17[1];  split_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_36: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_155, getitem_69);  getitem_69 = None
        add_156: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_170: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = rsqrt_17 = None
        mul_171: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, arg189_1);  mul_170 = arg189_1 = None
        add_157: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_171, arg190_1);  mul_171 = arg190_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_476: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_157, [128, 4096])
        permute_187: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_68: "f32[128, 4096]" = torch.ops.aten.mm.default(view_476, permute_187);  view_476 = permute_187 = None
        view_477: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 4096]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_482: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_477, [1, 128, 16, 256]);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_142: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_482, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_240: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_71, 2)
        unsqueeze_241: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 4);  unsqueeze_240 = None
        expand_140: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_241, [1, 128, 1, 32, 2]);  unsqueeze_241 = None
        clone_140: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_489: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_140, [1, 128, 1, 64]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_174: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_142, view_489);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_147: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_142, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_35: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_147);  slice_147 = None
        unsqueeze_242: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_35, 4);  neg_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_146: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_142, 3, 0, 9223372036854775807, 2);  slice_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_243: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_146, 4);  slice_146 = None
        cat_70: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_242, unsqueeze_243], -1);  unsqueeze_242 = unsqueeze_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_490: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_70, [1, 128, 16, 64]);  cat_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_238: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_70, 2)
        unsqueeze_239: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 4);  unsqueeze_238 = None
        expand_139: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_239, [1, 128, 1, 32, 2]);  unsqueeze_239 = None
        clone_139: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_488: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_139, [1, 128, 1, 64]);  clone_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_175: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_490, view_488);  view_490 = view_488 = None
        add_159: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_143: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_482, 3, 64, 9223372036854775807);  view_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_72: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_159, slice_143], -1);  add_159 = slice_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_192: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_72, [0, 2, 1, 3]);  cat_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_478: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_157, [128, 4096])
        permute_188: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        mm_69: "f32[128, 4096]" = torch.ops.aten.mm.default(view_478, permute_188);  view_478 = permute_188 = None
        view_479: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 4096]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_483: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_479, [1, 128, 16, 256]);  view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_140: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_483, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_234: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_71, 2);  getitem_71 = None
        unsqueeze_235: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 4);  unsqueeze_234 = None
        expand_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_235, [1, 128, 1, 32, 2]);  unsqueeze_235 = None
        clone_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_486: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_172: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_140, view_486);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_145: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_140, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_34: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_145);  slice_145 = None
        unsqueeze_236: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_34, 4);  neg_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_144: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_140, 3, 0, 9223372036854775807, 2);  slice_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_237: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_144, 4);  slice_144 = None
        cat_69: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_236, unsqueeze_237], -1);  unsqueeze_236 = unsqueeze_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_487: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_69, [1, 128, 16, 64]);  cat_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_232: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_70, 2);  getitem_70 = None
        unsqueeze_233: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 4);  unsqueeze_232 = None
        expand_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_233, [1, 128, 1, 32, 2]);  unsqueeze_233 = None
        clone_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_485: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_173: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_487, view_485);  view_487 = view_485 = None
        add_158: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_141: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_483, 3, 64, 9223372036854775807);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_71: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_158, slice_141], -1);  add_158 = slice_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_191: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_71, [0, 2, 1, 3]);  cat_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_480: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_157, [128, 4096])
        permute_189: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        mm_70: "f32[128, 4096]" = torch.ops.aten.mm.default(view_480, permute_189);  view_480 = permute_189 = None
        view_481: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 4096]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_484: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_481, [1, 128, 16, 256]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_190: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None

        # No stacktrace found for following nodes
        expand_default_10: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_192, permute_191, permute_190, expand_default_10, False, scale = 0.0625);  permute_192 = permute_191 = permute_190 = expand_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_126: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_194: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        clone_142: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_497: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_142, [1, 128, 4096]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_498: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_497, [128, 4096]);  view_497 = None
        permute_195: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_71: "f32[128, 4096]" = torch.ops.aten.mm.default(view_498, permute_195);  view_498 = permute_195 = None
        view_499: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 4096]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_500: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_157, [128, 4096]);  add_157 = None
        permute_196: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_34: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg197_1, view_500, permute_196);  arg197_1 = view_500 = permute_196 = None
        view_501: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_176: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        pow_18: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_501, 3.0)
        mul_177: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_161: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_501, mul_177);  view_501 = mul_177 = None
        mul_178: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_17: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_162: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_179: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_176, add_162);  mul_176 = add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_502: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_179, [128, 16384]);  mul_179 = None
        permute_197: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_35: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg199_1, view_502, permute_197);  arg199_1 = view_502 = permute_197 = None
        view_503: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 4096]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_163: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_499, view_503);  view_499 = view_503 = None
        add_164: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_163, add_155);  add_163 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 128, 1]" = var_mean_18[0]
        getitem_73: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_36: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg205_1, [1, 1, 1]);  arg205_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_244: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_37: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_244, [1, 1, 64]);  unsqueeze_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_18: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_36, 1, repeat_37);  repeat_36 = repeat_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_18 = torch.ops.aten.split.Tensor(gather_18, 32, -1);  gather_18 = None
        getitem_74: "f32[1, 128, 32]" = split_18[0]
        getitem_75: "f32[1, 128, 32]" = split_18[1];  split_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_38: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_164, getitem_73);  getitem_73 = None
        add_165: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_180: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_18);  sub_38 = rsqrt_18 = None
        mul_181: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, arg200_1);  mul_180 = arg200_1 = None
        add_166: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_181, arg201_1);  mul_181 = arg201_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_504: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_166, [128, 4096])
        permute_198: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        mm_72: "f32[128, 4096]" = torch.ops.aten.mm.default(view_504, permute_198);  view_504 = permute_198 = None
        view_505: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 4096]);  mm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_510: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_505, [1, 128, 16, 256]);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_150: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_510, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_253: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_75, 2)
        unsqueeze_254: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 4);  unsqueeze_253 = None
        expand_148: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_254, [1, 128, 1, 32, 2]);  unsqueeze_254 = None
        clone_148: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_517: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_148, [1, 128, 1, 64]);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_184: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_150, view_517);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_155: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_150, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_37: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_155);  slice_155 = None
        unsqueeze_255: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_37, 4);  neg_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_154: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_150, 3, 0, 9223372036854775807, 2);  slice_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_256: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_154, 4);  slice_154 = None
        cat_74: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_255, unsqueeze_256], -1);  unsqueeze_255 = unsqueeze_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_518: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_74, [1, 128, 16, 64]);  cat_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_251: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_74, 2)
        unsqueeze_252: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
        expand_147: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_252, [1, 128, 1, 32, 2]);  unsqueeze_252 = None
        clone_147: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_516: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_147, [1, 128, 1, 64]);  clone_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_185: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_518, view_516);  view_518 = view_516 = None
        add_168: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_151: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_510, 3, 64, 9223372036854775807);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_76: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_168, slice_151], -1);  add_168 = slice_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_203: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_76, [0, 2, 1, 3]);  cat_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_506: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_166, [128, 4096])
        permute_199: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        mm_73: "f32[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_199);  view_506 = permute_199 = None
        view_507: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_73, [1, 128, 4096]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_511: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_507, [1, 128, 16, 256]);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_148: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_511, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_247: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_75, 2);  getitem_75 = None
        unsqueeze_248: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 4);  unsqueeze_247 = None
        expand_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_248, [1, 128, 1, 32, 2]);  unsqueeze_248 = None
        clone_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_514: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_182: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_148, view_514);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_153: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_148, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_36: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_153);  slice_153 = None
        unsqueeze_249: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_36, 4);  neg_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_152: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_148, 3, 0, 9223372036854775807, 2);  slice_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_250: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_152, 4);  slice_152 = None
        cat_73: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_249, unsqueeze_250], -1);  unsqueeze_249 = unsqueeze_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_515: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_73, [1, 128, 16, 64]);  cat_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_245: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_74, 2);  getitem_74 = None
        unsqueeze_246: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 4);  unsqueeze_245 = None
        expand_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_246, [1, 128, 1, 32, 2]);  unsqueeze_246 = None
        clone_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_513: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_183: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_515, view_513);  view_515 = view_513 = None
        add_167: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_149: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_511, 3, 64, 9223372036854775807);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_75: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_167, slice_149], -1);  add_167 = slice_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_202: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_75, [0, 2, 1, 3]);  cat_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_508: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_166, [128, 4096])
        permute_200: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        mm_74: "f32[128, 4096]" = torch.ops.aten.mm.default(view_508, permute_200);  view_508 = permute_200 = None
        view_509: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_74, [1, 128, 4096]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_512: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_509, [1, 128, 16, 256]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_201: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # No stacktrace found for following nodes
        expand_default_9: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_203, permute_202, permute_201, expand_default_9, False, scale = 0.0625);  permute_203 = permute_202 = permute_201 = expand_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_125: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_205: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_125, [0, 2, 1, 3]);  getitem_125 = None
        clone_150: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_525: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_150, [1, 128, 4096]);  clone_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_526: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_525, [128, 4096]);  view_525 = None
        permute_206: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        mm_75: "f32[128, 4096]" = torch.ops.aten.mm.default(view_526, permute_206);  view_526 = permute_206 = None
        view_527: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_75, [1, 128, 4096]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_528: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_166, [128, 4096]);  add_166 = None
        permute_207: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_36: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg208_1, view_528, permute_207);  arg208_1 = view_528 = permute_207 = None
        view_529: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_186: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        pow_19: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_529, 3.0)
        mul_187: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_170: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_529, mul_187);  view_529 = mul_187 = None
        mul_188: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_170, 0.7978845608028654);  add_170 = None
        tanh_18: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_171: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_189: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_186, add_171);  mul_186 = add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_530: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_189, [128, 16384]);  mul_189 = None
        permute_208: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_37: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg210_1, view_530, permute_208);  arg210_1 = view_530 = permute_208 = None
        view_531: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 4096]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_172: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_527, view_531);  view_527 = view_531 = None
        add_173: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_172, add_164);  add_172 = add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 128, 1]" = var_mean_19[0]
        getitem_77: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_38: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg216_1, [1, 1, 1]);  arg216_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_257: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_39: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_257, [1, 1, 64]);  unsqueeze_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_19: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_38, 1, repeat_39);  repeat_38 = repeat_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_19 = torch.ops.aten.split.Tensor(gather_19, 32, -1);  gather_19 = None
        getitem_78: "f32[1, 128, 32]" = split_19[0]
        getitem_79: "f32[1, 128, 32]" = split_19[1];  split_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_40: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_173, getitem_77);  getitem_77 = None
        add_174: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = rsqrt_19 = None
        mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, arg211_1);  mul_190 = arg211_1 = None
        add_175: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_191, arg212_1);  mul_191 = arg212_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_532: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_175, [128, 4096])
        permute_209: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        mm_76: "f32[128, 4096]" = torch.ops.aten.mm.default(view_532, permute_209);  view_532 = permute_209 = None
        view_533: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_76, [1, 128, 4096]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_538: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_533, [1, 128, 16, 256]);  view_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_158: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_538, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_266: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_79, 2)
        unsqueeze_267: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 4);  unsqueeze_266 = None
        expand_156: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_267, [1, 128, 1, 32, 2]);  unsqueeze_267 = None
        clone_156: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_545: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_156, [1, 128, 1, 64]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_194: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_158, view_545);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_163: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_158, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_39: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_163);  slice_163 = None
        unsqueeze_268: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_39, 4);  neg_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_162: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_158, 3, 0, 9223372036854775807, 2);  slice_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_269: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_162, 4);  slice_162 = None
        cat_78: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_268, unsqueeze_269], -1);  unsqueeze_268 = unsqueeze_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_546: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_78, [1, 128, 16, 64]);  cat_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_264: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_78, 2)
        unsqueeze_265: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 4);  unsqueeze_264 = None
        expand_155: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_265, [1, 128, 1, 32, 2]);  unsqueeze_265 = None
        clone_155: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_544: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_155, [1, 128, 1, 64]);  clone_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_195: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_546, view_544);  view_546 = view_544 = None
        add_177: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_159: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_538, 3, 64, 9223372036854775807);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_80: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_177, slice_159], -1);  add_177 = slice_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_214: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_80, [0, 2, 1, 3]);  cat_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_534: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_175, [128, 4096])
        permute_210: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        mm_77: "f32[128, 4096]" = torch.ops.aten.mm.default(view_534, permute_210);  view_534 = permute_210 = None
        view_535: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_77, [1, 128, 4096]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_539: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_535, [1, 128, 16, 256]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_156: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_539, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_260: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_79, 2);  getitem_79 = None
        unsqueeze_261: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 4);  unsqueeze_260 = None
        expand_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_261, [1, 128, 1, 32, 2]);  unsqueeze_261 = None
        clone_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_542: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_192: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_156, view_542);  view_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_161: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_156, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_38: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_161);  slice_161 = None
        unsqueeze_262: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_38, 4);  neg_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_160: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_156, 3, 0, 9223372036854775807, 2);  slice_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_263: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_160, 4);  slice_160 = None
        cat_77: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_262, unsqueeze_263], -1);  unsqueeze_262 = unsqueeze_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_543: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_77, [1, 128, 16, 64]);  cat_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_258: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_78, 2);  getitem_78 = None
        unsqueeze_259: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 4);  unsqueeze_258 = None
        expand_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_259, [1, 128, 1, 32, 2]);  unsqueeze_259 = None
        clone_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_541: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_193: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_543, view_541);  view_543 = view_541 = None
        add_176: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_157: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_539, 3, 64, 9223372036854775807);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_79: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_176, slice_157], -1);  add_176 = slice_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_213: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_79, [0, 2, 1, 3]);  cat_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_536: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_175, [128, 4096])
        permute_211: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        mm_78: "f32[128, 4096]" = torch.ops.aten.mm.default(view_536, permute_211);  view_536 = permute_211 = None
        view_537: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_78, [1, 128, 4096]);  mm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_540: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_537, [1, 128, 16, 256]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_212: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # No stacktrace found for following nodes
        expand_default_8: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_214, permute_213, permute_212, expand_default_8, False, scale = 0.0625);  permute_214 = permute_213 = permute_212 = expand_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_124: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_216: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_124, [0, 2, 1, 3]);  getitem_124 = None
        clone_158: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_553: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_158, [1, 128, 4096]);  clone_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_554: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_553, [128, 4096]);  view_553 = None
        permute_217: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_79: "f32[128, 4096]" = torch.ops.aten.mm.default(view_554, permute_217);  view_554 = permute_217 = None
        view_555: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_79, [1, 128, 4096]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_556: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_175, [128, 4096]);  add_175 = None
        permute_218: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_38: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg219_1, view_556, permute_218);  arg219_1 = view_556 = permute_218 = None
        view_557: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_196: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_557, 0.5)
        pow_20: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_557, 3.0)
        mul_197: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_179: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_557, mul_197);  view_557 = mul_197 = None
        mul_198: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_179, 0.7978845608028654);  add_179 = None
        tanh_19: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_180: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_199: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_196, add_180);  mul_196 = add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_558: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_199, [128, 16384]);  mul_199 = None
        permute_219: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_39: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg221_1, view_558, permute_219);  arg221_1 = view_558 = permute_219 = None
        view_559: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 4096]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_181: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_555, view_559);  view_555 = view_559 = None
        add_182: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_181, add_173);  add_181 = add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 128, 1]" = var_mean_20[0]
        getitem_81: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_40: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg227_1, [1, 1, 1]);  arg227_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_270: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_41: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_270, [1, 1, 64]);  unsqueeze_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_20: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_40, 1, repeat_41);  repeat_40 = repeat_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_20 = torch.ops.aten.split.Tensor(gather_20, 32, -1);  gather_20 = None
        getitem_82: "f32[1, 128, 32]" = split_20[0]
        getitem_83: "f32[1, 128, 32]" = split_20[1];  split_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_42: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_182, getitem_81);  getitem_81 = None
        add_183: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        mul_200: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_20);  sub_42 = rsqrt_20 = None
        mul_201: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, arg222_1);  mul_200 = arg222_1 = None
        add_184: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_201, arg223_1);  mul_201 = arg223_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_560: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_184, [128, 4096])
        permute_220: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        mm_80: "f32[128, 4096]" = torch.ops.aten.mm.default(view_560, permute_220);  view_560 = permute_220 = None
        view_561: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_80, [1, 128, 4096]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_566: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_561, [1, 128, 16, 256]);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_166: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_566, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_279: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_83, 2)
        unsqueeze_280: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 4);  unsqueeze_279 = None
        expand_164: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_280, [1, 128, 1, 32, 2]);  unsqueeze_280 = None
        clone_164: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_573: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_164, [1, 128, 1, 64]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_204: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_166, view_573);  view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_171: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_166, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_41: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_171);  slice_171 = None
        unsqueeze_281: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_41, 4);  neg_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_170: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_166, 3, 0, 9223372036854775807, 2);  slice_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_282: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_170, 4);  slice_170 = None
        cat_82: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_281, unsqueeze_282], -1);  unsqueeze_281 = unsqueeze_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_574: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_82, [1, 128, 16, 64]);  cat_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_277: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_82, 2)
        unsqueeze_278: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 4);  unsqueeze_277 = None
        expand_163: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_278, [1, 128, 1, 32, 2]);  unsqueeze_278 = None
        clone_163: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_572: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_163, [1, 128, 1, 64]);  clone_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_205: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_574, view_572);  view_574 = view_572 = None
        add_186: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_167: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_566, 3, 64, 9223372036854775807);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_84: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_186, slice_167], -1);  add_186 = slice_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_225: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_84, [0, 2, 1, 3]);  cat_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_562: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_184, [128, 4096])
        permute_221: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        mm_81: "f32[128, 4096]" = torch.ops.aten.mm.default(view_562, permute_221);  view_562 = permute_221 = None
        view_563: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_81, [1, 128, 4096]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_567: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_563, [1, 128, 16, 256]);  view_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_164: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_567, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_273: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_83, 2);  getitem_83 = None
        unsqueeze_274: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 4);  unsqueeze_273 = None
        expand_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_274, [1, 128, 1, 32, 2]);  unsqueeze_274 = None
        clone_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_570: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_202: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_164, view_570);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_169: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_164, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_40: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_169);  slice_169 = None
        unsqueeze_275: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_40, 4);  neg_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_168: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_164, 3, 0, 9223372036854775807, 2);  slice_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_276: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_168, 4);  slice_168 = None
        cat_81: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_275, unsqueeze_276], -1);  unsqueeze_275 = unsqueeze_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_571: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_81, [1, 128, 16, 64]);  cat_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_271: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_82, 2);  getitem_82 = None
        unsqueeze_272: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 4);  unsqueeze_271 = None
        expand_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_272, [1, 128, 1, 32, 2]);  unsqueeze_272 = None
        clone_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_569: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_203: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_571, view_569);  view_571 = view_569 = None
        add_185: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_165: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_567, 3, 64, 9223372036854775807);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_83: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_185, slice_165], -1);  add_185 = slice_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_224: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_83, [0, 2, 1, 3]);  cat_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_564: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_184, [128, 4096])
        permute_222: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        mm_82: "f32[128, 4096]" = torch.ops.aten.mm.default(view_564, permute_222);  view_564 = permute_222 = None
        view_565: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_82, [1, 128, 4096]);  mm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_568: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_565, [1, 128, 16, 256]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_223: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # No stacktrace found for following nodes
        expand_default_7: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_225, permute_224, permute_223, expand_default_7, False, scale = 0.0625);  permute_225 = permute_224 = permute_223 = expand_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_123: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_227: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_123, [0, 2, 1, 3]);  getitem_123 = None
        clone_166: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_581: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_166, [1, 128, 4096]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_582: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_581, [128, 4096]);  view_581 = None
        permute_228: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        mm_83: "f32[128, 4096]" = torch.ops.aten.mm.default(view_582, permute_228);  view_582 = permute_228 = None
        view_583: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_83, [1, 128, 4096]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_584: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_184, [128, 4096]);  add_184 = None
        permute_229: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_40: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg230_1, view_584, permute_229);  arg230_1 = view_584 = permute_229 = None
        view_585: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_206: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_585, 0.5)
        pow_21: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_585, 3.0)
        mul_207: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_188: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_585, mul_207);  view_585 = mul_207 = None
        mul_208: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_188, 0.7978845608028654);  add_188 = None
        tanh_20: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
        add_189: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_209: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_206, add_189);  mul_206 = add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_586: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_209, [128, 16384]);  mul_209 = None
        permute_230: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_41: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg232_1, view_586, permute_230);  arg232_1 = view_586 = permute_230 = None
        view_587: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 4096]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_190: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_583, view_587);  view_583 = view_587 = None
        add_191: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_190, add_182);  add_190 = add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
        getitem_84: "f32[1, 128, 1]" = var_mean_21[0]
        getitem_85: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_42: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg238_1, [1, 1, 1]);  arg238_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_283: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_43: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_283, [1, 1, 64]);  unsqueeze_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_21: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_42, 1, repeat_43);  repeat_42 = repeat_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_21 = torch.ops.aten.split.Tensor(gather_21, 32, -1);  gather_21 = None
        getitem_86: "f32[1, 128, 32]" = split_21[0]
        getitem_87: "f32[1, 128, 32]" = split_21[1];  split_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_44: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_191, getitem_85);  getitem_85 = None
        add_192: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_210: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = rsqrt_21 = None
        mul_211: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, arg233_1);  mul_210 = arg233_1 = None
        add_193: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_211, arg234_1);  mul_211 = arg234_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_588: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096])
        permute_231: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        mm_84: "f32[128, 4096]" = torch.ops.aten.mm.default(view_588, permute_231);  view_588 = permute_231 = None
        view_589: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_84, [1, 128, 4096]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_594: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_589, [1, 128, 16, 256]);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_174: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_594, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_292: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_87, 2)
        unsqueeze_293: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 4);  unsqueeze_292 = None
        expand_172: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_293, [1, 128, 1, 32, 2]);  unsqueeze_293 = None
        clone_172: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_601: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_172, [1, 128, 1, 64]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_214: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_174, view_601);  view_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_179: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_174, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_43: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_179);  slice_179 = None
        unsqueeze_294: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_43, 4);  neg_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_178: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_174, 3, 0, 9223372036854775807, 2);  slice_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_295: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_178, 4);  slice_178 = None
        cat_86: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_294, unsqueeze_295], -1);  unsqueeze_294 = unsqueeze_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_602: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_86, [1, 128, 16, 64]);  cat_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_290: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_86, 2)
        unsqueeze_291: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 4);  unsqueeze_290 = None
        expand_171: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_291, [1, 128, 1, 32, 2]);  unsqueeze_291 = None
        clone_171: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_600: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_171, [1, 128, 1, 64]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_215: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_602, view_600);  view_602 = view_600 = None
        add_195: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_175: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_594, 3, 64, 9223372036854775807);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_88: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_195, slice_175], -1);  add_195 = slice_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_236: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_88, [0, 2, 1, 3]);  cat_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_590: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096])
        permute_232: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        mm_85: "f32[128, 4096]" = torch.ops.aten.mm.default(view_590, permute_232);  view_590 = permute_232 = None
        view_591: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_85, [1, 128, 4096]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_595: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_591, [1, 128, 16, 256]);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_172: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_595, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_286: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_87, 2);  getitem_87 = None
        unsqueeze_287: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 4);  unsqueeze_286 = None
        expand_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_287, [1, 128, 1, 32, 2]);  unsqueeze_287 = None
        clone_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_598: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_212: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_172, view_598);  view_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_177: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_172, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_42: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_177);  slice_177 = None
        unsqueeze_288: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_42, 4);  neg_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_176: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_172, 3, 0, 9223372036854775807, 2);  slice_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_289: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_176, 4);  slice_176 = None
        cat_85: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_288, unsqueeze_289], -1);  unsqueeze_288 = unsqueeze_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_599: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_85, [1, 128, 16, 64]);  cat_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_284: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_86, 2);  getitem_86 = None
        unsqueeze_285: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 4);  unsqueeze_284 = None
        expand_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_285, [1, 128, 1, 32, 2]);  unsqueeze_285 = None
        clone_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_597: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_213: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_599, view_597);  view_599 = view_597 = None
        add_194: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_173: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_595, 3, 64, 9223372036854775807);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_87: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_194, slice_173], -1);  add_194 = slice_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_235: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_87, [0, 2, 1, 3]);  cat_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_592: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096])
        permute_233: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        mm_86: "f32[128, 4096]" = torch.ops.aten.mm.default(view_592, permute_233);  view_592 = permute_233 = None
        view_593: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_86, [1, 128, 4096]);  mm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_596: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_593, [1, 128, 16, 256]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_234: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # No stacktrace found for following nodes
        expand_default_6: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_236, permute_235, permute_234, expand_default_6, False, scale = 0.0625);  permute_236 = permute_235 = permute_234 = expand_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_122: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_238: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3]);  getitem_122 = None
        clone_174: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_609: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_174, [1, 128, 4096]);  clone_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_610: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_609, [128, 4096]);  view_609 = None
        permute_239: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_87: "f32[128, 4096]" = torch.ops.aten.mm.default(view_610, permute_239);  view_610 = permute_239 = None
        view_611: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_87, [1, 128, 4096]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_612: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096]);  add_193 = None
        permute_240: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_42: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg241_1, view_612, permute_240);  arg241_1 = view_612 = permute_240 = None
        view_613: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_216: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_613, 0.5)
        pow_22: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_613, 3.0)
        mul_217: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_197: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_613, mul_217);  view_613 = mul_217 = None
        mul_218: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_197, 0.7978845608028654);  add_197 = None
        tanh_21: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_198: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_219: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_216, add_198);  mul_216 = add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_614: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_219, [128, 16384]);  mul_219 = None
        permute_241: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_43: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg243_1, view_614, permute_241);  arg243_1 = view_614 = permute_241 = None
        view_615: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 4096]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_199: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_611, view_615);  view_611 = view_615 = None
        add_200: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_199, add_191);  add_199 = add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_200, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 128, 1]" = var_mean_22[0]
        getitem_89: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_44: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg249_1, [1, 1, 1]);  arg249_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_296: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_45: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_296, [1, 1, 64]);  unsqueeze_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_22: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_44, 1, repeat_45);  repeat_44 = repeat_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_22 = torch.ops.aten.split.Tensor(gather_22, 32, -1);  gather_22 = None
        getitem_90: "f32[1, 128, 32]" = split_22[0]
        getitem_91: "f32[1, 128, 32]" = split_22[1];  split_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_46: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_200, getitem_89);  getitem_89 = None
        add_201: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        mul_220: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_22);  sub_46 = rsqrt_22 = None
        mul_221: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, arg244_1);  mul_220 = arg244_1 = None
        add_202: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_221, arg245_1);  mul_221 = arg245_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_616: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_202, [128, 4096])
        permute_242: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        mm_88: "f32[128, 4096]" = torch.ops.aten.mm.default(view_616, permute_242);  view_616 = permute_242 = None
        view_617: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_88, [1, 128, 4096]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_622: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_617, [1, 128, 16, 256]);  view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_182: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_622, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_305: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_91, 2)
        unsqueeze_306: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 4);  unsqueeze_305 = None
        expand_180: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_306, [1, 128, 1, 32, 2]);  unsqueeze_306 = None
        clone_180: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_629: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_180, [1, 128, 1, 64]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_224: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_182, view_629);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_187: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_182, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_45: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_187);  slice_187 = None
        unsqueeze_307: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_45, 4);  neg_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_186: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_182, 3, 0, 9223372036854775807, 2);  slice_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_308: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_186, 4);  slice_186 = None
        cat_90: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_307, unsqueeze_308], -1);  unsqueeze_307 = unsqueeze_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_630: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_90, [1, 128, 16, 64]);  cat_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_303: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_90, 2)
        unsqueeze_304: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
        expand_179: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_304, [1, 128, 1, 32, 2]);  unsqueeze_304 = None
        clone_179: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_628: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_179, [1, 128, 1, 64]);  clone_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_225: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_630, view_628);  view_630 = view_628 = None
        add_204: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_183: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_622, 3, 64, 9223372036854775807);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_92: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_204, slice_183], -1);  add_204 = slice_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_247: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_92, [0, 2, 1, 3]);  cat_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_618: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_202, [128, 4096])
        permute_243: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        mm_89: "f32[128, 4096]" = torch.ops.aten.mm.default(view_618, permute_243);  view_618 = permute_243 = None
        view_619: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_89, [1, 128, 4096]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_623: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_619, [1, 128, 16, 256]);  view_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_180: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_623, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_299: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_91, 2);  getitem_91 = None
        unsqueeze_300: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 4);  unsqueeze_299 = None
        expand_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_300, [1, 128, 1, 32, 2]);  unsqueeze_300 = None
        clone_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_626: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_222: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_180, view_626);  view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_185: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_180, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_44: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_185);  slice_185 = None
        unsqueeze_301: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_44, 4);  neg_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_184: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_180, 3, 0, 9223372036854775807, 2);  slice_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_302: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_184, 4);  slice_184 = None
        cat_89: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_301, unsqueeze_302], -1);  unsqueeze_301 = unsqueeze_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_627: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_89, [1, 128, 16, 64]);  cat_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_297: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_90, 2);  getitem_90 = None
        unsqueeze_298: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 4);  unsqueeze_297 = None
        expand_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_298, [1, 128, 1, 32, 2]);  unsqueeze_298 = None
        clone_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_625: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_223: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_627, view_625);  view_627 = view_625 = None
        add_203: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_181: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_623, 3, 64, 9223372036854775807);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_91: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_203, slice_181], -1);  add_203 = slice_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_246: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_91, [0, 2, 1, 3]);  cat_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_620: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_202, [128, 4096])
        permute_244: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        mm_90: "f32[128, 4096]" = torch.ops.aten.mm.default(view_620, permute_244);  view_620 = permute_244 = None
        view_621: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_90, [1, 128, 4096]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_624: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_621, [1, 128, 16, 256]);  view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_245: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None

        # No stacktrace found for following nodes
        expand_default_5: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_247, permute_246, permute_245, expand_default_5, False, scale = 0.0625);  permute_247 = permute_246 = permute_245 = expand_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_121: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_249: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
        clone_182: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_637: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_182, [1, 128, 4096]);  clone_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_638: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_637, [128, 4096]);  view_637 = None
        permute_250: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_91: "f32[128, 4096]" = torch.ops.aten.mm.default(view_638, permute_250);  view_638 = permute_250 = None
        view_639: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_91, [1, 128, 4096]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_640: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_202, [128, 4096]);  add_202 = None
        permute_251: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_44: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg252_1, view_640, permute_251);  arg252_1 = view_640 = permute_251 = None
        view_641: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_226: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_641, 0.5)
        pow_23: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_641, 3.0)
        mul_227: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_206: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_641, mul_227);  view_641 = mul_227 = None
        mul_228: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_206, 0.7978845608028654);  add_206 = None
        tanh_22: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
        add_207: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_229: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_226, add_207);  mul_226 = add_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_642: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_229, [128, 16384]);  mul_229 = None
        permute_252: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_45: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg254_1, view_642, permute_252);  arg254_1 = view_642 = permute_252 = None
        view_643: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 4096]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_208: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_639, view_643);  view_639 = view_643 = None
        add_209: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_208, add_200);  add_208 = add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_209, [2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 128, 1]" = var_mean_23[0]
        getitem_93: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_46: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg260_1, [1, 1, 1]);  arg260_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_309: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_47: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_309, [1, 1, 64]);  unsqueeze_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_23: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_46, 1, repeat_47);  repeat_46 = repeat_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_23 = torch.ops.aten.split.Tensor(gather_23, 32, -1);  gather_23 = None
        getitem_94: "f32[1, 128, 32]" = split_23[0]
        getitem_95: "f32[1, 128, 32]" = split_23[1];  split_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_48: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_209, getitem_93);  getitem_93 = None
        add_210: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_230: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = rsqrt_23 = None
        mul_231: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, arg255_1);  mul_230 = arg255_1 = None
        add_211: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_231, arg256_1);  mul_231 = arg256_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_644: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_211, [128, 4096])
        permute_253: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        mm_92: "f32[128, 4096]" = torch.ops.aten.mm.default(view_644, permute_253);  view_644 = permute_253 = None
        view_645: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_92, [1, 128, 4096]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_650: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_645, [1, 128, 16, 256]);  view_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_190: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_650, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_318: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_95, 2)
        unsqueeze_319: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 4);  unsqueeze_318 = None
        expand_188: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_319, [1, 128, 1, 32, 2]);  unsqueeze_319 = None
        clone_188: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_657: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_188, [1, 128, 1, 64]);  clone_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_234: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_190, view_657);  view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_195: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_190, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_47: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_195);  slice_195 = None
        unsqueeze_320: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_47, 4);  neg_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_194: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_190, 3, 0, 9223372036854775807, 2);  slice_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_321: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_194, 4);  slice_194 = None
        cat_94: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_320, unsqueeze_321], -1);  unsqueeze_320 = unsqueeze_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_658: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_94, [1, 128, 16, 64]);  cat_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_316: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_94, 2)
        unsqueeze_317: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 4);  unsqueeze_316 = None
        expand_187: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_317, [1, 128, 1, 32, 2]);  unsqueeze_317 = None
        clone_187: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_656: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_187, [1, 128, 1, 64]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_235: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_658, view_656);  view_658 = view_656 = None
        add_213: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_191: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_650, 3, 64, 9223372036854775807);  view_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_96: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_213, slice_191], -1);  add_213 = slice_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_258: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_96, [0, 2, 1, 3]);  cat_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_646: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_211, [128, 4096])
        permute_254: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        mm_93: "f32[128, 4096]" = torch.ops.aten.mm.default(view_646, permute_254);  view_646 = permute_254 = None
        view_647: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_93, [1, 128, 4096]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_651: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_647, [1, 128, 16, 256]);  view_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_188: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_651, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_312: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_95, 2);  getitem_95 = None
        unsqueeze_313: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 4);  unsqueeze_312 = None
        expand_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_313, [1, 128, 1, 32, 2]);  unsqueeze_313 = None
        clone_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_654: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_232: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_188, view_654);  view_654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_193: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_188, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_46: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_193);  slice_193 = None
        unsqueeze_314: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_46, 4);  neg_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_192: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_188, 3, 0, 9223372036854775807, 2);  slice_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_315: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_192, 4);  slice_192 = None
        cat_93: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_314, unsqueeze_315], -1);  unsqueeze_314 = unsqueeze_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_655: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_93, [1, 128, 16, 64]);  cat_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_310: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_94, 2);  getitem_94 = None
        unsqueeze_311: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 4);  unsqueeze_310 = None
        expand_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_311, [1, 128, 1, 32, 2]);  unsqueeze_311 = None
        clone_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_653: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_233: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_655, view_653);  view_655 = view_653 = None
        add_212: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_189: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_651, 3, 64, 9223372036854775807);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_95: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_212, slice_189], -1);  add_212 = slice_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_257: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_95, [0, 2, 1, 3]);  cat_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_648: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_211, [128, 4096])
        permute_255: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        mm_94: "f32[128, 4096]" = torch.ops.aten.mm.default(view_648, permute_255);  view_648 = permute_255 = None
        view_649: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_94, [1, 128, 4096]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_652: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_649, [1, 128, 16, 256]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_256: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None

        # No stacktrace found for following nodes
        expand_default_4: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_258, permute_257, permute_256, expand_default_4, False, scale = 0.0625);  permute_258 = permute_257 = permute_256 = expand_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_120: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_260: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        clone_190: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_665: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_190, [1, 128, 4096]);  clone_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_666: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_665, [128, 4096]);  view_665 = None
        permute_261: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_95: "f32[128, 4096]" = torch.ops.aten.mm.default(view_666, permute_261);  view_666 = permute_261 = None
        view_667: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_95, [1, 128, 4096]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_668: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_211, [128, 4096]);  add_211 = None
        permute_262: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_46: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg263_1, view_668, permute_262);  arg263_1 = view_668 = permute_262 = None
        view_669: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_669, 0.5)
        pow_24: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_669, 3.0)
        mul_237: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_215: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_669, mul_237);  view_669 = mul_237 = None
        mul_238: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_215, 0.7978845608028654);  add_215 = None
        tanh_23: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_216: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_239: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_236, add_216);  mul_236 = add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_670: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_239, [128, 16384]);  mul_239 = None
        permute_263: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_47: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg265_1, view_670, permute_263);  arg265_1 = view_670 = permute_263 = None
        view_671: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 4096]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_217: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_667, view_671);  view_667 = view_671 = None
        add_218: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_217, add_209);  add_217 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_218, [2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 128, 1]" = var_mean_24[0]
        getitem_97: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_48: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg271_1, [1, 1, 1]);  arg271_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_322: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_49: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_322, [1, 1, 64]);  unsqueeze_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_24: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_48, 1, repeat_49);  repeat_48 = repeat_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_24 = torch.ops.aten.split.Tensor(gather_24, 32, -1);  gather_24 = None
        getitem_98: "f32[1, 128, 32]" = split_24[0]
        getitem_99: "f32[1, 128, 32]" = split_24[1];  split_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_50: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_218, getitem_97);  getitem_97 = None
        add_219: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_240: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_24);  sub_50 = rsqrt_24 = None
        mul_241: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, arg266_1);  mul_240 = arg266_1 = None
        add_220: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_241, arg267_1);  mul_241 = arg267_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_672: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_220, [128, 4096])
        permute_264: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        mm_96: "f32[128, 4096]" = torch.ops.aten.mm.default(view_672, permute_264);  view_672 = permute_264 = None
        view_673: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_96, [1, 128, 4096]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_678: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_673, [1, 128, 16, 256]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_198: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_678, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_331: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_99, 2)
        unsqueeze_332: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 4);  unsqueeze_331 = None
        expand_196: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_332, [1, 128, 1, 32, 2]);  unsqueeze_332 = None
        clone_196: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_196, memory_format = torch.contiguous_format);  expand_196 = None
        view_685: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_196, [1, 128, 1, 64]);  clone_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_244: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_198, view_685);  view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_203: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_198, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_49: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_203);  slice_203 = None
        unsqueeze_333: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_49, 4);  neg_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_202: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_198, 3, 0, 9223372036854775807, 2);  slice_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_334: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_202, 4);  slice_202 = None
        cat_98: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_333, unsqueeze_334], -1);  unsqueeze_333 = unsqueeze_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_686: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_98, [1, 128, 16, 64]);  cat_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_329: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_98, 2)
        unsqueeze_330: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 4);  unsqueeze_329 = None
        expand_195: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_330, [1, 128, 1, 32, 2]);  unsqueeze_330 = None
        clone_195: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_195, memory_format = torch.contiguous_format);  expand_195 = None
        view_684: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_195, [1, 128, 1, 64]);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_245: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_686, view_684);  view_686 = view_684 = None
        add_222: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_199: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_678, 3, 64, 9223372036854775807);  view_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_100: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_222, slice_199], -1);  add_222 = slice_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_269: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_100, [0, 2, 1, 3]);  cat_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_674: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_220, [128, 4096])
        permute_265: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        mm_97: "f32[128, 4096]" = torch.ops.aten.mm.default(view_674, permute_265);  view_674 = permute_265 = None
        view_675: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_97, [1, 128, 4096]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_679: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_675, [1, 128, 16, 256]);  view_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_196: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_679, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_325: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_99, 2);  getitem_99 = None
        unsqueeze_326: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 4);  unsqueeze_325 = None
        expand_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_326, [1, 128, 1, 32, 2]);  unsqueeze_326 = None
        clone_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_682: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_242: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_196, view_682);  view_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_201: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_196, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_48: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_201);  slice_201 = None
        unsqueeze_327: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_48, 4);  neg_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_200: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_196, 3, 0, 9223372036854775807, 2);  slice_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_328: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_200, 4);  slice_200 = None
        cat_97: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_327, unsqueeze_328], -1);  unsqueeze_327 = unsqueeze_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_683: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_97, [1, 128, 16, 64]);  cat_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_323: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_98, 2);  getitem_98 = None
        unsqueeze_324: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 4);  unsqueeze_323 = None
        expand_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_324, [1, 128, 1, 32, 2]);  unsqueeze_324 = None
        clone_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_681: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_243: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_683, view_681);  view_683 = view_681 = None
        add_221: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_197: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_679, 3, 64, 9223372036854775807);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_99: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_221, slice_197], -1);  add_221 = slice_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_268: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_99, [0, 2, 1, 3]);  cat_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_676: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_220, [128, 4096])
        permute_266: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        mm_98: "f32[128, 4096]" = torch.ops.aten.mm.default(view_676, permute_266);  view_676 = permute_266 = None
        view_677: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_98, [1, 128, 4096]);  mm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_680: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_677, [1, 128, 16, 256]);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_267: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_680, [0, 2, 1, 3]);  view_680 = None

        # No stacktrace found for following nodes
        expand_default_3: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_269, permute_268, permute_267, expand_default_3, False, scale = 0.0625);  permute_269 = permute_268 = permute_267 = expand_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_119: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_271: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None
        clone_198: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_693: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_198, [1, 128, 4096]);  clone_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_694: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_693, [128, 4096]);  view_693 = None
        permute_272: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        mm_99: "f32[128, 4096]" = torch.ops.aten.mm.default(view_694, permute_272);  view_694 = permute_272 = None
        view_695: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_99, [1, 128, 4096]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_696: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_220, [128, 4096]);  add_220 = None
        permute_273: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_48: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg274_1, view_696, permute_273);  arg274_1 = view_696 = permute_273 = None
        view_697: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_246: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_697, 0.5)
        pow_25: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_697, 3.0)
        mul_247: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_224: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_697, mul_247);  view_697 = mul_247 = None
        mul_248: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_224, 0.7978845608028654);  add_224 = None
        tanh_24: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
        add_225: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0);  tanh_24 = None
        mul_249: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_246, add_225);  mul_246 = add_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_698: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_249, [128, 16384]);  mul_249 = None
        permute_274: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_49: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg276_1, view_698, permute_274);  arg276_1 = view_698 = permute_274 = None
        view_699: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 4096]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_226: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_695, view_699);  view_695 = view_699 = None
        add_227: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_226, add_218);  add_226 = add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_227, [2], correction = 0, keepdim = True)
        getitem_100: "f32[1, 128, 1]" = var_mean_25[0]
        getitem_101: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_50: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg282_1, [1, 1, 1]);  arg282_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_335: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_51: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_335, [1, 1, 64]);  unsqueeze_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_25: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_50, 1, repeat_51);  repeat_50 = repeat_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_25 = torch.ops.aten.split.Tensor(gather_25, 32, -1);  gather_25 = None
        getitem_102: "f32[1, 128, 32]" = split_25[0]
        getitem_103: "f32[1, 128, 32]" = split_25[1];  split_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_52: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_227, getitem_101);  getitem_101 = None
        add_228: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        mul_250: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_25);  sub_52 = rsqrt_25 = None
        mul_251: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, arg277_1);  mul_250 = arg277_1 = None
        add_229: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_251, arg278_1);  mul_251 = arg278_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_700: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_229, [128, 4096])
        permute_275: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        mm_100: "f32[128, 4096]" = torch.ops.aten.mm.default(view_700, permute_275);  view_700 = permute_275 = None
        view_701: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_100, [1, 128, 4096]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_706: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_701, [1, 128, 16, 256]);  view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_206: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_706, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_344: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_103, 2)
        unsqueeze_345: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 4);  unsqueeze_344 = None
        expand_204: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_345, [1, 128, 1, 32, 2]);  unsqueeze_345 = None
        clone_204: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_204, memory_format = torch.contiguous_format);  expand_204 = None
        view_713: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_204, [1, 128, 1, 64]);  clone_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_254: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_206, view_713);  view_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_211: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_206, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_51: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_211);  slice_211 = None
        unsqueeze_346: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_51, 4);  neg_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_210: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_206, 3, 0, 9223372036854775807, 2);  slice_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_347: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_210, 4);  slice_210 = None
        cat_102: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_346, unsqueeze_347], -1);  unsqueeze_346 = unsqueeze_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_714: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_102, [1, 128, 16, 64]);  cat_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_342: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_102, 2)
        unsqueeze_343: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 4);  unsqueeze_342 = None
        expand_203: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_343, [1, 128, 1, 32, 2]);  unsqueeze_343 = None
        clone_203: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_203, memory_format = torch.contiguous_format);  expand_203 = None
        view_712: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_203, [1, 128, 1, 64]);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_255: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_714, view_712);  view_714 = view_712 = None
        add_231: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_207: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_706, 3, 64, 9223372036854775807);  view_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_104: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_231, slice_207], -1);  add_231 = slice_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_280: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_104, [0, 2, 1, 3]);  cat_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_702: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_229, [128, 4096])
        permute_276: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        mm_101: "f32[128, 4096]" = torch.ops.aten.mm.default(view_702, permute_276);  view_702 = permute_276 = None
        view_703: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_101, [1, 128, 4096]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_707: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_703, [1, 128, 16, 256]);  view_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_204: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_707, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_338: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_103, 2);  getitem_103 = None
        unsqueeze_339: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
        expand_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_339, [1, 128, 1, 32, 2]);  unsqueeze_339 = None
        clone_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_710: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_252: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_204, view_710);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_209: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_204, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_50: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_209);  slice_209 = None
        unsqueeze_340: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_50, 4);  neg_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_208: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_204, 3, 0, 9223372036854775807, 2);  slice_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_341: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_208, 4);  slice_208 = None
        cat_101: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_340, unsqueeze_341], -1);  unsqueeze_340 = unsqueeze_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_711: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_101, [1, 128, 16, 64]);  cat_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_336: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_102, 2);  getitem_102 = None
        unsqueeze_337: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 4);  unsqueeze_336 = None
        expand_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_337, [1, 128, 1, 32, 2]);  unsqueeze_337 = None
        clone_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_709: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_253: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_711, view_709);  view_711 = view_709 = None
        add_230: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_205: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_707, 3, 64, 9223372036854775807);  view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_103: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_230, slice_205], -1);  add_230 = slice_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_279: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_103, [0, 2, 1, 3]);  cat_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_704: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_229, [128, 4096])
        permute_277: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        mm_102: "f32[128, 4096]" = torch.ops.aten.mm.default(view_704, permute_277);  view_704 = permute_277 = None
        view_705: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_102, [1, 128, 4096]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_708: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_705, [1, 128, 16, 256]);  view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_278: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_708, [0, 2, 1, 3]);  view_708 = None

        # No stacktrace found for following nodes
        expand_default_2: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_280, permute_279, permute_278, expand_default_2, False, scale = 0.0625);  permute_280 = permute_279 = permute_278 = expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_118: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_282: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_118, [0, 2, 1, 3]);  getitem_118 = None
        clone_206: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_721: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_206, [1, 128, 4096]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_722: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_721, [128, 4096]);  view_721 = None
        permute_283: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        mm_103: "f32[128, 4096]" = torch.ops.aten.mm.default(view_722, permute_283);  view_722 = permute_283 = None
        view_723: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_103, [1, 128, 4096]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_724: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_229, [128, 4096]);  add_229 = None
        permute_284: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_50: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg285_1, view_724, permute_284);  arg285_1 = view_724 = permute_284 = None
        view_725: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_256: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_725, 0.5)
        pow_26: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_725, 3.0)
        mul_257: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_233: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_725, mul_257);  view_725 = mul_257 = None
        mul_258: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_25: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
        add_234: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0);  tanh_25 = None
        mul_259: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_256, add_234);  mul_256 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_726: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_259, [128, 16384]);  mul_259 = None
        permute_285: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        addmm_51: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg287_1, view_726, permute_285);  arg287_1 = view_726 = permute_285 = None
        view_727: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 4096]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_235: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_723, view_727);  view_723 = view_727 = None
        add_236: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_235, add_227);  add_235 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_236, [2], correction = 0, keepdim = True)
        getitem_104: "f32[1, 128, 1]" = var_mean_26[0]
        getitem_105: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_52: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg293_1, [1, 1, 1]);  arg293_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_348: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_53: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_348, [1, 1, 64]);  unsqueeze_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_26: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_52, 1, repeat_53);  repeat_52 = repeat_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_26 = torch.ops.aten.split.Tensor(gather_26, 32, -1);  gather_26 = None
        getitem_106: "f32[1, 128, 32]" = split_26[0]
        getitem_107: "f32[1, 128, 32]" = split_26[1];  split_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_54: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_236, getitem_105);  getitem_105 = None
        add_237: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        mul_260: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_26);  sub_54 = rsqrt_26 = None
        mul_261: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, arg288_1);  mul_260 = arg288_1 = None
        add_238: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_261, arg289_1);  mul_261 = arg289_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_728: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_238, [128, 4096])
        permute_286: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        mm_104: "f32[128, 4096]" = torch.ops.aten.mm.default(view_728, permute_286);  view_728 = permute_286 = None
        view_729: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_104, [1, 128, 4096]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_734: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_729, [1, 128, 16, 256]);  view_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_214: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_734, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_357: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_107, 2)
        unsqueeze_358: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 4);  unsqueeze_357 = None
        expand_212: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_358, [1, 128, 1, 32, 2]);  unsqueeze_358 = None
        clone_212: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_212, memory_format = torch.contiguous_format);  expand_212 = None
        view_741: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_212, [1, 128, 1, 64]);  clone_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_264: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_214, view_741);  view_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_219: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_214, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_53: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_219);  slice_219 = None
        unsqueeze_359: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_53, 4);  neg_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_218: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_214, 3, 0, 9223372036854775807, 2);  slice_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_360: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_218, 4);  slice_218 = None
        cat_106: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_359, unsqueeze_360], -1);  unsqueeze_359 = unsqueeze_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_742: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_106, [1, 128, 16, 64]);  cat_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_355: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_106, 2)
        unsqueeze_356: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
        expand_211: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_356, [1, 128, 1, 32, 2]);  unsqueeze_356 = None
        clone_211: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_211, memory_format = torch.contiguous_format);  expand_211 = None
        view_740: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_211, [1, 128, 1, 64]);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_265: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_742, view_740);  view_742 = view_740 = None
        add_240: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_215: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_734, 3, 64, 9223372036854775807);  view_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_108: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_240, slice_215], -1);  add_240 = slice_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_291: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_108, [0, 2, 1, 3]);  cat_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_730: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_238, [128, 4096])
        permute_287: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        mm_105: "f32[128, 4096]" = torch.ops.aten.mm.default(view_730, permute_287);  view_730 = permute_287 = None
        view_731: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_105, [1, 128, 4096]);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_735: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_731, [1, 128, 16, 256]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_212: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_735, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_351: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_107, 2);  getitem_107 = None
        unsqueeze_352: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 4);  unsqueeze_351 = None
        expand_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_352, [1, 128, 1, 32, 2]);  unsqueeze_352 = None
        clone_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_738: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_262: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_212, view_738);  view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_217: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_212, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_52: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_217);  slice_217 = None
        unsqueeze_353: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_52, 4);  neg_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_216: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_212, 3, 0, 9223372036854775807, 2);  slice_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_354: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_216, 4);  slice_216 = None
        cat_105: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_353, unsqueeze_354], -1);  unsqueeze_353 = unsqueeze_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_739: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_105, [1, 128, 16, 64]);  cat_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_349: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_106, 2);  getitem_106 = None
        unsqueeze_350: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 4);  unsqueeze_349 = None
        expand_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_350, [1, 128, 1, 32, 2]);  unsqueeze_350 = None
        clone_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_737: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_263: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_739, view_737);  view_739 = view_737 = None
        add_239: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_213: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_735, 3, 64, 9223372036854775807);  view_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_107: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_239, slice_213], -1);  add_239 = slice_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_290: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_107, [0, 2, 1, 3]);  cat_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_732: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_238, [128, 4096])
        permute_288: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        mm_106: "f32[128, 4096]" = torch.ops.aten.mm.default(view_732, permute_288);  view_732 = permute_288 = None
        view_733: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_106, [1, 128, 4096]);  mm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_736: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_733, [1, 128, 16, 256]);  view_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_289: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_736, [0, 2, 1, 3]);  view_736 = None

        # No stacktrace found for following nodes
        expand_default_1: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128])
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_291, permute_290, permute_289, expand_default_1, False, scale = 0.0625);  permute_291 = permute_290 = permute_289 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_117: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_293: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        clone_214: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_749: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_214, [1, 128, 4096]);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_750: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_749, [128, 4096]);  view_749 = None
        permute_294: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        mm_107: "f32[128, 4096]" = torch.ops.aten.mm.default(view_750, permute_294);  view_750 = permute_294 = None
        view_751: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_107, [1, 128, 4096]);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_752: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_238, [128, 4096]);  add_238 = None
        permute_295: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_52: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg296_1, view_752, permute_295);  arg296_1 = view_752 = permute_295 = None
        view_753: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_266: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_753, 0.5)
        pow_27: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_753, 3.0)
        mul_267: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_242: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_753, mul_267);  view_753 = mul_267 = None
        mul_268: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_242, 0.7978845608028654);  add_242 = None
        tanh_26: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
        add_243: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0);  tanh_26 = None
        mul_269: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_266, add_243);  mul_266 = add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_754: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_269, [128, 16384]);  mul_269 = None
        permute_296: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_53: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg298_1, view_754, permute_296);  arg298_1 = view_754 = permute_296 = None
        view_755: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 4096]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_244: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_751, view_755);  view_751 = view_755 = None
        add_245: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_244, add_236);  add_244 = add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_245, [2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 128, 1]" = var_mean_27[0]
        getitem_109: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_54: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg304_1, [1, 1, 1]);  arg304_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_361: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        repeat_55: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_361, [1, 1, 64]);  unsqueeze_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_27: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_54, 1, repeat_55);  repeat_54 = repeat_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_27 = torch.ops.aten.split.Tensor(gather_27, 32, -1);  gather_27 = None
        getitem_110: "f32[1, 128, 32]" = split_27[0]
        getitem_111: "f32[1, 128, 32]" = split_27[1];  split_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_56: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_245, getitem_109);  getitem_109 = None
        add_246: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        mul_270: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_27);  sub_56 = rsqrt_27 = None
        mul_271: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, arg299_1);  mul_270 = arg299_1 = None
        add_247: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_271, arg300_1);  mul_271 = arg300_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_756: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_247, [128, 4096])
        permute_297: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        mm_108: "f32[128, 4096]" = torch.ops.aten.mm.default(view_756, permute_297);  view_756 = permute_297 = None
        view_757: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_108, [1, 128, 4096]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_762: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_757, [1, 128, 16, 256]);  view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_222: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_762, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_370: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_111, 2)
        unsqueeze_371: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 4);  unsqueeze_370 = None
        expand_220: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_371, [1, 128, 1, 32, 2]);  unsqueeze_371 = None
        clone_220: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_220, memory_format = torch.contiguous_format);  expand_220 = None
        view_769: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_220, [1, 128, 1, 64]);  clone_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_274: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_222, view_769);  view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_227: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_222, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_55: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_227);  slice_227 = None
        unsqueeze_372: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_55, 4);  neg_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_226: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_222, 3, 0, 9223372036854775807, 2);  slice_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_373: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_226, 4);  slice_226 = None
        cat_110: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_372, unsqueeze_373], -1);  unsqueeze_372 = unsqueeze_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_770: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_110, [1, 128, 16, 64]);  cat_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_368: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_110, 2)
        unsqueeze_369: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 4);  unsqueeze_368 = None
        expand_219: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_369, [1, 128, 1, 32, 2]);  unsqueeze_369 = None
        clone_219: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_219, memory_format = torch.contiguous_format);  expand_219 = None
        view_768: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_219, [1, 128, 1, 64]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_275: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_770, view_768);  view_770 = view_768 = None
        add_249: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_223: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_762, 3, 64, 9223372036854775807);  view_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_112: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_249, slice_223], -1);  add_249 = slice_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_302: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_112, [0, 2, 1, 3]);  cat_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_758: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_247, [128, 4096])
        permute_298: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        mm_109: "f32[128, 4096]" = torch.ops.aten.mm.default(view_758, permute_298);  view_758 = permute_298 = None
        view_759: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_109, [1, 128, 4096]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_763: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_759, [1, 128, 16, 256]);  view_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_220: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_763, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_364: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_111, 2);  getitem_111 = None
        unsqueeze_365: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 4);  unsqueeze_364 = None
        expand_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_365, [1, 128, 1, 32, 2]);  unsqueeze_365 = None
        clone_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_766: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_272: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_220, view_766);  view_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_225: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_220, 3, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_54: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_225);  slice_225 = None
        unsqueeze_366: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_54, 4);  neg_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_224: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_220, 3, 0, 9223372036854775807, 2);  slice_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        unsqueeze_367: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_224, 4);  slice_224 = None
        cat_109: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_366, unsqueeze_367], -1);  unsqueeze_366 = unsqueeze_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_767: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_109, [1, 128, 16, 64]);  cat_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_362: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_110, 2);  getitem_110 = None
        unsqueeze_363: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 4);  unsqueeze_362 = None
        expand_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_363, [1, 128, 1, 32, 2]);  unsqueeze_363 = None
        clone_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_765: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_273: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_767, view_765);  view_767 = view_765 = None
        add_248: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_221: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_763, 3, 64, 9223372036854775807);  view_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_111: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_248, slice_221], -1);  add_248 = slice_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_301: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_111, [0, 2, 1, 3]);  cat_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_760: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_247, [128, 4096])
        permute_299: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        mm_110: "f32[128, 4096]" = torch.ops.aten.mm.default(view_760, permute_299);  view_760 = permute_299 = None
        view_761: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_110, [1, 128, 4096]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_764: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_761, [1, 128, 16, 256]);  view_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_300: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_764, [0, 2, 1, 3]);  view_764 = None

        # No stacktrace found for following nodes
        expand_default: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_302, permute_301, permute_300, expand_default, False, scale = 0.0625);  permute_302 = permute_301 = permute_300 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_116: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_304: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_116, [0, 2, 1, 3]);  getitem_116 = None
        clone_222: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_777: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_222, [1, 128, 4096]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        view_778: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_777, [128, 4096]);  view_777 = None
        permute_305: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        mm_111: "f32[128, 4096]" = torch.ops.aten.mm.default(view_778, permute_305);  view_778 = permute_305 = None
        view_779: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_111, [1, 128, 4096]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_780: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_247, [128, 4096]);  add_247 = None
        permute_306: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_54: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg307_1, view_780, permute_306);  arg307_1 = view_780 = permute_306 = None
        view_781: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_781, 0.5)
        pow_28: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_781, 3.0)
        mul_277: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_251: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_781, mul_277);  view_781 = mul_277 = None
        mul_278: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_251, 0.7978845608028654);  add_251 = None
        tanh_27: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_252: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0);  tanh_27 = None
        mul_279: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_276, add_252);  mul_276 = add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_782: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_279, [128, 16384]);  mul_279 = None
        permute_307: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        addmm_55: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg309_1, view_782, permute_307);  arg309_1 = view_782 = permute_307 = None
        view_783: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 4096]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_253: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_779, view_783);  view_779 = view_783 = None
        add_254: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_253, add_245);  add_253 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_254, [2], correction = 0, keepdim = True)
        getitem_112: "f32[1, 128, 1]" = var_mean_28[0]
        getitem_113: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_58: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_254, getitem_113);  add_254 = getitem_113 = None
        add_255: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        mul_280: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_28);  sub_58 = rsqrt_28 = None
        mul_281: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, arg310_1);  mul_280 = arg310_1 = None
        add_256: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_281, arg311_1);  mul_281 = arg311_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:815 in forward, code: logits = self.qa_outputs(sequence_output)
        view_785: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_256, [128, 4096]);  add_256 = None
        permute_308: "f32[4096, 2]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_56: "f32[128, 2]" = torch.ops.aten.addmm.default(arg313_1, view_785, permute_308);  arg313_1 = view_785 = permute_308 = None
        view_786: "f32[1, 128, 2]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 2]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:816 in forward, code: start_logits, end_logits = logits.split(1, dim=-1)
        split_28 = torch.ops.aten.split.Tensor(view_786, 1, -1);  view_786 = None
        getitem_114: "f32[1, 128, 1]" = split_28[0]
        getitem_115: "f32[1, 128, 1]" = split_28[1];  split_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:829 in forward, code: start_positions = start_positions.clamp(0, ignored_index)
        clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg314_1, 0);  arg314_1 = None
        clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:817 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        squeeze: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_114, -1);  getitem_114 = None
        clone_225: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        amax_28: "f32[1, 1]" = torch.ops.aten.amax.default(clone_225, [1], True)
        sub_59: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_225, amax_28);  amax_28 = None
        exp_28: "f32[1, 128]" = torch.ops.aten.exp.default(sub_59)
        sum_29: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True);  exp_28 = None
        log: "f32[1, 1]" = torch.ops.aten.log.default(sum_29);  sum_29 = None
        sub_60: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_59, log);  sub_59 = log = None
        ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[1]" = torch.ops.aten.where.self(ne_1, clamp_max, full_default_3);  ne_1 = full_default_3 = None
        unsqueeze_374: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather_28: "f32[1, 1]" = torch.ops.aten.gather.default(sub_60, 1, unsqueeze_374);  sub_60 = unsqueeze_374 = None
        squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather_28, 1);  gather_28 = None
        neg_56: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[1]" = torch.ops.aten.where.self(ne_2, neg_56, full_default_4);  ne_2 = neg_56 = full_default_4 = None
        sum_31: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128);  clamp_max = None
        sum_30: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
        div_56: "f32[]" = torch.ops.aten.div.Tensor(sum_31, convert_element_type);  sum_31 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:830 in forward, code: end_positions = end_positions.clamp(0, ignored_index)
        clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg315_1, 0);  arg315_1 = None
        clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:818 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        squeeze_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_115, -1);  getitem_115 = None
        clone_226: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        amax_29: "f32[1, 1]" = torch.ops.aten.amax.default(clone_226, [1], True)
        sub_61: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_226, amax_29);  amax_29 = None
        exp_29: "f32[1, 128]" = torch.ops.aten.exp.default(sub_61)
        sum_32: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [1], True);  exp_29 = None
        log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_32);  sum_32 = None
        sub_62: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_61, log_1);  sub_61 = log_1 = None
        ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
        full_default_5: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[1]" = torch.ops.aten.where.self(ne_4, clamp_max_1, full_default_5);  ne_4 = full_default_5 = None
        unsqueeze_375: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather_29: "f32[1, 1]" = torch.ops.aten.gather.default(sub_62, 1, unsqueeze_375);  sub_62 = unsqueeze_375 = None
        squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_29, 1);  gather_29 = None
        neg_57: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[1]" = torch.ops.aten.where.self(ne_5, neg_57, full_default_6);  ne_5 = neg_57 = full_default_6 = None
        sum_34: "f32[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        ne_6: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128);  clamp_max_1 = None
        sum_33: "i64[]" = torch.ops.aten.sum.default(ne_6);  ne_6 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_33, torch.float32);  sum_33 = None
        div_57: "f32[]" = torch.ops.aten.div.Tensor(sum_34, convert_element_type_1);  sum_34 = convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:835 in forward, code: total_loss = (start_loss + end_loss) / 2
        add_257: "f32[]" = torch.ops.aten.add.Tensor(div_56, div_57);  div_56 = div_57 = None
        div_58: "f32[]" = torch.ops.aten.div.Tensor(add_257, 2);  add_257 = None
        return (div_58, clone_225, clone_226)
