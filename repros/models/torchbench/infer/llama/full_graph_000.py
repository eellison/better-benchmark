class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i32[32, 32]", arg1_1: "f32[32000, 512]", arg2_1: "c64[2048, 32]", arg3_1: "f32[512]", arg4_1: "f32[512, 512]", arg5_1: "f32[512, 512]", arg6_1: "f32[512, 512]", arg7_1: "f32[64, 1024, 8, 64]", arg8_1: "f32[64, 1024, 8, 64]", arg9_1: "f32[512, 512]", arg10_1: "f32[512]", arg11_1: "f32[1536, 512]", arg12_1: "f32[1536, 512]", arg13_1: "f32[512, 1536]", arg14_1: "f32[512]", arg15_1: "f32[512, 512]", arg16_1: "f32[512, 512]", arg17_1: "f32[512, 512]", arg18_1: "f32[64, 1024, 8, 64]", arg19_1: "f32[64, 1024, 8, 64]", arg20_1: "f32[512, 512]", arg21_1: "f32[512]", arg22_1: "f32[1536, 512]", arg23_1: "f32[1536, 512]", arg24_1: "f32[512, 1536]", arg25_1: "f32[512]", arg26_1: "f32[512, 512]", arg27_1: "f32[512, 512]", arg28_1: "f32[512, 512]", arg29_1: "f32[64, 1024, 8, 64]", arg30_1: "f32[64, 1024, 8, 64]", arg31_1: "f32[512, 512]", arg32_1: "f32[512]", arg33_1: "f32[1536, 512]", arg34_1: "f32[1536, 512]", arg35_1: "f32[512, 1536]", arg36_1: "f32[512]", arg37_1: "f32[512, 512]", arg38_1: "f32[512, 512]", arg39_1: "f32[512, 512]", arg40_1: "f32[64, 1024, 8, 64]", arg41_1: "f32[64, 1024, 8, 64]", arg42_1: "f32[512, 512]", arg43_1: "f32[512]", arg44_1: "f32[1536, 512]", arg45_1: "f32[1536, 512]", arg46_1: "f32[512, 1536]", arg47_1: "f32[512]", arg48_1: "f32[512, 512]", arg49_1: "f32[512, 512]", arg50_1: "f32[512, 512]", arg51_1: "f32[64, 1024, 8, 64]", arg52_1: "f32[64, 1024, 8, 64]", arg53_1: "f32[512, 512]", arg54_1: "f32[512]", arg55_1: "f32[1536, 512]", arg56_1: "f32[1536, 512]", arg57_1: "f32[512, 1536]", arg58_1: "f32[512]", arg59_1: "f32[512, 512]", arg60_1: "f32[512, 512]", arg61_1: "f32[512, 512]", arg62_1: "f32[64, 1024, 8, 64]", arg63_1: "f32[64, 1024, 8, 64]", arg64_1: "f32[512, 512]", arg65_1: "f32[512]", arg66_1: "f32[1536, 512]", arg67_1: "f32[1536, 512]", arg68_1: "f32[512, 1536]", arg69_1: "f32[512]", arg70_1: "f32[512, 512]", arg71_1: "f32[512, 512]", arg72_1: "f32[512, 512]", arg73_1: "f32[64, 1024, 8, 64]", arg74_1: "f32[64, 1024, 8, 64]", arg75_1: "f32[512, 512]", arg76_1: "f32[512]", arg77_1: "f32[1536, 512]", arg78_1: "f32[1536, 512]", arg79_1: "f32[512, 1536]", arg80_1: "f32[512]", arg81_1: "f32[512, 512]", arg82_1: "f32[512, 512]", arg83_1: "f32[512, 512]", arg84_1: "f32[64, 1024, 8, 64]", arg85_1: "f32[64, 1024, 8, 64]", arg86_1: "f32[512, 512]", arg87_1: "f32[512]", arg88_1: "f32[1536, 512]", arg89_1: "f32[1536, 512]", arg90_1: "f32[512, 1536]", arg91_1: "f32[512]", arg92_1: "f32[32000, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        embedding: "f32[32, 32, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        pow_1: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding, 2)
        mean: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None
        mul_1: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        view: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_1, [1024, 512])
        permute: "f32[512, 512]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "f32[1024, 512]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm, [32, 32, 512]);  mm = None
        view_6: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_1, [32, 32, 8, 64]);  view_1 = None
        view_9: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_6, [32, 32, 8, -1, 2]);  view_6 = None
        view_as_complex: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_9);  view_9 = None
        slice_1: "c64[32, 32]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 1, 33);  arg2_1 = None
        view_11: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32])
        mul_2: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex, view_11);  view_as_complex = None
        view_as_real: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
        view_12: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real, [32, 32, 8, 64]);  view_as_real = None
        permute_3: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        expand: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_3, [32, 8, 32, 64]);  permute_3 = None
        clone: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_14: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone, [256, 32, 64]);  clone = None
        slice_4: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, 32)
        slice_2: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, 32)
        slice_3: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_2, 1, 1, 33);  slice_2 = None
        view_2: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_1, [1024, 512])
        permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "f32[1024, 512]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_1, [32, 32, 512]);  mm_1 = None
        view_7: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_3, [32, 32, 8, 64]);  view_3 = None
        view_10: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_7, [32, 32, 8, -1, 2]);  view_7 = None
        view_as_complex_1: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_10);  view_10 = None
        mul_3: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_1, view_11);  view_as_complex_1 = view_11 = None
        view_as_real_1: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_3);  mul_3 = None
        view_13: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_1, [32, 32, 8, 64]);  view_as_real_1 = None
        copy: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_3, view_13);  slice_3 = view_13 = None
        slice_scatter: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_4, copy, 1, 1, 33);  slice_4 = copy = None
        slice_scatter_1: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg7_1, slice_scatter, 0, 0, 32);  slice_scatter = None
        slice_26: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 32)
        slice_27: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_26, 1, 0, 33);  slice_26 = None
        permute_10: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_27, [0, 2, 1, 3]);  slice_27 = None
        permute_11: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_10, [0, 1, 3, 2]);  permute_10 = None
        expand_2: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_11, [32, 8, 64, 33]);  permute_11 = None
        clone_1: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_15: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_1, [256, 64, 33]);  clone_1 = None
        bmm: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
        view_16: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm, [32, 8, 32, 33]);  bmm = None

        # No stacktrace found for following nodes
        mul_tensor_7: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_16, 1);  view_16 = None
        amax_default_7: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor_7, [-1], True)
        sub_tensor_7: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor_7, amax_default_7);  mul_tensor_7 = amax_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor_7: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor_7, 8.0);  sub_tensor_7 = None
        exp: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor_7);  div_tensor_7 = None
        sum_1: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_3: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_1, [32, 8, 32, 33]);  div_1 = None
        view_17: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_3, [256, 32, 33]);  expand_3 = None
        slice_9: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, 32)
        slice_7: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, 32)
        slice_8: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_7, 1, 1, 33);  slice_7 = None
        view_4: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_1, [1024, 512]);  mul_1 = None
        permute_2: "f32[512, 512]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "f32[1024, 512]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_2, [32, 32, 512]);  mm_2 = None
        view_8: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_5, [32, 32, 8, 64]);  view_5 = None
        copy_1: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_8, view_8);  slice_8 = view_8 = None
        slice_scatter_2: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_9, copy_1, 1, 1, 33);  slice_9 = copy_1 = None
        slice_scatter_3: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg8_1, slice_scatter_2, 0, 0, 32);  slice_scatter_2 = None
        slice_30: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0, 32)
        slice_31: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_30, 1, 0, 33);  slice_30 = None
        permute_13: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_31, [0, 2, 1, 3]);  slice_31 = None
        expand_5: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_13, [32, 8, 33, 64]);  permute_13 = None
        clone_2: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_18: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_2, [256, 33, 64]);  clone_2 = None
        bmm_1: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_17, view_18);  view_17 = view_18 = None
        view_19: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_1, [32, 8, 32, 64]);  bmm_1 = None
        permute_14: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        clone_3: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_20: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_3, [32, 32, -1]);  clone_3 = None
        view_21: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_20, [1024, 512]);  view_20 = None
        permute_15: "f32[512, 512]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_3: "f32[1024, 512]" = torch.ops.aten.mm.default(view_21, permute_15);  view_21 = permute_15 = None
        view_22: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_3, [32, 32, 512]);  mm_3 = None
        add_1: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(embedding, view_22);  embedding = view_22 = None
        pow_2: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_1, 2)
        mean_1: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_2: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_1, rsqrt_1);  rsqrt_1 = None
        mul_5: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg10_1);  mul_4 = arg10_1 = None
        view_23: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_5, [1024, 512])
        permute_16: "f32[512, 1536]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_4: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_23, permute_16);  view_23 = permute_16 = None
        view_24: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_4, [32, 32, 1536]);  mm_4 = None
        neg: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_24)
        exp_1: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg);  neg = None
        add_3: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_2: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_24, add_3);  view_24 = add_3 = None
        view_25: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_5, [1024, 512]);  mul_5 = None
        permute_17: "f32[512, 1536]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_5: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_25, permute_17);  view_25 = permute_17 = None
        view_26: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_5, [32, 32, 1536]);  mm_5 = None
        mul_6: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_2, view_26);  div_2 = view_26 = None
        view_27: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_6, [1024, 1536]);  mul_6 = None
        permute_18: "f32[1536, 512]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_6: "f32[1024, 512]" = torch.ops.aten.mm.default(view_27, permute_18);  view_27 = permute_18 = None
        view_28: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_6, [32, 32, 512]);  mm_6 = None
        add_4: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_1, view_28);  add_1 = view_28 = None
        pow_3: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_4, 2)
        mean_2: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_5: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
        rsqrt_2: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_7: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_4, rsqrt_2);  rsqrt_2 = None
        mul_8: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_7, arg14_1);  mul_7 = arg14_1 = None
        view_29: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_8, [1024, 512])
        permute_19: "f32[512, 512]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_7: "f32[1024, 512]" = torch.ops.aten.mm.default(view_29, permute_19);  view_29 = permute_19 = None
        view_30: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_7, [32, 32, 512]);  mm_7 = None
        view_35: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_30, [32, 32, 8, 64]);  view_30 = None
        view_38: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_35, [32, 32, 8, -1, 2]);  view_35 = None
        view_as_complex_2: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_38);  view_38 = None
        view_40: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32])
        mul_9: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_2, view_40);  view_as_complex_2 = None
        view_as_real_2: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_9);  mul_9 = None
        view_41: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_2, [32, 32, 8, 64]);  view_as_real_2 = None
        permute_22: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        expand_6: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_22, [32, 8, 32, 64]);  permute_22 = None
        clone_4: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_43: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone_4, [256, 32, 64]);  clone_4 = None
        slice_34: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 0, 32)
        slice_32: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 0, 32)
        slice_33: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_32, 1, 1, 33);  slice_32 = None
        view_31: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_8, [1024, 512])
        permute_20: "f32[512, 512]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_8: "f32[1024, 512]" = torch.ops.aten.mm.default(view_31, permute_20);  view_31 = permute_20 = None
        view_32: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_8, [32, 32, 512]);  mm_8 = None
        view_36: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_32, [32, 32, 8, 64]);  view_32 = None
        view_39: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_36, [32, 32, 8, -1, 2]);  view_36 = None
        view_as_complex_3: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_39);  view_39 = None
        mul_10: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_3, view_40);  view_as_complex_3 = view_40 = None
        view_as_real_3: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_10);  mul_10 = None
        view_42: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_3, [32, 32, 8, 64]);  view_as_real_3 = None
        copy_2: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_33, view_42);  slice_33 = view_42 = None
        slice_scatter_4: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_34, copy_2, 1, 1, 33);  slice_34 = copy_2 = None
        slice_scatter_5: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg18_1, slice_scatter_4, 0, 0, 32);  slice_scatter_4 = None
        slice_56: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 32)
        slice_57: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_56, 1, 0, 33);  slice_56 = None
        permute_29: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_57, [0, 2, 1, 3]);  slice_57 = None
        permute_30: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_29, [0, 1, 3, 2]);  permute_29 = None
        expand_8: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_30, [32, 8, 64, 33]);  permute_30 = None
        clone_5: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_44: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_5, [256, 64, 33]);  clone_5 = None
        bmm_2: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
        view_45: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_2, [32, 8, 32, 33]);  bmm_2 = None

        # No stacktrace found for following nodes
        mul_tensor_6: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_45, 1);  view_45 = None
        amax_default_6: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_6: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_6);  mul_tensor_6 = amax_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor_6: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor_6, 8.0);  sub_tensor_6 = None
        exp_2: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor_6);  div_tensor_6 = None
        sum_2: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None
        expand_9: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_4, [32, 8, 32, 33]);  div_4 = None
        view_46: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_9, [256, 32, 33]);  expand_9 = None
        slice_39: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 0, 32)
        slice_37: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 0, 32)
        slice_38: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_37, 1, 1, 33);  slice_37 = None
        view_33: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_8, [1024, 512]);  mul_8 = None
        permute_21: "f32[512, 512]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_9: "f32[1024, 512]" = torch.ops.aten.mm.default(view_33, permute_21);  view_33 = permute_21 = None
        view_34: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_9, [32, 32, 512]);  mm_9 = None
        view_37: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_34, [32, 32, 8, 64]);  view_34 = None
        copy_3: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_38, view_37);  slice_38 = view_37 = None
        slice_scatter_6: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_39, copy_3, 1, 1, 33);  slice_39 = copy_3 = None
        slice_scatter_7: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg19_1, slice_scatter_6, 0, 0, 32);  slice_scatter_6 = None
        slice_60: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 0, 32)
        slice_61: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_60, 1, 0, 33);  slice_60 = None
        permute_32: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_61, [0, 2, 1, 3]);  slice_61 = None
        expand_11: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_32, [32, 8, 33, 64]);  permute_32 = None
        clone_6: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_47: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_6, [256, 33, 64]);  clone_6 = None
        bmm_3: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_46, view_47);  view_46 = view_47 = None
        view_48: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_3, [32, 8, 32, 64]);  bmm_3 = None
        permute_33: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_7: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_49: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_7, [32, 32, -1]);  clone_7 = None
        view_50: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_49, [1024, 512]);  view_49 = None
        permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_10: "f32[1024, 512]" = torch.ops.aten.mm.default(view_50, permute_34);  view_50 = permute_34 = None
        view_51: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_10, [32, 32, 512]);  mm_10 = None
        add_6: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_4, view_51);  add_4 = view_51 = None
        pow_4: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_3: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_7: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_3: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_11: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_3);  rsqrt_3 = None
        mul_12: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None
        view_52: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_12, [1024, 512])
        permute_35: "f32[512, 1536]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_11: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_52, permute_35);  view_52 = permute_35 = None
        view_53: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_11, [32, 32, 1536]);  mm_11 = None
        neg_1: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_53)
        exp_3: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_8: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_5: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_53, add_8);  view_53 = add_8 = None
        view_54: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_12, [1024, 512]);  mul_12 = None
        permute_36: "f32[512, 1536]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_12: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_54, permute_36);  view_54 = permute_36 = None
        view_55: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_12, [32, 32, 1536]);  mm_12 = None
        mul_13: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_5, view_55);  div_5 = view_55 = None
        view_56: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_13, [1024, 1536]);  mul_13 = None
        permute_37: "f32[1536, 512]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_13: "f32[1024, 512]" = torch.ops.aten.mm.default(view_56, permute_37);  view_56 = permute_37 = None
        view_57: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_13, [32, 32, 512]);  mm_13 = None
        add_9: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_6, view_57);  add_6 = view_57 = None
        pow_5: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_4: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_10: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
        rsqrt_4: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_14: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_4);  rsqrt_4 = None
        mul_15: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_14, arg25_1);  mul_14 = arg25_1 = None
        view_58: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_15, [1024, 512])
        permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_14: "f32[1024, 512]" = torch.ops.aten.mm.default(view_58, permute_38);  view_58 = permute_38 = None
        view_59: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_14, [32, 32, 512]);  mm_14 = None
        view_64: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_59, [32, 32, 8, 64]);  view_59 = None
        view_67: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_64, [32, 32, 8, -1, 2]);  view_64 = None
        view_as_complex_4: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_67);  view_67 = None
        view_69: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32])
        mul_16: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_4, view_69);  view_as_complex_4 = None
        view_as_real_4: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_16);  mul_16 = None
        view_70: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_4, [32, 32, 8, 64]);  view_as_real_4 = None
        permute_41: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        expand_12: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_41, [32, 8, 32, 64]);  permute_41 = None
        clone_8: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_72: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone_8, [256, 32, 64]);  clone_8 = None
        slice_64: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg29_1, 0, 0, 32)
        slice_62: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg29_1, 0, 0, 32)
        slice_63: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_62, 1, 1, 33);  slice_62 = None
        view_60: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_15, [1024, 512])
        permute_39: "f32[512, 512]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_15: "f32[1024, 512]" = torch.ops.aten.mm.default(view_60, permute_39);  view_60 = permute_39 = None
        view_61: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_15, [32, 32, 512]);  mm_15 = None
        view_65: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_61, [32, 32, 8, 64]);  view_61 = None
        view_68: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_65, [32, 32, 8, -1, 2]);  view_65 = None
        view_as_complex_5: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_68);  view_68 = None
        mul_17: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_5, view_69);  view_as_complex_5 = view_69 = None
        view_as_real_5: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_17);  mul_17 = None
        view_71: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_5, [32, 32, 8, 64]);  view_as_real_5 = None
        copy_4: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_63, view_71);  slice_63 = view_71 = None
        slice_scatter_8: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_64, copy_4, 1, 1, 33);  slice_64 = copy_4 = None
        slice_scatter_9: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg29_1, slice_scatter_8, 0, 0, 32);  slice_scatter_8 = None
        slice_86: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_9, 0, 0, 32)
        slice_87: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_86, 1, 0, 33);  slice_86 = None
        permute_48: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_87, [0, 2, 1, 3]);  slice_87 = None
        permute_49: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_14: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_49, [32, 8, 64, 33]);  permute_49 = None
        clone_9: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_73: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_9, [256, 64, 33]);  clone_9 = None
        bmm_4: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_72, view_73);  view_72 = view_73 = None
        view_74: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_4, [32, 8, 32, 33]);  bmm_4 = None

        # No stacktrace found for following nodes
        mul_tensor_5: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_74, 1);  view_74 = None
        amax_default_5: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor_5, [-1], True)
        sub_tensor_5: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor_5, amax_default_5);  mul_tensor_5 = amax_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor_5: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor_5, 8.0);  sub_tensor_5 = None
        exp_4: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor_5);  div_tensor_5 = None
        sum_3: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_7: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None
        expand_15: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_7, [32, 8, 32, 33]);  div_7 = None
        view_75: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_15, [256, 32, 33]);  expand_15 = None
        slice_69: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg30_1, 0, 0, 32)
        slice_67: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg30_1, 0, 0, 32)
        slice_68: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_67, 1, 1, 33);  slice_67 = None
        view_62: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_15, [1024, 512]);  mul_15 = None
        permute_40: "f32[512, 512]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_16: "f32[1024, 512]" = torch.ops.aten.mm.default(view_62, permute_40);  view_62 = permute_40 = None
        view_63: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_16, [32, 32, 512]);  mm_16 = None
        view_66: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_63, [32, 32, 8, 64]);  view_63 = None
        copy_5: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_68, view_66);  slice_68 = view_66 = None
        slice_scatter_10: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_69, copy_5, 1, 1, 33);  slice_69 = copy_5 = None
        slice_scatter_11: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg30_1, slice_scatter_10, 0, 0, 32);  slice_scatter_10 = None
        slice_90: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 0, 32)
        slice_91: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_90, 1, 0, 33);  slice_90 = None
        permute_51: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_91, [0, 2, 1, 3]);  slice_91 = None
        expand_17: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_51, [32, 8, 33, 64]);  permute_51 = None
        clone_10: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_76: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_10, [256, 33, 64]);  clone_10 = None
        bmm_5: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
        view_77: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_5, [32, 8, 32, 64]);  bmm_5 = None
        permute_52: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        clone_11: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_78: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_11, [32, 32, -1]);  clone_11 = None
        view_79: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_78, [1024, 512]);  view_78 = None
        permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_17: "f32[1024, 512]" = torch.ops.aten.mm.default(view_79, permute_53);  view_79 = permute_53 = None
        view_80: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_17, [32, 32, 512]);  mm_17 = None
        add_11: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_9, view_80);  add_9 = view_80 = None
        pow_6: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_5: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        add_12: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
        rsqrt_5: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_18: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_5);  rsqrt_5 = None
        mul_19: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_18, arg32_1);  mul_18 = arg32_1 = None
        view_81: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_19, [1024, 512])
        permute_54: "f32[512, 1536]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_18: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_81, permute_54);  view_81 = permute_54 = None
        view_82: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_18, [32, 32, 1536]);  mm_18 = None
        neg_2: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_82)
        exp_5: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_13: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_8: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_82, add_13);  view_82 = add_13 = None
        view_83: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_19, [1024, 512]);  mul_19 = None
        permute_55: "f32[512, 1536]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_19: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_83, permute_55);  view_83 = permute_55 = None
        view_84: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_19, [32, 32, 1536]);  mm_19 = None
        mul_20: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_8, view_84);  div_8 = view_84 = None
        view_85: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_20, [1024, 1536]);  mul_20 = None
        permute_56: "f32[1536, 512]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_20: "f32[1024, 512]" = torch.ops.aten.mm.default(view_85, permute_56);  view_85 = permute_56 = None
        view_86: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_20, [32, 32, 512]);  mm_20 = None
        add_14: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_11, view_86);  add_11 = view_86 = None
        pow_7: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_6: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_15: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-05);  mean_6 = None
        rsqrt_6: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_21: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_6);  rsqrt_6 = None
        mul_22: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_21, arg36_1);  mul_21 = arg36_1 = None
        view_87: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_22, [1024, 512])
        permute_57: "f32[512, 512]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_21: "f32[1024, 512]" = torch.ops.aten.mm.default(view_87, permute_57);  view_87 = permute_57 = None
        view_88: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_21, [32, 32, 512]);  mm_21 = None
        view_93: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_88, [32, 32, 8, 64]);  view_88 = None
        view_96: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_93, [32, 32, 8, -1, 2]);  view_93 = None
        view_as_complex_6: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_96);  view_96 = None
        view_98: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32])
        mul_23: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_6, view_98);  view_as_complex_6 = None
        view_as_real_6: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_23);  mul_23 = None
        view_99: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_6, [32, 32, 8, 64]);  view_as_real_6 = None
        permute_60: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        expand_18: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_60, [32, 8, 32, 64]);  permute_60 = None
        clone_12: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_101: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone_12, [256, 32, 64]);  clone_12 = None
        slice_94: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg40_1, 0, 0, 32)
        slice_92: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg40_1, 0, 0, 32)
        slice_93: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_92, 1, 1, 33);  slice_92 = None
        view_89: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_22, [1024, 512])
        permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_22: "f32[1024, 512]" = torch.ops.aten.mm.default(view_89, permute_58);  view_89 = permute_58 = None
        view_90: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_22, [32, 32, 512]);  mm_22 = None
        view_94: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_90, [32, 32, 8, 64]);  view_90 = None
        view_97: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_94, [32, 32, 8, -1, 2]);  view_94 = None
        view_as_complex_7: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_97);  view_97 = None
        mul_24: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_7, view_98);  view_as_complex_7 = view_98 = None
        view_as_real_7: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_24);  mul_24 = None
        view_100: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_7, [32, 32, 8, 64]);  view_as_real_7 = None
        copy_6: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_93, view_100);  slice_93 = view_100 = None
        slice_scatter_12: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_94, copy_6, 1, 1, 33);  slice_94 = copy_6 = None
        slice_scatter_13: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg40_1, slice_scatter_12, 0, 0, 32);  slice_scatter_12 = None
        slice_116: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 0, 32)
        slice_117: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_116, 1, 0, 33);  slice_116 = None
        permute_67: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_117, [0, 2, 1, 3]);  slice_117 = None
        permute_68: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_67, [0, 1, 3, 2]);  permute_67 = None
        expand_20: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_68, [32, 8, 64, 33]);  permute_68 = None
        clone_13: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_102: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_13, [256, 64, 33]);  clone_13 = None
        bmm_6: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
        view_103: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_6, [32, 8, 32, 33]);  bmm_6 = None

        # No stacktrace found for following nodes
        mul_tensor_4: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_103, 1);  view_103 = None
        amax_default_4: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor_4, [-1], True)
        sub_tensor_4: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_4);  mul_tensor_4 = amax_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor_4: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor_4, 8.0);  sub_tensor_4 = None
        exp_6: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor_4);  div_tensor_4 = None
        sum_4: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None
        expand_21: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_10, [32, 8, 32, 33]);  div_10 = None
        view_104: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_21, [256, 32, 33]);  expand_21 = None
        slice_99: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg41_1, 0, 0, 32)
        slice_97: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg41_1, 0, 0, 32)
        slice_98: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_97, 1, 1, 33);  slice_97 = None
        view_91: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_22, [1024, 512]);  mul_22 = None
        permute_59: "f32[512, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_23: "f32[1024, 512]" = torch.ops.aten.mm.default(view_91, permute_59);  view_91 = permute_59 = None
        view_92: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_23, [32, 32, 512]);  mm_23 = None
        view_95: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_92, [32, 32, 8, 64]);  view_92 = None
        copy_7: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_98, view_95);  slice_98 = view_95 = None
        slice_scatter_14: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_99, copy_7, 1, 1, 33);  slice_99 = copy_7 = None
        slice_scatter_15: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg41_1, slice_scatter_14, 0, 0, 32);  slice_scatter_14 = None
        slice_120: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_15, 0, 0, 32)
        slice_121: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_120, 1, 0, 33);  slice_120 = None
        permute_70: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_121, [0, 2, 1, 3]);  slice_121 = None
        expand_23: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_70, [32, 8, 33, 64]);  permute_70 = None
        clone_14: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_105: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_14, [256, 33, 64]);  clone_14 = None
        bmm_7: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_104, view_105);  view_104 = view_105 = None
        view_106: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_7, [32, 8, 32, 64]);  bmm_7 = None
        permute_71: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_15: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_107: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_15, [32, 32, -1]);  clone_15 = None
        view_108: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_107, [1024, 512]);  view_107 = None
        permute_72: "f32[512, 512]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_24: "f32[1024, 512]" = torch.ops.aten.mm.default(view_108, permute_72);  view_108 = permute_72 = None
        view_109: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_24, [32, 32, 512]);  mm_24 = None
        add_16: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_14, view_109);  add_14 = view_109 = None
        pow_8: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_7: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_17: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-05);  mean_7 = None
        rsqrt_7: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_25: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_7);  rsqrt_7 = None
        mul_26: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_25, arg43_1);  mul_25 = arg43_1 = None
        view_110: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_26, [1024, 512])
        permute_73: "f32[512, 1536]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_25: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_110, permute_73);  view_110 = permute_73 = None
        view_111: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_25, [32, 32, 1536]);  mm_25 = None
        neg_3: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_111)
        exp_7: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_18: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_11: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_111, add_18);  view_111 = add_18 = None
        view_112: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_26, [1024, 512]);  mul_26 = None
        permute_74: "f32[512, 1536]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_26: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_112, permute_74);  view_112 = permute_74 = None
        view_113: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_26, [32, 32, 1536]);  mm_26 = None
        mul_27: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_11, view_113);  div_11 = view_113 = None
        view_114: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_27, [1024, 1536]);  mul_27 = None
        permute_75: "f32[1536, 512]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_27: "f32[1024, 512]" = torch.ops.aten.mm.default(view_114, permute_75);  view_114 = permute_75 = None
        view_115: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_27, [32, 32, 512]);  mm_27 = None
        add_19: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_16, view_115);  add_16 = view_115 = None
        pow_9: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_8: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_20: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-05);  mean_8 = None
        rsqrt_8: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_28: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_8);  rsqrt_8 = None
        mul_29: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_28, arg47_1);  mul_28 = arg47_1 = None
        view_116: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_29, [1024, 512])
        permute_76: "f32[512, 512]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_28: "f32[1024, 512]" = torch.ops.aten.mm.default(view_116, permute_76);  view_116 = permute_76 = None
        view_117: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_28, [32, 32, 512]);  mm_28 = None
        view_122: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_117, [32, 32, 8, 64]);  view_117 = None
        view_125: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_122, [32, 32, 8, -1, 2]);  view_122 = None
        view_as_complex_8: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_125);  view_125 = None
        view_127: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32])
        mul_30: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_8, view_127);  view_as_complex_8 = None
        view_as_real_8: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_30);  mul_30 = None
        view_128: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_8, [32, 32, 8, 64]);  view_as_real_8 = None
        permute_79: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        expand_24: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_79, [32, 8, 32, 64]);  permute_79 = None
        clone_16: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_130: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone_16, [256, 32, 64]);  clone_16 = None
        slice_124: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg51_1, 0, 0, 32)
        slice_122: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg51_1, 0, 0, 32)
        slice_123: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_122, 1, 1, 33);  slice_122 = None
        view_118: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_29, [1024, 512])
        permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_29: "f32[1024, 512]" = torch.ops.aten.mm.default(view_118, permute_77);  view_118 = permute_77 = None
        view_119: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_29, [32, 32, 512]);  mm_29 = None
        view_123: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_119, [32, 32, 8, 64]);  view_119 = None
        view_126: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_123, [32, 32, 8, -1, 2]);  view_123 = None
        view_as_complex_9: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_126);  view_126 = None
        mul_31: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_9, view_127);  view_as_complex_9 = view_127 = None
        view_as_real_9: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_31);  mul_31 = None
        view_129: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_9, [32, 32, 8, 64]);  view_as_real_9 = None
        copy_8: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_123, view_129);  slice_123 = view_129 = None
        slice_scatter_16: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_124, copy_8, 1, 1, 33);  slice_124 = copy_8 = None
        slice_scatter_17: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg51_1, slice_scatter_16, 0, 0, 32);  slice_scatter_16 = None
        slice_146: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 0, 32)
        slice_147: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_146, 1, 0, 33);  slice_146 = None
        permute_86: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_147, [0, 2, 1, 3]);  slice_147 = None
        permute_87: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_86, [0, 1, 3, 2]);  permute_86 = None
        expand_26: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_87, [32, 8, 64, 33]);  permute_87 = None
        clone_17: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_131: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_17, [256, 64, 33]);  clone_17 = None
        bmm_8: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
        view_132: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_8, [32, 8, 32, 33]);  bmm_8 = None

        # No stacktrace found for following nodes
        mul_tensor_3: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_132, 1);  view_132 = None
        amax_default_3: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor_3, [-1], True)
        sub_tensor_3: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor_3, amax_default_3);  mul_tensor_3 = amax_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor_3: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor_3, 8.0);  sub_tensor_3 = None
        exp_8: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor_3);  div_tensor_3 = None
        sum_5: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_13: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None
        expand_27: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_13, [32, 8, 32, 33]);  div_13 = None
        view_133: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_27, [256, 32, 33]);  expand_27 = None
        slice_129: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg52_1, 0, 0, 32)
        slice_127: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg52_1, 0, 0, 32)
        slice_128: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_127, 1, 1, 33);  slice_127 = None
        view_120: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_29, [1024, 512]);  mul_29 = None
        permute_78: "f32[512, 512]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_30: "f32[1024, 512]" = torch.ops.aten.mm.default(view_120, permute_78);  view_120 = permute_78 = None
        view_121: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_30, [32, 32, 512]);  mm_30 = None
        view_124: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_121, [32, 32, 8, 64]);  view_121 = None
        copy_9: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_128, view_124);  slice_128 = view_124 = None
        slice_scatter_18: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_129, copy_9, 1, 1, 33);  slice_129 = copy_9 = None
        slice_scatter_19: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg52_1, slice_scatter_18, 0, 0, 32);  slice_scatter_18 = None
        slice_150: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 0, 32)
        slice_151: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_150, 1, 0, 33);  slice_150 = None
        permute_89: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_151, [0, 2, 1, 3]);  slice_151 = None
        expand_29: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_89, [32, 8, 33, 64]);  permute_89 = None
        clone_18: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_134: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_18, [256, 33, 64]);  clone_18 = None
        bmm_9: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
        view_135: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_9, [32, 8, 32, 64]);  bmm_9 = None
        permute_90: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        clone_19: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_136: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_19, [32, 32, -1]);  clone_19 = None
        view_137: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_136, [1024, 512]);  view_136 = None
        permute_91: "f32[512, 512]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_31: "f32[1024, 512]" = torch.ops.aten.mm.default(view_137, permute_91);  view_137 = permute_91 = None
        view_138: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_31, [32, 32, 512]);  mm_31 = None
        add_21: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_19, view_138);  add_19 = view_138 = None
        pow_10: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
        mean_9: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_22: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-05);  mean_9 = None
        rsqrt_9: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_32: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_9);  rsqrt_9 = None
        mul_33: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_32, arg54_1);  mul_32 = arg54_1 = None
        view_139: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_33, [1024, 512])
        permute_92: "f32[512, 1536]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_32: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_139, permute_92);  view_139 = permute_92 = None
        view_140: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_32, [32, 32, 1536]);  mm_32 = None
        neg_4: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_140)
        exp_9: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_23: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_14: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_140, add_23);  view_140 = add_23 = None
        view_141: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_33, [1024, 512]);  mul_33 = None
        permute_93: "f32[512, 1536]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_33: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_141, permute_93);  view_141 = permute_93 = None
        view_142: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_33, [32, 32, 1536]);  mm_33 = None
        mul_34: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_14, view_142);  div_14 = view_142 = None
        view_143: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_34, [1024, 1536]);  mul_34 = None
        permute_94: "f32[1536, 512]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_34: "f32[1024, 512]" = torch.ops.aten.mm.default(view_143, permute_94);  view_143 = permute_94 = None
        view_144: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_34, [32, 32, 512]);  mm_34 = None
        add_24: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_21, view_144);  add_21 = view_144 = None
        pow_11: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_10: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_25: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-05);  mean_10 = None
        rsqrt_10: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_35: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_10);  rsqrt_10 = None
        mul_36: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        view_145: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_36, [1024, 512])
        permute_95: "f32[512, 512]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_35: "f32[1024, 512]" = torch.ops.aten.mm.default(view_145, permute_95);  view_145 = permute_95 = None
        view_146: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_35, [32, 32, 512]);  mm_35 = None
        view_151: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_146, [32, 32, 8, 64]);  view_146 = None
        view_154: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_151, [32, 32, 8, -1, 2]);  view_151 = None
        view_as_complex_10: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_154);  view_154 = None
        view_156: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32])
        mul_37: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_10, view_156);  view_as_complex_10 = None
        view_as_real_10: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_37);  mul_37 = None
        view_157: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_10, [32, 32, 8, 64]);  view_as_real_10 = None
        permute_98: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        expand_30: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_98, [32, 8, 32, 64]);  permute_98 = None
        clone_20: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_159: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone_20, [256, 32, 64]);  clone_20 = None
        slice_154: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg62_1, 0, 0, 32)
        slice_152: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg62_1, 0, 0, 32)
        slice_153: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_152, 1, 1, 33);  slice_152 = None
        view_147: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_36, [1024, 512])
        permute_96: "f32[512, 512]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_36: "f32[1024, 512]" = torch.ops.aten.mm.default(view_147, permute_96);  view_147 = permute_96 = None
        view_148: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_36, [32, 32, 512]);  mm_36 = None
        view_152: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_148, [32, 32, 8, 64]);  view_148 = None
        view_155: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_152, [32, 32, 8, -1, 2]);  view_152 = None
        view_as_complex_11: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_155);  view_155 = None
        mul_38: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_11, view_156);  view_as_complex_11 = view_156 = None
        view_as_real_11: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_38);  mul_38 = None
        view_158: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_11, [32, 32, 8, 64]);  view_as_real_11 = None
        copy_10: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_153, view_158);  slice_153 = view_158 = None
        slice_scatter_20: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_154, copy_10, 1, 1, 33);  slice_154 = copy_10 = None
        slice_scatter_21: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg62_1, slice_scatter_20, 0, 0, 32);  slice_scatter_20 = None
        slice_176: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_21, 0, 0, 32)
        slice_177: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_176, 1, 0, 33);  slice_176 = None
        permute_105: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_177, [0, 2, 1, 3]);  slice_177 = None
        permute_106: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_105, [0, 1, 3, 2]);  permute_105 = None
        expand_32: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_106, [32, 8, 64, 33]);  permute_106 = None
        clone_21: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_160: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_21, [256, 64, 33]);  clone_21 = None
        bmm_10: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_159, view_160);  view_159 = view_160 = None
        view_161: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_10, [32, 8, 32, 33]);  bmm_10 = None

        # No stacktrace found for following nodes
        mul_tensor_2: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_161, 1);  view_161 = None
        amax_default_2: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor_2, [-1], True)
        sub_tensor_2: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_2);  mul_tensor_2 = amax_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor_2: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor_2, 8.0);  sub_tensor_2 = None
        exp_10: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor_2);  div_tensor_2 = None
        sum_6: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_16: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None
        expand_33: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_16, [32, 8, 32, 33]);  div_16 = None
        view_162: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_33, [256, 32, 33]);  expand_33 = None
        slice_159: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg63_1, 0, 0, 32)
        slice_157: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg63_1, 0, 0, 32)
        slice_158: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_157, 1, 1, 33);  slice_157 = None
        view_149: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_36, [1024, 512]);  mul_36 = None
        permute_97: "f32[512, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_37: "f32[1024, 512]" = torch.ops.aten.mm.default(view_149, permute_97);  view_149 = permute_97 = None
        view_150: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_37, [32, 32, 512]);  mm_37 = None
        view_153: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_150, [32, 32, 8, 64]);  view_150 = None
        copy_11: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_158, view_153);  slice_158 = view_153 = None
        slice_scatter_22: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_159, copy_11, 1, 1, 33);  slice_159 = copy_11 = None
        slice_scatter_23: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg63_1, slice_scatter_22, 0, 0, 32);  slice_scatter_22 = None
        slice_180: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 0, 32)
        slice_181: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_180, 1, 0, 33);  slice_180 = None
        permute_108: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_181, [0, 2, 1, 3]);  slice_181 = None
        expand_35: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_108, [32, 8, 33, 64]);  permute_108 = None
        clone_22: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_163: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_22, [256, 33, 64]);  clone_22 = None
        bmm_11: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_162, view_163);  view_162 = view_163 = None
        view_164: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_11, [32, 8, 32, 64]);  bmm_11 = None
        permute_109: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        clone_23: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        view_165: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_23, [32, 32, -1]);  clone_23 = None
        view_166: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_165, [1024, 512]);  view_165 = None
        permute_110: "f32[512, 512]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_38: "f32[1024, 512]" = torch.ops.aten.mm.default(view_166, permute_110);  view_166 = permute_110 = None
        view_167: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_38, [32, 32, 512]);  mm_38 = None
        add_26: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_24, view_167);  add_24 = view_167 = None
        pow_12: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
        mean_11: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        add_27: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-05);  mean_11 = None
        rsqrt_11: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_39: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_11);  rsqrt_11 = None
        mul_40: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_39, arg65_1);  mul_39 = arg65_1 = None
        view_168: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_40, [1024, 512])
        permute_111: "f32[512, 1536]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_39: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_168, permute_111);  view_168 = permute_111 = None
        view_169: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_39, [32, 32, 1536]);  mm_39 = None
        neg_5: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_169)
        exp_11: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_28: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_17: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_169, add_28);  view_169 = add_28 = None
        view_170: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_40, [1024, 512]);  mul_40 = None
        permute_112: "f32[512, 1536]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_40: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_170, permute_112);  view_170 = permute_112 = None
        view_171: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_40, [32, 32, 1536]);  mm_40 = None
        mul_41: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_17, view_171);  div_17 = view_171 = None
        view_172: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_41, [1024, 1536]);  mul_41 = None
        permute_113: "f32[1536, 512]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_41: "f32[1024, 512]" = torch.ops.aten.mm.default(view_172, permute_113);  view_172 = permute_113 = None
        view_173: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_41, [32, 32, 512]);  mm_41 = None
        add_29: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_26, view_173);  add_26 = view_173 = None
        pow_13: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_29, 2)
        mean_12: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_30: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-05);  mean_12 = None
        rsqrt_12: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_42: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_12);  rsqrt_12 = None
        mul_43: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        view_174: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_43, [1024, 512])
        permute_114: "f32[512, 512]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_42: "f32[1024, 512]" = torch.ops.aten.mm.default(view_174, permute_114);  view_174 = permute_114 = None
        view_175: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_42, [32, 32, 512]);  mm_42 = None
        view_180: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_175, [32, 32, 8, 64]);  view_175 = None
        view_183: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_180, [32, 32, 8, -1, 2]);  view_180 = None
        view_as_complex_12: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_183);  view_183 = None
        view_185: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32])
        mul_44: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_12, view_185);  view_as_complex_12 = None
        view_as_real_12: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_44);  mul_44 = None
        view_186: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_12, [32, 32, 8, 64]);  view_as_real_12 = None
        permute_117: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        expand_36: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_117, [32, 8, 32, 64]);  permute_117 = None
        clone_24: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_188: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone_24, [256, 32, 64]);  clone_24 = None
        slice_184: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg73_1, 0, 0, 32)
        slice_182: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg73_1, 0, 0, 32)
        slice_183: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_182, 1, 1, 33);  slice_182 = None
        view_176: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_43, [1024, 512])
        permute_115: "f32[512, 512]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_43: "f32[1024, 512]" = torch.ops.aten.mm.default(view_176, permute_115);  view_176 = permute_115 = None
        view_177: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_43, [32, 32, 512]);  mm_43 = None
        view_181: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_177, [32, 32, 8, 64]);  view_177 = None
        view_184: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_181, [32, 32, 8, -1, 2]);  view_181 = None
        view_as_complex_13: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_184);  view_184 = None
        mul_45: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_13, view_185);  view_as_complex_13 = view_185 = None
        view_as_real_13: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_45);  mul_45 = None
        view_187: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_13, [32, 32, 8, 64]);  view_as_real_13 = None
        copy_12: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_183, view_187);  slice_183 = view_187 = None
        slice_scatter_24: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_184, copy_12, 1, 1, 33);  slice_184 = copy_12 = None
        slice_scatter_25: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg73_1, slice_scatter_24, 0, 0, 32);  slice_scatter_24 = None
        slice_206: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 0, 32)
        slice_207: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_206, 1, 0, 33);  slice_206 = None
        permute_124: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_207, [0, 2, 1, 3]);  slice_207 = None
        permute_125: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        expand_38: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_125, [32, 8, 64, 33]);  permute_125 = None
        clone_25: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_189: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_25, [256, 64, 33]);  clone_25 = None
        bmm_12: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
        view_190: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_12, [32, 8, 32, 33]);  bmm_12 = None

        # No stacktrace found for following nodes
        mul_tensor_1: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_190, 1);  view_190 = None
        amax_default_1: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor_1: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default_1);  mul_tensor_1 = amax_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor_1: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor_1, 8.0);  sub_tensor_1 = None
        exp_12: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor_1);  div_tensor_1 = None
        sum_7: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_19: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_12, sum_7);  exp_12 = sum_7 = None
        expand_39: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_19, [32, 8, 32, 33]);  div_19 = None
        view_191: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_39, [256, 32, 33]);  expand_39 = None
        slice_189: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg74_1, 0, 0, 32)
        slice_187: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg74_1, 0, 0, 32)
        slice_188: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_187, 1, 1, 33);  slice_187 = None
        view_178: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_43, [1024, 512]);  mul_43 = None
        permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_44: "f32[1024, 512]" = torch.ops.aten.mm.default(view_178, permute_116);  view_178 = permute_116 = None
        view_179: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_44, [32, 32, 512]);  mm_44 = None
        view_182: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_179, [32, 32, 8, 64]);  view_179 = None
        copy_13: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_188, view_182);  slice_188 = view_182 = None
        slice_scatter_26: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_189, copy_13, 1, 1, 33);  slice_189 = copy_13 = None
        slice_scatter_27: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg74_1, slice_scatter_26, 0, 0, 32);  slice_scatter_26 = None
        slice_210: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_27, 0, 0, 32)
        slice_211: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_210, 1, 0, 33);  slice_210 = None
        permute_127: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_211, [0, 2, 1, 3]);  slice_211 = None
        expand_41: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_127, [32, 8, 33, 64]);  permute_127 = None
        clone_26: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_192: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_26, [256, 33, 64]);  clone_26 = None
        bmm_13: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_191, view_192);  view_191 = view_192 = None
        view_193: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_13, [32, 8, 32, 64]);  bmm_13 = None
        permute_128: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        clone_27: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_194: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_27, [32, 32, -1]);  clone_27 = None
        view_195: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_194, [1024, 512]);  view_194 = None
        permute_129: "f32[512, 512]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_45: "f32[1024, 512]" = torch.ops.aten.mm.default(view_195, permute_129);  view_195 = permute_129 = None
        view_196: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_45, [32, 32, 512]);  mm_45 = None
        add_31: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_29, view_196);  add_29 = view_196 = None
        pow_14: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
        mean_13: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_32: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-05);  mean_13 = None
        rsqrt_13: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_46: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_13);  rsqrt_13 = None
        mul_47: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_46, arg76_1);  mul_46 = arg76_1 = None
        view_197: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_47, [1024, 512])
        permute_130: "f32[512, 1536]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_46: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_197, permute_130);  view_197 = permute_130 = None
        view_198: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_46, [32, 32, 1536]);  mm_46 = None
        neg_6: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_198)
        exp_13: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_33: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_20: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_198, add_33);  view_198 = add_33 = None
        view_199: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_47, [1024, 512]);  mul_47 = None
        permute_131: "f32[512, 1536]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_47: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_199, permute_131);  view_199 = permute_131 = None
        view_200: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_47, [32, 32, 1536]);  mm_47 = None
        mul_48: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_20, view_200);  div_20 = view_200 = None
        view_201: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_48, [1024, 1536]);  mul_48 = None
        permute_132: "f32[1536, 512]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_48: "f32[1024, 512]" = torch.ops.aten.mm.default(view_201, permute_132);  view_201 = permute_132 = None
        view_202: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_48, [32, 32, 512]);  mm_48 = None
        add_34: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_31, view_202);  add_31 = view_202 = None
        pow_15: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_14: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_35: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-05);  mean_14 = None
        rsqrt_14: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_49: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_14);  rsqrt_14 = None
        mul_50: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_49, arg80_1);  mul_49 = arg80_1 = None
        view_203: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_50, [1024, 512])
        permute_133: "f32[512, 512]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_49: "f32[1024, 512]" = torch.ops.aten.mm.default(view_203, permute_133);  view_203 = permute_133 = None
        view_204: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_49, [32, 32, 512]);  mm_49 = None
        view_209: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_204, [32, 32, 8, 64]);  view_204 = None
        view_212: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_209, [32, 32, 8, -1, 2]);  view_209 = None
        view_as_complex_14: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_212);  view_212 = None
        view_214: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, [1, 32, 1, 32]);  slice_1 = None
        mul_51: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_14, view_214);  view_as_complex_14 = None
        view_as_real_14: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_51);  mul_51 = None
        view_215: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_14, [32, 32, 8, 64]);  view_as_real_14 = None
        permute_136: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        expand_42: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_136, [32, 8, 32, 64]);  permute_136 = None
        clone_28: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_217: "f32[256, 32, 64]" = torch.ops.aten.reshape.default(clone_28, [256, 32, 64]);  clone_28 = None
        slice_214: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg84_1, 0, 0, 32)
        slice_212: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg84_1, 0, 0, 32)
        slice_213: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_212, 1, 1, 33);  slice_212 = None
        view_205: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_50, [1024, 512])
        permute_134: "f32[512, 512]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_50: "f32[1024, 512]" = torch.ops.aten.mm.default(view_205, permute_134);  view_205 = permute_134 = None
        view_206: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_50, [32, 32, 512]);  mm_50 = None
        view_210: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_206, [32, 32, 8, 64]);  view_206 = None
        view_213: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.reshape.default(view_210, [32, 32, 8, -1, 2]);  view_210 = None
        view_as_complex_15: "c64[32, 32, 8, 32]" = torch.ops.aten.view_as_complex.default(view_213);  view_213 = None
        mul_52: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_15, view_214);  view_as_complex_15 = view_214 = None
        view_as_real_15: "f32[32, 32, 8, 32, 2]" = torch.ops.aten.view_as_real.default(mul_52);  mul_52 = None
        view_216: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_as_real_15, [32, 32, 8, 64]);  view_as_real_15 = None
        copy_14: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_213, view_216);  slice_213 = view_216 = None
        slice_scatter_28: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_214, copy_14, 1, 1, 33);  slice_214 = copy_14 = None
        slice_scatter_29: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg84_1, slice_scatter_28, 0, 0, 32);  slice_scatter_28 = None
        slice_236: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 0, 32)
        slice_237: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_236, 1, 0, 33);  slice_236 = None
        permute_143: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_237, [0, 2, 1, 3]);  slice_237 = None
        permute_144: "f32[32, 8, 64, 33]" = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2]);  permute_143 = None
        expand_44: "f32[32, 8, 64, 33]" = torch.ops.aten.expand.default(permute_144, [32, 8, 64, 33]);  permute_144 = None
        clone_29: "f32[32, 8, 64, 33]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_218: "f32[256, 64, 33]" = torch.ops.aten.reshape.default(clone_29, [256, 64, 33]);  clone_29 = None
        bmm_14: "f32[256, 32, 33]" = torch.ops.aten.bmm.default(view_217, view_218);  view_217 = view_218 = None
        view_219: "f32[32, 8, 32, 33]" = torch.ops.aten.reshape.default(bmm_14, [32, 8, 32, 33]);  bmm_14 = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_219, 1);  view_219 = None
        amax_default: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        div_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor, 8.0);  sub_tensor = None
        exp_14: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        sum_8: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_22: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_14, sum_8);  exp_14 = sum_8 = None
        expand_45: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_22, [32, 8, 32, 33]);  div_22 = None
        view_220: "f32[256, 32, 33]" = torch.ops.aten.reshape.default(expand_45, [256, 32, 33]);  expand_45 = None
        slice_219: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg85_1, 0, 0, 32)
        slice_217: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg85_1, 0, 0, 32)
        slice_218: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_217, 1, 1, 33);  slice_217 = None
        view_207: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_50, [1024, 512]);  mul_50 = None
        permute_135: "f32[512, 512]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_51: "f32[1024, 512]" = torch.ops.aten.mm.default(view_207, permute_135);  view_207 = permute_135 = None
        view_208: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_51, [32, 32, 512]);  mm_51 = None
        view_211: "f32[32, 32, 8, 64]" = torch.ops.aten.reshape.default(view_208, [32, 32, 8, 64]);  view_208 = None
        copy_15: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_218, view_211);  slice_218 = view_211 = None
        slice_scatter_30: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_219, copy_15, 1, 1, 33);  slice_219 = copy_15 = None
        slice_scatter_31: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg85_1, slice_scatter_30, 0, 0, 32);  slice_scatter_30 = None
        slice_240: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_31, 0, 0, 32)
        slice_241: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_240, 1, 0, 33);  slice_240 = None
        permute_146: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_241, [0, 2, 1, 3]);  slice_241 = None
        expand_47: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_146, [32, 8, 33, 64]);  permute_146 = None
        clone_30: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_221: "f32[256, 33, 64]" = torch.ops.aten.reshape.default(clone_30, [256, 33, 64]);  clone_30 = None
        bmm_15: "f32[256, 32, 64]" = torch.ops.aten.bmm.default(view_220, view_221);  view_220 = view_221 = None
        view_222: "f32[32, 8, 32, 64]" = torch.ops.aten.reshape.default(bmm_15, [32, 8, 32, 64]);  bmm_15 = None
        permute_147: "f32[32, 32, 8, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        clone_31: "f32[32, 32, 8, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_223: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(clone_31, [32, 32, -1]);  clone_31 = None
        view_224: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_223, [1024, 512]);  view_223 = None
        permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_52: "f32[1024, 512]" = torch.ops.aten.mm.default(view_224, permute_148);  view_224 = permute_148 = None
        view_225: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_52, [32, 32, 512]);  mm_52 = None
        add_36: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_34, view_225);  add_34 = view_225 = None
        pow_16: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_36, 2)
        mean_15: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_37: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-05);  mean_15 = None
        rsqrt_15: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_53: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_36, rsqrt_15);  rsqrt_15 = None
        mul_54: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_53, arg87_1);  mul_53 = arg87_1 = None
        view_226: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_54, [1024, 512])
        permute_149: "f32[512, 1536]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_53: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_226, permute_149);  view_226 = permute_149 = None
        view_227: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_53, [32, 32, 1536]);  mm_53 = None
        neg_7: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(view_227)
        exp_15: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_38: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_23: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(view_227, add_38);  view_227 = add_38 = None
        view_228: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_54, [1024, 512]);  mul_54 = None
        permute_150: "f32[512, 1536]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_54: "f32[1024, 1536]" = torch.ops.aten.mm.default(view_228, permute_150);  view_228 = permute_150 = None
        view_229: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_54, [32, 32, 1536]);  mm_54 = None
        mul_55: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_23, view_229);  div_23 = view_229 = None
        view_230: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_55, [1024, 1536]);  mul_55 = None
        permute_151: "f32[1536, 512]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_55: "f32[1024, 512]" = torch.ops.aten.mm.default(view_230, permute_151);  view_230 = permute_151 = None
        view_231: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_55, [32, 32, 512]);  mm_55 = None
        add_39: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_36, view_231);  add_36 = view_231 = None
        pow_17: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_39, 2)
        mean_16: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_40: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-05);  mean_16 = None
        rsqrt_16: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_56: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_39, rsqrt_16);  add_39 = rsqrt_16 = None
        mul_57: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_56, arg91_1);  mul_56 = arg91_1 = None
        select: "f32[32, 512]" = torch.ops.aten.select.int(mul_57, 1, -1);  mul_57 = None
        permute_152: "f32[512, 32000]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_56: "f32[32, 32000]" = torch.ops.aten.mm.default(select, permute_152);  select = permute_152 = None
        copy_: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg7_1, slice_scatter_1);  arg7_1 = slice_scatter_1 = None
        copy__1: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg8_1, slice_scatter_3);  arg8_1 = slice_scatter_3 = None
        copy__2: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg18_1, slice_scatter_5);  arg18_1 = slice_scatter_5 = None
        copy__3: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg19_1, slice_scatter_7);  arg19_1 = slice_scatter_7 = None
        copy__4: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg29_1, slice_scatter_9);  arg29_1 = slice_scatter_9 = None
        copy__5: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg30_1, slice_scatter_11);  arg30_1 = slice_scatter_11 = None
        copy__6: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg40_1, slice_scatter_13);  arg40_1 = slice_scatter_13 = None
        copy__7: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg41_1, slice_scatter_15);  arg41_1 = slice_scatter_15 = None
        copy__8: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg51_1, slice_scatter_17);  arg51_1 = slice_scatter_17 = None
        copy__9: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg52_1, slice_scatter_19);  arg52_1 = slice_scatter_19 = None
        copy__10: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg62_1, slice_scatter_21);  arg62_1 = slice_scatter_21 = None
        copy__11: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg63_1, slice_scatter_23);  arg63_1 = slice_scatter_23 = None
        copy__12: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg73_1, slice_scatter_25);  arg73_1 = slice_scatter_25 = None
        copy__13: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg74_1, slice_scatter_27);  arg74_1 = slice_scatter_27 = None
        copy__14: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg84_1, slice_scatter_29);  arg84_1 = slice_scatter_29 = None
        copy__15: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg85_1, slice_scatter_31);  arg85_1 = slice_scatter_31 = None
        return (mm_56, copy__1, copy_, copy__3, copy__2, copy__5, copy__4, copy__7, copy__6, copy__9, copy__8, copy__11, copy__10, copy__13, copy__12, copy__15, copy__14)
