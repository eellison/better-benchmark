class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 1][1, 1]cuda:0", arg1_1: "f32[16][1]cuda:0", arg2_1: "f32[16, 16][16, 1]cuda:0", arg3_1: "f32[16][1]cuda:0", arg4_1: "f32[16, 16][16, 1]cuda:0", arg5_1: "f32[16][1]cuda:0", arg6_1: "f32[16, 16][16, 1]cuda:0", arg7_1: "f32[16][1]cuda:0", arg8_1: "f32[1, 16][16, 1]cuda:0", arg9_1: "f32[1][1]cuda:0", arg10_1: "f32[][]cuda:0", arg11_1: "f32[16][1]cuda:0", arg12_1: "f32[16][1]cuda:0", arg13_1: "f32[16, 1][1, 1]cuda:0", arg14_1: "f32[16, 1][1, 1]cuda:0", arg15_1: "f32[][]cuda:0", arg16_1: "f32[][]cuda:0", arg17_1: "f32[][]cuda:0", arg18_1: "f32[][]cuda:0", arg19_1: "f32[][]cuda:0", arg20_1: "f32[][]cuda:0", arg21_1: "f32[][]cuda:0", arg22_1: "f32[][]cuda:0", arg23_1: "f32[][]cuda:0", arg24_1: "f32[16, 16][16, 1]cuda:0", arg25_1: "f32[16][1]cuda:0", arg26_1: "f32[16, 16][16, 1]cuda:0", arg27_1: "f32[16][1]cuda:0", arg28_1: "f32[16, 16][16, 1]cuda:0", arg29_1: "f32[16][1]cuda:0", arg30_1: "f32[1, 16][16, 1]cuda:0", arg31_1: "f32[1][1]cuda:0", arg32_1: "f32[16, 16][16, 1]cuda:0", arg33_1: "f32[16][1]cuda:0", arg34_1: "f32[16, 16][16, 1]cuda:0", arg35_1: "f32[16][1]cuda:0", arg36_1: "f32[16, 16][16, 1]cuda:0", arg37_1: "f32[16][1]cuda:0", arg38_1: "f32[1, 16][16, 1]cuda:0", arg39_1: "f32[1][1]cuda:0", arg40_1: "f32[16, 1][1, 1]cuda:0", arg41_1: "f32[16][1]cuda:0", arg42_1: "f32[16, 16][16, 1]cuda:0", arg43_1: "f32[16][1]cuda:0", arg44_1: "f32[16, 16][16, 1]cuda:0", arg45_1: "f32[16][1]cuda:0", arg46_1: "f32[16, 16][16, 1]cuda:0", arg47_1: "f32[16][1]cuda:0", arg48_1: "f32[1, 16][16, 1]cuda:0", arg49_1: "f32[1][1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:686 in _multi_tensor_adam, code: torch._foreach_add_(device_state_steps, 1)
        _foreach_add = torch.ops.aten._foreach_add_.Scalar([arg15_1, arg10_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1], 1);  arg15_1 = arg10_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = None
        getitem: "f32[][]cuda:0" = _foreach_add[0]
        getitem_1: "f32[][]cuda:0" = _foreach_add[1]
        getitem_2: "f32[][]cuda:0" = _foreach_add[2]
        getitem_3: "f32[][]cuda:0" = _foreach_add[3]
        getitem_4: "f32[][]cuda:0" = _foreach_add[4]
        getitem_5: "f32[][]cuda:0" = _foreach_add[5]
        getitem_6: "f32[][]cuda:0" = _foreach_add[6]
        getitem_7: "f32[][]cuda:0" = _foreach_add[7]
        getitem_8: "f32[][]cuda:0" = _foreach_add[8]
        getitem_9: "f32[][]cuda:0" = _foreach_add[9];  _foreach_add = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:704 in _multi_tensor_adam, code: torch._foreach_lerp_(
        _foreach_sub = torch.ops.aten._foreach_sub.List([arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1], [arg13_1, arg11_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1])
        getitem_10: "f32[16, 1][1, 1]cuda:0" = _foreach_sub[0]
        getitem_11: "f32[16][1]cuda:0" = _foreach_sub[1]
        getitem_12: "f32[16, 16][16, 1]cuda:0" = _foreach_sub[2]
        getitem_13: "f32[16][1]cuda:0" = _foreach_sub[3]
        getitem_14: "f32[16, 16][16, 1]cuda:0" = _foreach_sub[4]
        getitem_15: "f32[16][1]cuda:0" = _foreach_sub[5]
        getitem_16: "f32[16, 16][16, 1]cuda:0" = _foreach_sub[6]
        getitem_17: "f32[16][1]cuda:0" = _foreach_sub[7]
        getitem_18: "f32[1, 16][16, 1]cuda:0" = _foreach_sub[8]
        getitem_19: "f32[1][1]cuda:0" = _foreach_sub[9];  _foreach_sub = None
        full_default: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.full.default([16, 1], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[16][1]cuda:0" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[16, 16][16, 1]cuda:0" = torch.ops.aten.full.default([16, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[16][1]cuda:0" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[16, 16][16, 1]cuda:0" = torch.ops.aten.full.default([16, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[16][1]cuda:0" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[16, 16][16, 1]cuda:0" = torch.ops.aten.full.default([16, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[16][1]cuda:0" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.full.default([1, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul = torch.ops.aten._foreach_addcmul_.Scalar([arg13_1, arg11_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9], [getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19]);  arg13_1 = arg11_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_20: "f32[16, 1][1, 1]cuda:0" = _foreach_addcmul[0]
        getitem_21: "f32[16][1]cuda:0" = _foreach_addcmul[1]
        getitem_22: "f32[16, 16][16, 1]cuda:0" = _foreach_addcmul[2]
        getitem_23: "f32[16][1]cuda:0" = _foreach_addcmul[3]
        getitem_24: "f32[16, 16][16, 1]cuda:0" = _foreach_addcmul[4]
        getitem_25: "f32[16][1]cuda:0" = _foreach_addcmul[5]
        getitem_26: "f32[16, 16][16, 1]cuda:0" = _foreach_addcmul[6]
        getitem_27: "f32[16][1]cuda:0" = _foreach_addcmul[7]
        getitem_28: "f32[1, 16][16, 1]cuda:0" = _foreach_addcmul[8]
        getitem_29: "f32[1][1]cuda:0" = _foreach_addcmul[9];  _foreach_addcmul = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:708 in _multi_tensor_adam, code: torch._foreach_mul_(device_exp_avg_sqs, beta2)
        _foreach_mul = torch.ops.aten._foreach_mul.Scalar([arg14_1, arg12_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1], 0.999)
        getitem_30: "f32[16, 1][1, 1]cuda:0" = _foreach_mul[0]
        getitem_31: "f32[16][1]cuda:0" = _foreach_mul[1]
        getitem_32: "f32[16, 16][16, 1]cuda:0" = _foreach_mul[2]
        getitem_33: "f32[16][1]cuda:0" = _foreach_mul[3]
        getitem_34: "f32[16, 16][16, 1]cuda:0" = _foreach_mul[4]
        getitem_35: "f32[16][1]cuda:0" = _foreach_mul[5]
        getitem_36: "f32[16, 16][16, 1]cuda:0" = _foreach_mul[6]
        getitem_37: "f32[16][1]cuda:0" = _foreach_mul[7]
        getitem_38: "f32[1, 16][16, 1]cuda:0" = _foreach_mul[8]
        getitem_39: "f32[1][1]cuda:0" = _foreach_mul[9];  _foreach_mul = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:721 in _multi_tensor_adam, code: torch._foreach_addcmul_(
        _foreach_addcmul_1 = torch.ops.aten._foreach_addcmul.Scalar([getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39], [arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1], [arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1], 0.0010000000000000009);  getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = None
        getitem_40: "f32[16, 1][1, 1]cuda:0" = _foreach_addcmul_1[0]
        getitem_41: "f32[16][1]cuda:0" = _foreach_addcmul_1[1]
        getitem_42: "f32[16, 16][16, 1]cuda:0" = _foreach_addcmul_1[2]
        getitem_43: "f32[16][1]cuda:0" = _foreach_addcmul_1[3]
        getitem_44: "f32[16, 16][16, 1]cuda:0" = _foreach_addcmul_1[4]
        getitem_45: "f32[16][1]cuda:0" = _foreach_addcmul_1[5]
        getitem_46: "f32[16, 16][16, 1]cuda:0" = _foreach_addcmul_1[6]
        getitem_47: "f32[16][1]cuda:0" = _foreach_addcmul_1[7]
        getitem_48: "f32[1, 16][16, 1]cuda:0" = _foreach_addcmul_1[8]
        getitem_49: "f32[1][1]cuda:0" = _foreach_addcmul_1[9];  _foreach_addcmul_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:734 in _multi_tensor_adam, code: bias_correction1 = torch._foreach_pow(beta1, device_state_steps)  # type: ignore[arg-type]
        _foreach_pow = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9])
        getitem_50: "f32[][]cuda:0" = _foreach_pow[0]
        getitem_51: "f32[][]cuda:0" = _foreach_pow[1]
        getitem_52: "f32[][]cuda:0" = _foreach_pow[2]
        getitem_53: "f32[][]cuda:0" = _foreach_pow[3]
        getitem_54: "f32[][]cuda:0" = _foreach_pow[4]
        getitem_55: "f32[][]cuda:0" = _foreach_pow[5]
        getitem_56: "f32[][]cuda:0" = _foreach_pow[6]
        getitem_57: "f32[][]cuda:0" = _foreach_pow[7]
        getitem_58: "f32[][]cuda:0" = _foreach_pow[8]
        getitem_59: "f32[][]cuda:0" = _foreach_pow[9];  _foreach_pow = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:735 in _multi_tensor_adam, code: bias_correction2 = torch._foreach_pow(beta2, device_state_steps)  # type: ignore[arg-type]
        _foreach_pow_1 = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9]);  getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = None
        getitem_60: "f32[][]cuda:0" = _foreach_pow_1[0]
        getitem_61: "f32[][]cuda:0" = _foreach_pow_1[1]
        getitem_62: "f32[][]cuda:0" = _foreach_pow_1[2]
        getitem_63: "f32[][]cuda:0" = _foreach_pow_1[3]
        getitem_64: "f32[][]cuda:0" = _foreach_pow_1[4]
        getitem_65: "f32[][]cuda:0" = _foreach_pow_1[5]
        getitem_66: "f32[][]cuda:0" = _foreach_pow_1[6]
        getitem_67: "f32[][]cuda:0" = _foreach_pow_1[7]
        getitem_68: "f32[][]cuda:0" = _foreach_pow_1[8]
        getitem_69: "f32[][]cuda:0" = _foreach_pow_1[9];  _foreach_pow_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:737 in _multi_tensor_adam, code: torch._foreach_sub_(bias_correction1, 1)
        _foreach_sub_1 = torch.ops.aten._foreach_sub.Scalar([getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59], 1);  getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = None
        getitem_70: "f32[][]cuda:0" = _foreach_sub_1[0]
        getitem_71: "f32[][]cuda:0" = _foreach_sub_1[1]
        getitem_72: "f32[][]cuda:0" = _foreach_sub_1[2]
        getitem_73: "f32[][]cuda:0" = _foreach_sub_1[3]
        getitem_74: "f32[][]cuda:0" = _foreach_sub_1[4]
        getitem_75: "f32[][]cuda:0" = _foreach_sub_1[5]
        getitem_76: "f32[][]cuda:0" = _foreach_sub_1[6]
        getitem_77: "f32[][]cuda:0" = _foreach_sub_1[7]
        getitem_78: "f32[][]cuda:0" = _foreach_sub_1[8]
        getitem_79: "f32[][]cuda:0" = _foreach_sub_1[9];  _foreach_sub_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:738 in _multi_tensor_adam, code: torch._foreach_sub_(bias_correction2, 1)
        _foreach_sub_2 = torch.ops.aten._foreach_sub.Scalar([getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69], 1);  getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = None
        getitem_80: "f32[][]cuda:0" = _foreach_sub_2[0]
        getitem_81: "f32[][]cuda:0" = _foreach_sub_2[1]
        getitem_82: "f32[][]cuda:0" = _foreach_sub_2[2]
        getitem_83: "f32[][]cuda:0" = _foreach_sub_2[3]
        getitem_84: "f32[][]cuda:0" = _foreach_sub_2[4]
        getitem_85: "f32[][]cuda:0" = _foreach_sub_2[5]
        getitem_86: "f32[][]cuda:0" = _foreach_sub_2[6]
        getitem_87: "f32[][]cuda:0" = _foreach_sub_2[7]
        getitem_88: "f32[][]cuda:0" = _foreach_sub_2[8]
        getitem_89: "f32[][]cuda:0" = _foreach_sub_2[9];  _foreach_sub_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:740 in _multi_tensor_adam, code: torch._foreach_neg_(bias_correction2)
        _foreach_neg = torch.ops.aten._foreach_neg.default([getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89]);  getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = None
        getitem_90: "f32[][]cuda:0" = _foreach_neg[0]
        getitem_91: "f32[][]cuda:0" = _foreach_neg[1]
        getitem_92: "f32[][]cuda:0" = _foreach_neg[2]
        getitem_93: "f32[][]cuda:0" = _foreach_neg[3]
        getitem_94: "f32[][]cuda:0" = _foreach_neg[4]
        getitem_95: "f32[][]cuda:0" = _foreach_neg[5]
        getitem_96: "f32[][]cuda:0" = _foreach_neg[6]
        getitem_97: "f32[][]cuda:0" = _foreach_neg[7]
        getitem_98: "f32[][]cuda:0" = _foreach_neg[8]
        getitem_99: "f32[][]cuda:0" = _foreach_neg[9];  _foreach_neg = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:743 in _multi_tensor_adam, code: torch._foreach_div_(bias_correction1, lr)
        _foreach_div = torch.ops.aten._foreach_div.Scalar([getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79], 0.01);  getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = None
        getitem_100: "f32[][]cuda:0" = _foreach_div[0]
        getitem_101: "f32[][]cuda:0" = _foreach_div[1]
        getitem_102: "f32[][]cuda:0" = _foreach_div[2]
        getitem_103: "f32[][]cuda:0" = _foreach_div[3]
        getitem_104: "f32[][]cuda:0" = _foreach_div[4]
        getitem_105: "f32[][]cuda:0" = _foreach_div[5]
        getitem_106: "f32[][]cuda:0" = _foreach_div[6]
        getitem_107: "f32[][]cuda:0" = _foreach_div[7]
        getitem_108: "f32[][]cuda:0" = _foreach_div[8]
        getitem_109: "f32[][]cuda:0" = _foreach_div[9];  _foreach_div = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:744 in _multi_tensor_adam, code: torch._foreach_reciprocal_(bias_correction1)
        _foreach_reciprocal = torch.ops.aten._foreach_reciprocal.default([getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109]);  getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = None
        getitem_110: "f32[][]cuda:0" = _foreach_reciprocal[0]
        getitem_111: "f32[][]cuda:0" = _foreach_reciprocal[1]
        getitem_112: "f32[][]cuda:0" = _foreach_reciprocal[2]
        getitem_113: "f32[][]cuda:0" = _foreach_reciprocal[3]
        getitem_114: "f32[][]cuda:0" = _foreach_reciprocal[4]
        getitem_115: "f32[][]cuda:0" = _foreach_reciprocal[5]
        getitem_116: "f32[][]cuda:0" = _foreach_reciprocal[6]
        getitem_117: "f32[][]cuda:0" = _foreach_reciprocal[7]
        getitem_118: "f32[][]cuda:0" = _foreach_reciprocal[8]
        getitem_119: "f32[][]cuda:0" = _foreach_reciprocal[9];  _foreach_reciprocal = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:746 in _multi_tensor_adam, code: torch._foreach_sqrt_(bias_correction2)
        _foreach_sqrt = torch.ops.aten._foreach_sqrt.default([getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99]);  getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = None
        getitem_120: "f32[][]cuda:0" = _foreach_sqrt[0]
        getitem_121: "f32[][]cuda:0" = _foreach_sqrt[1]
        getitem_122: "f32[][]cuda:0" = _foreach_sqrt[2]
        getitem_123: "f32[][]cuda:0" = _foreach_sqrt[3]
        getitem_124: "f32[][]cuda:0" = _foreach_sqrt[4]
        getitem_125: "f32[][]cuda:0" = _foreach_sqrt[5]
        getitem_126: "f32[][]cuda:0" = _foreach_sqrt[6]
        getitem_127: "f32[][]cuda:0" = _foreach_sqrt[7]
        getitem_128: "f32[][]cuda:0" = _foreach_sqrt[8]
        getitem_129: "f32[][]cuda:0" = _foreach_sqrt[9];  _foreach_sqrt = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:762 in _multi_tensor_adam, code: exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        _foreach_sqrt_1 = torch.ops.aten._foreach_sqrt.default([getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49])
        getitem_130: "f32[16, 1][1, 1]cuda:0" = _foreach_sqrt_1[0]
        getitem_131: "f32[16][1]cuda:0" = _foreach_sqrt_1[1]
        getitem_132: "f32[16, 16][16, 1]cuda:0" = _foreach_sqrt_1[2]
        getitem_133: "f32[16][1]cuda:0" = _foreach_sqrt_1[3]
        getitem_134: "f32[16, 16][16, 1]cuda:0" = _foreach_sqrt_1[4]
        getitem_135: "f32[16][1]cuda:0" = _foreach_sqrt_1[5]
        getitem_136: "f32[16, 16][16, 1]cuda:0" = _foreach_sqrt_1[6]
        getitem_137: "f32[16][1]cuda:0" = _foreach_sqrt_1[7]
        getitem_138: "f32[1, 16][16, 1]cuda:0" = _foreach_sqrt_1[8]
        getitem_139: "f32[1][1]cuda:0" = _foreach_sqrt_1[9];  _foreach_sqrt_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:764 in _multi_tensor_adam, code: torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
        _foreach_div_1 = torch.ops.aten._foreach_div.List([getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139], [getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129]);  getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_140: "f32[16, 1][1, 1]cuda:0" = _foreach_div_1[0]
        getitem_141: "f32[16][1]cuda:0" = _foreach_div_1[1]
        getitem_142: "f32[16, 16][16, 1]cuda:0" = _foreach_div_1[2]
        getitem_143: "f32[16][1]cuda:0" = _foreach_div_1[3]
        getitem_144: "f32[16, 16][16, 1]cuda:0" = _foreach_div_1[4]
        getitem_145: "f32[16][1]cuda:0" = _foreach_div_1[5]
        getitem_146: "f32[16, 16][16, 1]cuda:0" = _foreach_div_1[6]
        getitem_147: "f32[16][1]cuda:0" = _foreach_div_1[7]
        getitem_148: "f32[1, 16][16, 1]cuda:0" = _foreach_div_1[8]
        getitem_149: "f32[1][1]cuda:0" = _foreach_div_1[9];  _foreach_div_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:765 in _multi_tensor_adam, code: torch._foreach_add_(exp_avg_sq_sqrt, eps)
        _foreach_add_1 = torch.ops.aten._foreach_add.Scalar([getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149], 1e-08);  getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = None
        getitem_150: "f32[16, 1][1, 1]cuda:0" = _foreach_add_1[0]
        getitem_151: "f32[16][1]cuda:0" = _foreach_add_1[1]
        getitem_152: "f32[16, 16][16, 1]cuda:0" = _foreach_add_1[2]
        getitem_153: "f32[16][1]cuda:0" = _foreach_add_1[3]
        getitem_154: "f32[16, 16][16, 1]cuda:0" = _foreach_add_1[4]
        getitem_155: "f32[16][1]cuda:0" = _foreach_add_1[5]
        getitem_156: "f32[16, 16][16, 1]cuda:0" = _foreach_add_1[6]
        getitem_157: "f32[16][1]cuda:0" = _foreach_add_1[7]
        getitem_158: "f32[1, 16][16, 1]cuda:0" = _foreach_add_1[8]
        getitem_159: "f32[1][1]cuda:0" = _foreach_add_1[9];  _foreach_add_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:766 in _multi_tensor_adam, code: torch._foreach_div_(exp_avg_sq_sqrt, step_size)
        _foreach_div_2 = torch.ops.aten._foreach_div.List([getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159], [getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119]);  getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = None
        getitem_160: "f32[16, 1][1, 1]cuda:0" = _foreach_div_2[0]
        getitem_161: "f32[16][1]cuda:0" = _foreach_div_2[1]
        getitem_162: "f32[16, 16][16, 1]cuda:0" = _foreach_div_2[2]
        getitem_163: "f32[16][1]cuda:0" = _foreach_div_2[3]
        getitem_164: "f32[16, 16][16, 1]cuda:0" = _foreach_div_2[4]
        getitem_165: "f32[16][1]cuda:0" = _foreach_div_2[5]
        getitem_166: "f32[16, 16][16, 1]cuda:0" = _foreach_div_2[6]
        getitem_167: "f32[16][1]cuda:0" = _foreach_div_2[7]
        getitem_168: "f32[1, 16][16, 1]cuda:0" = _foreach_div_2[8]
        getitem_169: "f32[1][1]cuda:0" = _foreach_div_2[9];  _foreach_div_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:769 in _multi_tensor_adam, code: torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        _foreach_addcdiv = torch.ops.aten._foreach_addcdiv_.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1], [getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29], [getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = arg9_1 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = None
        getitem_170: "f32[16, 1][1, 1]cuda:0" = _foreach_addcdiv[0];  getitem_170 = None
        getitem_171: "f32[16][1]cuda:0" = _foreach_addcdiv[1];  getitem_171 = None
        getitem_172: "f32[16, 16][16, 1]cuda:0" = _foreach_addcdiv[2];  getitem_172 = None
        getitem_173: "f32[16][1]cuda:0" = _foreach_addcdiv[3];  getitem_173 = None
        getitem_174: "f32[16, 16][16, 1]cuda:0" = _foreach_addcdiv[4];  getitem_174 = None
        getitem_175: "f32[16][1]cuda:0" = _foreach_addcdiv[5];  getitem_175 = None
        getitem_176: "f32[16, 16][16, 1]cuda:0" = _foreach_addcdiv[6];  getitem_176 = None
        getitem_177: "f32[16][1]cuda:0" = _foreach_addcdiv[7];  getitem_177 = None
        getitem_178: "f32[1, 16][16, 1]cuda:0" = _foreach_addcdiv[8];  getitem_178 = None
        getitem_179: "f32[1][1]cuda:0" = _foreach_addcdiv[9];  _foreach_addcdiv = getitem_179 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/optim/adam.py:721 in _multi_tensor_adam, code: torch._foreach_addcmul_(
        copy__12: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(arg12_1, getitem_41);  arg12_1 = getitem_41 = copy__12 = None
        copy__14: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.copy_.default(arg14_1, getitem_40);  arg14_1 = getitem_40 = copy__14 = None
        copy__32: "f32[16, 16][16, 1]cuda:0" = torch.ops.aten.copy_.default(arg32_1, getitem_42);  arg32_1 = getitem_42 = copy__32 = None
        copy__33: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(arg33_1, getitem_43);  arg33_1 = getitem_43 = copy__33 = None
        copy__34: "f32[16, 16][16, 1]cuda:0" = torch.ops.aten.copy_.default(arg34_1, getitem_44);  arg34_1 = getitem_44 = copy__34 = None
        copy__35: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(arg35_1, getitem_45);  arg35_1 = getitem_45 = copy__35 = None
        copy__36: "f32[16, 16][16, 1]cuda:0" = torch.ops.aten.copy_.default(arg36_1, getitem_46);  arg36_1 = getitem_46 = copy__36 = None
        copy__37: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(arg37_1, getitem_47);  arg37_1 = getitem_47 = copy__37 = None
        copy__38: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.copy_.default(arg38_1, getitem_48);  arg38_1 = getitem_48 = copy__38 = None
        copy__39: "f32[1][1]cuda:0" = torch.ops.aten.copy_.default(arg39_1, getitem_49);  arg39_1 = getitem_49 = copy__39 = None
        return ()
