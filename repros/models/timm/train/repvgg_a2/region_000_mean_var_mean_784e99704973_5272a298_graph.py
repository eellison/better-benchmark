class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "i64[]", getitem_1: "f32[1, 64, 1, 1]", rsqrt: "f32[1, 64, 1, 1]", primals_4: "f32[64]", getitem: "f32[1, 64, 1, 1]", primals_5: "f32[64]", primals_9: "i64[]", getitem_3: "f32[1, 64, 1, 1]", rsqrt_1: "f32[1, 64, 1, 1]", primals_10: "f32[64]", getitem_2: "f32[1, 64, 1, 1]", primals_11: "f32[64]", primals_15: "i64[]", getitem_5: "f32[1, 96, 1, 1]", rsqrt_2: "f32[1, 96, 1, 1]", primals_16: "f32[96]", getitem_4: "f32[1, 96, 1, 1]", primals_17: "f32[96]", primals_21: "i64[]", getitem_7: "f32[1, 96, 1, 1]", rsqrt_3: "f32[1, 96, 1, 1]", primals_22: "f32[96]", getitem_6: "f32[1, 96, 1, 1]", primals_23: "f32[96]", primals_26: "i64[]", getitem_9: "f32[1, 96, 1, 1]", rsqrt_4: "f32[1, 96, 1, 1]", primals_27: "f32[96]", getitem_8: "f32[1, 96, 1, 1]", primals_28: "f32[96]", primals_32: "i64[]", getitem_11: "f32[1, 96, 1, 1]", rsqrt_5: "f32[1, 96, 1, 1]", primals_33: "f32[96]", getitem_10: "f32[1, 96, 1, 1]", primals_34: "f32[96]", primals_38: "i64[]", getitem_13: "f32[1, 96, 1, 1]", rsqrt_6: "f32[1, 96, 1, 1]", primals_39: "f32[96]", getitem_12: "f32[1, 96, 1, 1]", primals_40: "f32[96]", primals_44: "i64[]", getitem_15: "f32[1, 192, 1, 1]", rsqrt_7: "f32[1, 192, 1, 1]", primals_45: "f32[192]", getitem_14: "f32[1, 192, 1, 1]", primals_46: "f32[192]", primals_50: "i64[]", getitem_17: "f32[1, 192, 1, 1]", rsqrt_8: "f32[1, 192, 1, 1]", primals_51: "f32[192]", getitem_16: "f32[1, 192, 1, 1]", primals_52: "f32[192]", primals_55: "i64[]", getitem_19: "f32[1, 192, 1, 1]", rsqrt_9: "f32[1, 192, 1, 1]", primals_56: "f32[192]", getitem_18: "f32[1, 192, 1, 1]", primals_57: "f32[192]", primals_61: "i64[]", getitem_21: "f32[1, 192, 1, 1]", rsqrt_10: "f32[1, 192, 1, 1]", primals_62: "f32[192]", getitem_20: "f32[1, 192, 1, 1]", primals_63: "f32[192]", primals_67: "i64[]", getitem_23: "f32[1, 192, 1, 1]", rsqrt_11: "f32[1, 192, 1, 1]", primals_68: "f32[192]", getitem_22: "f32[1, 192, 1, 1]", primals_69: "f32[192]", primals_72: "i64[]", getitem_25: "f32[1, 192, 1, 1]", rsqrt_12: "f32[1, 192, 1, 1]", primals_73: "f32[192]", getitem_24: "f32[1, 192, 1, 1]", primals_74: "f32[192]", primals_78: "i64[]", getitem_27: "f32[1, 192, 1, 1]", rsqrt_13: "f32[1, 192, 1, 1]", primals_79: "f32[192]", getitem_26: "f32[1, 192, 1, 1]", primals_80: "f32[192]", primals_84: "i64[]", getitem_29: "f32[1, 192, 1, 1]", rsqrt_14: "f32[1, 192, 1, 1]", primals_85: "f32[192]", getitem_28: "f32[1, 192, 1, 1]", primals_86: "f32[192]", primals_89: "i64[]", getitem_31: "f32[1, 192, 1, 1]", rsqrt_15: "f32[1, 192, 1, 1]", primals_90: "f32[192]", getitem_30: "f32[1, 192, 1, 1]", primals_91: "f32[192]", primals_95: "i64[]", getitem_33: "f32[1, 192, 1, 1]", rsqrt_16: "f32[1, 192, 1, 1]", primals_96: "f32[192]", getitem_32: "f32[1, 192, 1, 1]", primals_97: "f32[192]", primals_101: "i64[]", getitem_35: "f32[1, 192, 1, 1]", rsqrt_17: "f32[1, 192, 1, 1]", primals_102: "f32[192]", getitem_34: "f32[1, 192, 1, 1]", primals_103: "f32[192]", primals_107: "i64[]", getitem_37: "f32[1, 384, 1, 1]", rsqrt_18: "f32[1, 384, 1, 1]", primals_108: "f32[384]", getitem_36: "f32[1, 384, 1, 1]", primals_109: "f32[384]", primals_113: "i64[]", getitem_39: "f32[1, 384, 1, 1]", rsqrt_19: "f32[1, 384, 1, 1]", primals_114: "f32[384]", getitem_38: "f32[1, 384, 1, 1]", primals_115: "f32[384]", primals_118: "i64[]", getitem_41: "f32[1, 384, 1, 1]", rsqrt_20: "f32[1, 384, 1, 1]", primals_119: "f32[384]", getitem_40: "f32[1, 384, 1, 1]", primals_120: "f32[384]", primals_124: "i64[]", getitem_43: "f32[1, 384, 1, 1]", rsqrt_21: "f32[1, 384, 1, 1]", primals_125: "f32[384]", getitem_42: "f32[1, 384, 1, 1]", primals_126: "f32[384]", primals_130: "i64[]", getitem_45: "f32[1, 384, 1, 1]", rsqrt_22: "f32[1, 384, 1, 1]", primals_131: "f32[384]", getitem_44: "f32[1, 384, 1, 1]", primals_132: "f32[384]", primals_135: "i64[]", getitem_47: "f32[1, 384, 1, 1]", rsqrt_23: "f32[1, 384, 1, 1]", primals_136: "f32[384]", getitem_46: "f32[1, 384, 1, 1]", primals_137: "f32[384]", primals_141: "i64[]", getitem_49: "f32[1, 384, 1, 1]", rsqrt_24: "f32[1, 384, 1, 1]", primals_142: "f32[384]", getitem_48: "f32[1, 384, 1, 1]", primals_143: "f32[384]", primals_147: "i64[]", getitem_51: "f32[1, 384, 1, 1]", rsqrt_25: "f32[1, 384, 1, 1]", primals_148: "f32[384]", getitem_50: "f32[1, 384, 1, 1]", primals_149: "f32[384]", primals_152: "i64[]", getitem_53: "f32[1, 384, 1, 1]", rsqrt_26: "f32[1, 384, 1, 1]", primals_153: "f32[384]", getitem_52: "f32[1, 384, 1, 1]", primals_154: "f32[384]", primals_158: "i64[]", getitem_55: "f32[1, 384, 1, 1]", rsqrt_27: "f32[1, 384, 1, 1]", primals_159: "f32[384]", getitem_54: "f32[1, 384, 1, 1]", primals_160: "f32[384]", primals_164: "i64[]", getitem_57: "f32[1, 384, 1, 1]", rsqrt_28: "f32[1, 384, 1, 1]", primals_165: "f32[384]", getitem_56: "f32[1, 384, 1, 1]", primals_166: "f32[384]", primals_169: "i64[]", getitem_59: "f32[1, 384, 1, 1]", rsqrt_29: "f32[1, 384, 1, 1]", primals_170: "f32[384]", getitem_58: "f32[1, 384, 1, 1]", primals_171: "f32[384]", primals_175: "i64[]", getitem_61: "f32[1, 384, 1, 1]", rsqrt_30: "f32[1, 384, 1, 1]", primals_176: "f32[384]", getitem_60: "f32[1, 384, 1, 1]", primals_177: "f32[384]", primals_181: "i64[]", getitem_63: "f32[1, 384, 1, 1]", rsqrt_31: "f32[1, 384, 1, 1]", primals_182: "f32[384]", getitem_62: "f32[1, 384, 1, 1]", primals_183: "f32[384]", primals_186: "i64[]", getitem_65: "f32[1, 384, 1, 1]", rsqrt_32: "f32[1, 384, 1, 1]", primals_187: "f32[384]", getitem_64: "f32[1, 384, 1, 1]", primals_188: "f32[384]", primals_192: "i64[]", getitem_67: "f32[1, 384, 1, 1]", rsqrt_33: "f32[1, 384, 1, 1]", primals_193: "f32[384]", getitem_66: "f32[1, 384, 1, 1]", primals_194: "f32[384]", primals_198: "i64[]", getitem_69: "f32[1, 384, 1, 1]", rsqrt_34: "f32[1, 384, 1, 1]", primals_199: "f32[384]", getitem_68: "f32[1, 384, 1, 1]", primals_200: "f32[384]", primals_203: "i64[]", getitem_71: "f32[1, 384, 1, 1]", rsqrt_35: "f32[1, 384, 1, 1]", primals_204: "f32[384]", getitem_70: "f32[1, 384, 1, 1]", primals_205: "f32[384]", primals_209: "i64[]", getitem_73: "f32[1, 384, 1, 1]", rsqrt_36: "f32[1, 384, 1, 1]", primals_210: "f32[384]", getitem_72: "f32[1, 384, 1, 1]", primals_211: "f32[384]", primals_215: "i64[]", getitem_75: "f32[1, 384, 1, 1]", rsqrt_37: "f32[1, 384, 1, 1]", primals_216: "f32[384]", getitem_74: "f32[1, 384, 1, 1]", primals_217: "f32[384]", primals_220: "i64[]", getitem_77: "f32[1, 384, 1, 1]", rsqrt_38: "f32[1, 384, 1, 1]", primals_221: "f32[384]", getitem_76: "f32[1, 384, 1, 1]", primals_222: "f32[384]", primals_226: "i64[]", getitem_79: "f32[1, 384, 1, 1]", rsqrt_39: "f32[1, 384, 1, 1]", primals_227: "f32[384]", getitem_78: "f32[1, 384, 1, 1]", primals_228: "f32[384]", primals_232: "i64[]", getitem_81: "f32[1, 384, 1, 1]", rsqrt_40: "f32[1, 384, 1, 1]", primals_233: "f32[384]", getitem_80: "f32[1, 384, 1, 1]", primals_234: "f32[384]", primals_237: "i64[]", getitem_83: "f32[1, 384, 1, 1]", rsqrt_41: "f32[1, 384, 1, 1]", primals_238: "f32[384]", getitem_82: "f32[1, 384, 1, 1]", primals_239: "f32[384]", primals_243: "i64[]", getitem_85: "f32[1, 384, 1, 1]", rsqrt_42: "f32[1, 384, 1, 1]", primals_244: "f32[384]", getitem_84: "f32[1, 384, 1, 1]", primals_245: "f32[384]", primals_249: "i64[]", getitem_87: "f32[1, 384, 1, 1]", rsqrt_43: "f32[1, 384, 1, 1]", primals_250: "f32[384]", getitem_86: "f32[1, 384, 1, 1]", primals_251: "f32[384]", primals_254: "i64[]", getitem_89: "f32[1, 384, 1, 1]", rsqrt_44: "f32[1, 384, 1, 1]", primals_255: "f32[384]", getitem_88: "f32[1, 384, 1, 1]", primals_256: "f32[384]", primals_260: "i64[]", getitem_91: "f32[1, 384, 1, 1]", rsqrt_45: "f32[1, 384, 1, 1]", primals_261: "f32[384]", getitem_90: "f32[1, 384, 1, 1]", primals_262: "f32[384]", primals_266: "i64[]", getitem_93: "f32[1, 384, 1, 1]", rsqrt_46: "f32[1, 384, 1, 1]", primals_267: "f32[384]", getitem_92: "f32[1, 384, 1, 1]", primals_268: "f32[384]", primals_271: "i64[]", getitem_95: "f32[1, 384, 1, 1]", rsqrt_47: "f32[1, 384, 1, 1]", primals_272: "f32[384]", getitem_94: "f32[1, 384, 1, 1]", primals_273: "f32[384]", primals_277: "i64[]", getitem_97: "f32[1, 384, 1, 1]", rsqrt_48: "f32[1, 384, 1, 1]", primals_278: "f32[384]", getitem_96: "f32[1, 384, 1, 1]", primals_279: "f32[384]", primals_283: "i64[]", getitem_99: "f32[1, 384, 1, 1]", rsqrt_49: "f32[1, 384, 1, 1]", primals_284: "f32[384]", getitem_98: "f32[1, 384, 1, 1]", primals_285: "f32[384]", primals_288: "i64[]", getitem_101: "f32[1, 384, 1, 1]", rsqrt_50: "f32[1, 384, 1, 1]", primals_289: "f32[384]", getitem_100: "f32[1, 384, 1, 1]", primals_290: "f32[384]", primals_294: "i64[]", getitem_103: "f32[1, 384, 1, 1]", rsqrt_51: "f32[1, 384, 1, 1]", primals_295: "f32[384]", getitem_102: "f32[1, 384, 1, 1]", primals_296: "f32[384]", primals_300: "i64[]", getitem_105: "f32[1, 384, 1, 1]", rsqrt_52: "f32[1, 384, 1, 1]", primals_301: "f32[384]", getitem_104: "f32[1, 384, 1, 1]", primals_302: "f32[384]", primals_305: "i64[]", getitem_107: "f32[1, 384, 1, 1]", rsqrt_53: "f32[1, 384, 1, 1]", primals_306: "f32[384]", getitem_106: "f32[1, 384, 1, 1]", primals_307: "f32[384]", primals_311: "i64[]", getitem_109: "f32[1, 384, 1, 1]", rsqrt_54: "f32[1, 384, 1, 1]", primals_312: "f32[384]", getitem_108: "f32[1, 384, 1, 1]", primals_313: "f32[384]", primals_317: "i64[]", getitem_111: "f32[1, 384, 1, 1]", rsqrt_55: "f32[1, 384, 1, 1]", primals_318: "f32[384]", getitem_110: "f32[1, 384, 1, 1]", primals_319: "f32[384]", primals_322: "i64[]", getitem_113: "f32[1, 384, 1, 1]", rsqrt_56: "f32[1, 384, 1, 1]", primals_323: "f32[384]", getitem_112: "f32[1, 384, 1, 1]", primals_324: "f32[384]", primals_328: "i64[]", getitem_115: "f32[1, 384, 1, 1]", rsqrt_57: "f32[1, 384, 1, 1]", primals_329: "f32[384]", getitem_114: "f32[1, 384, 1, 1]", primals_330: "f32[384]", primals_334: "i64[]", getitem_117: "f32[1, 384, 1, 1]", rsqrt_58: "f32[1, 384, 1, 1]", primals_335: "f32[384]", getitem_116: "f32[1, 384, 1, 1]", primals_336: "f32[384]", primals_340: "i64[]", convolution_42: "f32[128, 1408, 7, 7]", primals_341: "f32[1408]", primals_342: "f32[1408]", primals_343: "f32[1408]", primals_344: "f32[1408]", primals_346: "i64[]", convolution_43: "f32[128, 1408, 7, 7]", primals_347: "f32[1408]", primals_348: "f32[1408]", primals_349: "f32[1408]", primals_350: "f32[1408]", primals_351: "f32[1000, 1408]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000006228081046);  squeeze_dims_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_tensor_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000006228081046);  squeeze_dims_5 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_6: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_7: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1)
        mul_tensor_11: "f32[96]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_tensor_7: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_8: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.0000024912370735);  squeeze_dims_8 = None
        mul_tensor_13: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[96]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_tensor_8: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_9: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_10: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 0.1)
        mul_tensor_16: "f32[96]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_tensor_10: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_11: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 1.0000024912370735);  squeeze_dims_11 = None
        mul_tensor_18: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[96]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_tensor_11: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(primals_26, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_12: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_13: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_12, 0.1)
        mul_tensor_21: "f32[96]" = torch.ops.aten.mul.Tensor(primals_27, 0.9)
        add_tensor_13: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_14: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 1.0000024912370735);  squeeze_dims_14 = None
        mul_tensor_23: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[96]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_tensor_14: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(primals_32, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_15: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_16: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_15, 0.1)
        mul_tensor_26: "f32[96]" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_tensor_16: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_17: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 1.0000024912370735);  squeeze_dims_17 = None
        mul_tensor_28: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[96]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_tensor_17: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(primals_38, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_18: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_19: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 0.1)
        mul_tensor_31: "f32[96]" = torch.ops.aten.mul.Tensor(primals_39, 0.9)
        add_tensor_19: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_20: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 1.0000024912370735);  squeeze_dims_20 = None
        mul_tensor_33: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[96]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_tensor_20: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(primals_44, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_21: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_22: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_21, 0.1)
        mul_tensor_36: "f32[192]" = torch.ops.aten.mul.Tensor(primals_45, 0.9)
        add_tensor_22: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_23: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 1.00000996502277);  squeeze_dims_23 = None
        mul_tensor_38: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[192]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_tensor_23: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(primals_50, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_24: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_25: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 0.1)
        mul_tensor_41: "f32[192]" = torch.ops.aten.mul.Tensor(primals_51, 0.9)
        add_tensor_25: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_26: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_26, 1.00000996502277);  squeeze_dims_26 = None
        mul_tensor_43: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[192]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_tensor_26: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(primals_55, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_27: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_28: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 0.1)
        mul_tensor_46: "f32[192]" = torch.ops.aten.mul.Tensor(primals_56, 0.9)
        add_tensor_28: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_29: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 1.00000996502277);  squeeze_dims_29 = None
        mul_tensor_48: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[192]" = torch.ops.aten.mul.Tensor(primals_57, 0.9)
        add_tensor_29: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(primals_61, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_30: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_31: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 0.1)
        mul_tensor_51: "f32[192]" = torch.ops.aten.mul.Tensor(primals_62, 0.9)
        add_tensor_31: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_32: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 1.00000996502277);  squeeze_dims_32 = None
        mul_tensor_53: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[192]" = torch.ops.aten.mul.Tensor(primals_63, 0.9)
        add_tensor_32: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(primals_67, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_33: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_34: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 0.1)
        mul_tensor_56: "f32[192]" = torch.ops.aten.mul.Tensor(primals_68, 0.9)
        add_tensor_34: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_35: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_35, 1.00000996502277);  squeeze_dims_35 = None
        mul_tensor_58: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[192]" = torch.ops.aten.mul.Tensor(primals_69, 0.9)
        add_tensor_35: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(primals_72, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_36: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_37: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 0.1)
        mul_tensor_61: "f32[192]" = torch.ops.aten.mul.Tensor(primals_73, 0.9)
        add_tensor_37: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_38: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.00000996502277);  squeeze_dims_38 = None
        mul_tensor_63: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[192]" = torch.ops.aten.mul.Tensor(primals_74, 0.9)
        add_tensor_38: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(primals_78, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_39: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_dims_40: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_tensor_65: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.1)
        mul_tensor_66: "f32[192]" = torch.ops.aten.mul.Tensor(primals_79, 0.9)
        add_tensor_40: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_41: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 1.00000996502277);  squeeze_dims_41 = None
        mul_tensor_68: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[192]" = torch.ops.aten.mul.Tensor(primals_80, 0.9)
        add_tensor_41: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(primals_84, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_42: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_43: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 0.1)
        mul_tensor_71: "f32[192]" = torch.ops.aten.mul.Tensor(primals_85, 0.9)
        add_tensor_43: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_44: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 1.00000996502277);  squeeze_dims_44 = None
        mul_tensor_73: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[192]" = torch.ops.aten.mul.Tensor(primals_86, 0.9)
        add_tensor_44: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(primals_89, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_45: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_46: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 0.1)
        mul_tensor_76: "f32[192]" = torch.ops.aten.mul.Tensor(primals_90, 0.9)
        add_tensor_46: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_47: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 1.00000996502277);  squeeze_dims_47 = None
        mul_tensor_78: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[192]" = torch.ops.aten.mul.Tensor(primals_91, 0.9)
        add_tensor_47: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(primals_95, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_48: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_49: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 0.1)
        mul_tensor_81: "f32[192]" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
        add_tensor_49: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_50: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 1.00000996502277);  squeeze_dims_50 = None
        mul_tensor_83: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[192]" = torch.ops.aten.mul.Tensor(primals_97, 0.9)
        add_tensor_50: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(primals_101, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_51: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_dims_52: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_85: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 0.1)
        mul_tensor_86: "f32[192]" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
        add_tensor_52: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_53: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_87: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 1.00000996502277);  squeeze_dims_53 = None
        mul_tensor_88: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[192]" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
        add_tensor_53: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(primals_107, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_54: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_dims_55: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_tensor_90: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 0.1)
        mul_tensor_91: "f32[384]" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
        add_tensor_55: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_56: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_92: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 1.0000398612827361);  squeeze_dims_56 = None
        mul_tensor_93: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[384]" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        add_tensor_56: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(primals_113, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_57: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_dims_58: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 0.1)
        mul_tensor_96: "f32[384]" = torch.ops.aten.mul.Tensor(primals_114, 0.9)
        add_tensor_58: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_59: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_97: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 1.0000398612827361);  squeeze_dims_59 = None
        mul_tensor_98: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[384]" = torch.ops.aten.mul.Tensor(primals_115, 0.9)
        add_tensor_59: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(primals_118, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_60: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_61: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 0.1)
        mul_tensor_101: "f32[384]" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_tensor_61: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_62: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_102: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 1.0000398612827361);  squeeze_dims_62 = None
        mul_tensor_103: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[384]" = torch.ops.aten.mul.Tensor(primals_120, 0.9)
        add_tensor_62: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(primals_124, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_63: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_dims_64: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_105: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 0.1)
        mul_tensor_106: "f32[384]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_tensor_64: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_65: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_107: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_65, 1.0000398612827361);  squeeze_dims_65 = None
        mul_tensor_108: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[384]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
        add_tensor_65: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(primals_130, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_66: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_dims_67: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_tensor_110: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 0.1)
        mul_tensor_111: "f32[384]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_tensor_67: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_68: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_tensor_112: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 1.0000398612827361);  squeeze_dims_68 = None
        mul_tensor_113: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[384]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
        add_tensor_68: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(primals_135, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_69: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_70: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 0.1)
        mul_tensor_116: "f32[384]" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_tensor_70: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_71: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_117: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 1.0000398612827361);  squeeze_dims_71 = None
        mul_tensor_118: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[384]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_tensor_71: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_72: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_dims_73: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_tensor_120: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_72, 0.1)
        mul_tensor_121: "f32[384]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_tensor_73: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_74: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_122: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 1.0000398612827361);  squeeze_dims_74 = None
        mul_tensor_123: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[384]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_tensor_74: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_75: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_dims_76: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_125: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_75, 0.1)
        mul_tensor_126: "f32[384]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_tensor_76: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_77: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_127: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 1.0000398612827361);  squeeze_dims_77 = None
        mul_tensor_128: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[384]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_tensor_77: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(primals_152, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_78: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_dims_79: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_78, 0.1)
        mul_tensor_131: "f32[384]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
        add_tensor_79: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_80: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_tensor_132: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 1.0000398612827361);  squeeze_dims_80 = None
        mul_tensor_133: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[384]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_tensor_80: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(primals_158, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_81: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_dims_82: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_tensor_135: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_81, 0.1)
        mul_tensor_136: "f32[384]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
        add_tensor_82: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_83: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_tensor_137: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_83, 1.0000398612827361);  squeeze_dims_83 = None
        mul_tensor_138: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.1);  mul_tensor_137 = None
        mul_tensor_139: "f32[384]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_tensor_83: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(primals_164, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_84: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_dims_85: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_tensor_140: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_84, 0.1)
        mul_tensor_141: "f32[384]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
        add_tensor_85: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_86: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_tensor_142: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_86, 1.0000398612827361);  squeeze_dims_86 = None
        mul_tensor_143: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.1);  mul_tensor_142 = None
        mul_tensor_144: "f32[384]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_tensor_86: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(primals_169, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_87: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_dims_88: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_tensor_145: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_87, 0.1)
        mul_tensor_146: "f32[384]" = torch.ops.aten.mul.Tensor(primals_170, 0.9)
        add_tensor_88: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_89: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_147: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_89, 1.0000398612827361);  squeeze_dims_89 = None
        mul_tensor_148: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.1);  mul_tensor_147 = None
        mul_tensor_149: "f32[384]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
        add_tensor_89: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(primals_175, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_90: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_dims_91: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_tensor_150: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_90, 0.1)
        mul_tensor_151: "f32[384]" = torch.ops.aten.mul.Tensor(primals_176, 0.9)
        add_tensor_91: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_92: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_152: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_92, 1.0000398612827361);  squeeze_dims_92 = None
        mul_tensor_153: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.1);  mul_tensor_152 = None
        mul_tensor_154: "f32[384]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
        add_tensor_92: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(primals_181, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_93: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_dims_94: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_tensor_155: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_93, 0.1)
        mul_tensor_156: "f32[384]" = torch.ops.aten.mul.Tensor(primals_182, 0.9)
        add_tensor_94: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_155, mul_tensor_156);  mul_tensor_155 = mul_tensor_156 = None
        squeeze_dims_95: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_157: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_95, 1.0000398612827361);  squeeze_dims_95 = None
        mul_tensor_158: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_157, 0.1);  mul_tensor_157 = None
        mul_tensor_159: "f32[384]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
        add_tensor_95: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_96: "i64[]" = torch.ops.aten.add.Tensor(primals_186, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_96: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_97: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_tensor_160: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_96, 0.1)
        mul_tensor_161: "f32[384]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        add_tensor_97: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        squeeze_dims_98: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_162: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_98, 1.0000398612827361);  squeeze_dims_98 = None
        mul_tensor_163: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_162, 0.1);  mul_tensor_162 = None
        mul_tensor_164: "f32[384]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
        add_tensor_98: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_163, mul_tensor_164);  mul_tensor_163 = mul_tensor_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_99: "i64[]" = torch.ops.aten.add.Tensor(primals_192, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_99: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_dims_100: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_tensor_165: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_99, 0.1)
        mul_tensor_166: "f32[384]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
        add_tensor_100: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_165, mul_tensor_166);  mul_tensor_165 = mul_tensor_166 = None
        squeeze_dims_101: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_167: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_101, 1.0000398612827361);  squeeze_dims_101 = None
        mul_tensor_168: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_167, 0.1);  mul_tensor_167 = None
        mul_tensor_169: "f32[384]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
        add_tensor_101: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_102: "i64[]" = torch.ops.aten.add.Tensor(primals_198, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_102: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_dims_103: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_tensor_170: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_102, 0.1)
        mul_tensor_171: "f32[384]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
        add_tensor_103: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        squeeze_dims_104: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_172: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_104, 1.0000398612827361);  squeeze_dims_104 = None
        mul_tensor_173: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_172, 0.1);  mul_tensor_172 = None
        mul_tensor_174: "f32[384]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
        add_tensor_104: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_173, mul_tensor_174);  mul_tensor_173 = mul_tensor_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_105: "i64[]" = torch.ops.aten.add.Tensor(primals_203, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_105: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_dims_106: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_tensor_175: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_105, 0.1)
        mul_tensor_176: "f32[384]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
        add_tensor_106: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_175, mul_tensor_176);  mul_tensor_175 = mul_tensor_176 = None
        squeeze_dims_107: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_tensor_177: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_107, 1.0000398612827361);  squeeze_dims_107 = None
        mul_tensor_178: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_177, 0.1);  mul_tensor_177 = None
        mul_tensor_179: "f32[384]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
        add_tensor_107: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_108: "i64[]" = torch.ops.aten.add.Tensor(primals_209, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_108: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_dims_109: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_tensor_180: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_108, 0.1)
        mul_tensor_181: "f32[384]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
        add_tensor_109: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        squeeze_dims_110: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_tensor_182: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_110, 1.0000398612827361);  squeeze_dims_110 = None
        mul_tensor_183: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_182, 0.1);  mul_tensor_182 = None
        mul_tensor_184: "f32[384]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
        add_tensor_110: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_183, mul_tensor_184);  mul_tensor_183 = mul_tensor_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_111: "i64[]" = torch.ops.aten.add.Tensor(primals_215, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_111: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_dims_112: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_tensor_185: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_111, 0.1)
        mul_tensor_186: "f32[384]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
        add_tensor_112: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_185, mul_tensor_186);  mul_tensor_185 = mul_tensor_186 = None
        squeeze_dims_113: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_tensor_187: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_113, 1.0000398612827361);  squeeze_dims_113 = None
        mul_tensor_188: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_187, 0.1);  mul_tensor_187 = None
        mul_tensor_189: "f32[384]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
        add_tensor_113: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_114: "i64[]" = torch.ops.aten.add.Tensor(primals_220, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_114: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_115: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_tensor_190: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_114, 0.1)
        mul_tensor_191: "f32[384]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_tensor_115: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        squeeze_dims_116: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_192: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_116, 1.0000398612827361);  squeeze_dims_116 = None
        mul_tensor_193: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_192, 0.1);  mul_tensor_192 = None
        mul_tensor_194: "f32[384]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
        add_tensor_116: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_193, mul_tensor_194);  mul_tensor_193 = mul_tensor_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_117: "i64[]" = torch.ops.aten.add.Tensor(primals_226, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_117: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_dims_118: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_tensor_195: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_117, 0.1)
        mul_tensor_196: "f32[384]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_tensor_118: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_195, mul_tensor_196);  mul_tensor_195 = mul_tensor_196 = None
        squeeze_dims_119: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_197: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_119, 1.0000398612827361);  squeeze_dims_119 = None
        mul_tensor_198: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_197, 0.1);  mul_tensor_197 = None
        mul_tensor_199: "f32[384]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
        add_tensor_119: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_120: "i64[]" = torch.ops.aten.add.Tensor(primals_232, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_120: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_dims_121: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_200: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_120, 0.1)
        mul_tensor_201: "f32[384]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_tensor_121: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        squeeze_dims_122: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_tensor_202: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_122, 1.0000398612827361);  squeeze_dims_122 = None
        mul_tensor_203: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_202, 0.1);  mul_tensor_202 = None
        mul_tensor_204: "f32[384]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
        add_tensor_122: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_203, mul_tensor_204);  mul_tensor_203 = mul_tensor_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_123: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_123: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_dims_124: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_205: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_123, 0.1)
        mul_tensor_206: "f32[384]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_tensor_124: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_205, mul_tensor_206);  mul_tensor_205 = mul_tensor_206 = None
        squeeze_dims_125: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_tensor_207: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_125, 1.0000398612827361);  squeeze_dims_125 = None
        mul_tensor_208: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_207, 0.1);  mul_tensor_207 = None
        mul_tensor_209: "f32[384]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_tensor_125: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_126: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_126: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_dims_127: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_tensor_210: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_126, 0.1)
        mul_tensor_211: "f32[384]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_tensor_127: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        squeeze_dims_128: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_tensor_212: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_128, 1.0000398612827361);  squeeze_dims_128 = None
        mul_tensor_213: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_212, 0.1);  mul_tensor_212 = None
        mul_tensor_214: "f32[384]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_tensor_128: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_129: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_129: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_dims_130: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_tensor_215: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_129, 0.1)
        mul_tensor_216: "f32[384]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_tensor_130: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_215, mul_tensor_216);  mul_tensor_215 = mul_tensor_216 = None
        squeeze_dims_131: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_tensor_217: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_131, 1.0000398612827361);  squeeze_dims_131 = None
        mul_tensor_218: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_217, 0.1);  mul_tensor_217 = None
        mul_tensor_219: "f32[384]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_tensor_131: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_132: "i64[]" = torch.ops.aten.add.Tensor(primals_254, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_132: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_dims_133: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_tensor_220: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_132, 0.1)
        mul_tensor_221: "f32[384]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
        add_tensor_133: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        squeeze_dims_134: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_tensor_222: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_134, 1.0000398612827361);  squeeze_dims_134 = None
        mul_tensor_223: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_222, 0.1);  mul_tensor_222 = None
        mul_tensor_224: "f32[384]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_tensor_134: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_223, mul_tensor_224);  mul_tensor_223 = mul_tensor_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_135: "i64[]" = torch.ops.aten.add.Tensor(primals_260, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_135: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_dims_136: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_tensor_225: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_135, 0.1)
        mul_tensor_226: "f32[384]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
        add_tensor_136: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_225, mul_tensor_226);  mul_tensor_225 = mul_tensor_226 = None
        squeeze_dims_137: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_tensor_227: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_137, 1.0000398612827361);  squeeze_dims_137 = None
        mul_tensor_228: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_227, 0.1);  mul_tensor_227 = None
        mul_tensor_229: "f32[384]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_tensor_137: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_138: "i64[]" = torch.ops.aten.add.Tensor(primals_266, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_138: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_dims_139: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_tensor_230: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_138, 0.1)
        mul_tensor_231: "f32[384]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
        add_tensor_139: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        squeeze_dims_140: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_tensor_232: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_140, 1.0000398612827361);  squeeze_dims_140 = None
        mul_tensor_233: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_232, 0.1);  mul_tensor_232 = None
        mul_tensor_234: "f32[384]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_tensor_140: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_233, mul_tensor_234);  mul_tensor_233 = mul_tensor_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_141: "i64[]" = torch.ops.aten.add.Tensor(primals_271, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_141: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_dims_142: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_tensor_235: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_141, 0.1)
        mul_tensor_236: "f32[384]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
        add_tensor_142: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        squeeze_dims_143: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_tensor_237: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_143, 1.0000398612827361);  squeeze_dims_143 = None
        mul_tensor_238: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_237, 0.1);  mul_tensor_237 = None
        mul_tensor_239: "f32[384]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
        add_tensor_143: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_144: "i64[]" = torch.ops.aten.add.Tensor(primals_277, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_144: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_dims_145: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_tensor_240: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_144, 0.1)
        mul_tensor_241: "f32[384]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
        add_tensor_145: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_240, mul_tensor_241);  mul_tensor_240 = mul_tensor_241 = None
        squeeze_dims_146: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_tensor_242: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_146, 1.0000398612827361);  squeeze_dims_146 = None
        mul_tensor_243: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_242, 0.1);  mul_tensor_242 = None
        mul_tensor_244: "f32[384]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
        add_tensor_146: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_243, mul_tensor_244);  mul_tensor_243 = mul_tensor_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_147: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_147: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_dims_148: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_tensor_245: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_147, 0.1)
        mul_tensor_246: "f32[384]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
        add_tensor_148: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_245, mul_tensor_246);  mul_tensor_245 = mul_tensor_246 = None
        squeeze_dims_149: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_tensor_247: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_149, 1.0000398612827361);  squeeze_dims_149 = None
        mul_tensor_248: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_247, 0.1);  mul_tensor_247 = None
        mul_tensor_249: "f32[384]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
        add_tensor_149: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_248, mul_tensor_249);  mul_tensor_248 = mul_tensor_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_150: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_150: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_dims_151: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_tensor_250: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_150, 0.1)
        mul_tensor_251: "f32[384]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
        add_tensor_151: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_250, mul_tensor_251);  mul_tensor_250 = mul_tensor_251 = None
        squeeze_dims_152: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_tensor_252: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_152, 1.0000398612827361);  squeeze_dims_152 = None
        mul_tensor_253: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_252, 0.1);  mul_tensor_252 = None
        mul_tensor_254: "f32[384]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
        add_tensor_152: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_253, mul_tensor_254);  mul_tensor_253 = mul_tensor_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_153: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_153: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_dims_154: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_tensor_255: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_153, 0.1)
        mul_tensor_256: "f32[384]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
        add_tensor_154: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_255, mul_tensor_256);  mul_tensor_255 = mul_tensor_256 = None
        squeeze_dims_155: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_tensor_257: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_155, 1.0000398612827361);  squeeze_dims_155 = None
        mul_tensor_258: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_257, 0.1);  mul_tensor_257 = None
        mul_tensor_259: "f32[384]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
        add_tensor_155: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_258, mul_tensor_259);  mul_tensor_258 = mul_tensor_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_156: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_156: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_dims_157: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_tensor_260: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_156, 0.1)
        mul_tensor_261: "f32[384]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
        add_tensor_157: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_260, mul_tensor_261);  mul_tensor_260 = mul_tensor_261 = None
        squeeze_dims_158: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_tensor_262: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_158, 1.0000398612827361);  squeeze_dims_158 = None
        mul_tensor_263: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_262, 0.1);  mul_tensor_262 = None
        mul_tensor_264: "f32[384]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
        add_tensor_158: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_263, mul_tensor_264);  mul_tensor_263 = mul_tensor_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_159: "i64[]" = torch.ops.aten.add.Tensor(primals_305, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_159: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_dims_160: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_tensor_265: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_159, 0.1)
        mul_tensor_266: "f32[384]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
        add_tensor_160: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_265, mul_tensor_266);  mul_tensor_265 = mul_tensor_266 = None
        squeeze_dims_161: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_tensor_267: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_161, 1.0000398612827361);  squeeze_dims_161 = None
        mul_tensor_268: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_267, 0.1);  mul_tensor_267 = None
        mul_tensor_269: "f32[384]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_tensor_161: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_268, mul_tensor_269);  mul_tensor_268 = mul_tensor_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_162: "i64[]" = torch.ops.aten.add.Tensor(primals_311, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_162: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_dims_163: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_tensor_270: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_162, 0.1)
        mul_tensor_271: "f32[384]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
        add_tensor_163: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_270, mul_tensor_271);  mul_tensor_270 = mul_tensor_271 = None
        squeeze_dims_164: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_tensor_272: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_164, 1.0000398612827361);  squeeze_dims_164 = None
        mul_tensor_273: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_272, 0.1);  mul_tensor_272 = None
        mul_tensor_274: "f32[384]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
        add_tensor_164: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_273, mul_tensor_274);  mul_tensor_273 = mul_tensor_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_165: "i64[]" = torch.ops.aten.add.Tensor(primals_317, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_165: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_dims_166: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_tensor_275: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_165, 0.1)
        mul_tensor_276: "f32[384]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
        add_tensor_166: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_275, mul_tensor_276);  mul_tensor_275 = mul_tensor_276 = None
        squeeze_dims_167: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_tensor_277: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_167, 1.0000398612827361);  squeeze_dims_167 = None
        mul_tensor_278: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_277, 0.1);  mul_tensor_277 = None
        mul_tensor_279: "f32[384]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
        add_tensor_167: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_278, mul_tensor_279);  mul_tensor_278 = mul_tensor_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_168: "i64[]" = torch.ops.aten.add.Tensor(primals_322, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_168: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_dims_169: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_tensor_280: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_168, 0.1)
        mul_tensor_281: "f32[384]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
        add_tensor_169: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_280, mul_tensor_281);  mul_tensor_280 = mul_tensor_281 = None
        squeeze_dims_170: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_tensor_282: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_170, 1.0000398612827361);  squeeze_dims_170 = None
        mul_tensor_283: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_282, 0.1);  mul_tensor_282 = None
        mul_tensor_284: "f32[384]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
        add_tensor_170: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_283, mul_tensor_284);  mul_tensor_283 = mul_tensor_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_171: "i64[]" = torch.ops.aten.add.Tensor(primals_328, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_171: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_dims_172: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_tensor_285: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_171, 0.1)
        mul_tensor_286: "f32[384]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_tensor_172: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_285, mul_tensor_286);  mul_tensor_285 = mul_tensor_286 = None
        squeeze_dims_173: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_tensor_287: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_173, 1.0000398612827361);  squeeze_dims_173 = None
        mul_tensor_288: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_287, 0.1);  mul_tensor_287 = None
        mul_tensor_289: "f32[384]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
        add_tensor_173: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_288, mul_tensor_289);  mul_tensor_288 = mul_tensor_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_174: "i64[]" = torch.ops.aten.add.Tensor(primals_334, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_174: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_dims_175: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_tensor_290: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_174, 0.1)
        mul_tensor_291: "f32[384]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_tensor_175: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_290, mul_tensor_291);  mul_tensor_290 = mul_tensor_291 = None
        squeeze_dims_176: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_tensor_292: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_176, 1.0000398612827361);  squeeze_dims_176 = None
        mul_tensor_293: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_292, 0.1);  mul_tensor_292 = None
        mul_tensor_294: "f32[384]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
        add_tensor_176: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_293, mul_tensor_294);  mul_tensor_293 = mul_tensor_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_177: "i64[]" = torch.ops.aten.add.Tensor(primals_340, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_118: "f32[1, 1408, 1, 1]" = var_mean_correction[0]
        getitem_119: "f32[1, 1408, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_178: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_default: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_178);  add_tensor_178 = None
        sub_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_119);  convolution_42 = None
        mul_tensor_295: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims_177: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        mul_tensor_296: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_177, 0.1);  squeeze_dims_177 = None
        mul_tensor_297: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_tensor_179: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_296, mul_tensor_297);  mul_tensor_296 = mul_tensor_297 = None
        squeeze_dims_178: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_tensor_298: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_178, 1.0001594642002871);  squeeze_dims_178 = None
        mul_tensor_299: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_298, 0.1);  mul_tensor_298 = None
        mul_tensor_300: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
        add_tensor_180: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_299, mul_tensor_300);  mul_tensor_299 = mul_tensor_300 = None
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_301: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_295, unsqueeze_default_1);  mul_tensor_295 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1);  primals_344 = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_181: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_301, unsqueeze_default_3);  mul_tensor_301 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_182: "i64[]" = torch.ops.aten.add.Tensor(primals_346, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_120: "f32[1, 1408, 1, 1]" = var_mean_correction_1[0]
        getitem_121: "f32[1, 1408, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_183: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_default_1: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_183);  add_tensor_183 = None
        sub_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_121);  convolution_43 = None
        mul_tensor_302: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_179: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        mul_tensor_303: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_179, 0.1);  squeeze_dims_179 = None
        mul_tensor_304: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
        add_tensor_184: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_303, mul_tensor_304);  mul_tensor_303 = mul_tensor_304 = None
        squeeze_dims_180: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_tensor_305: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_180, 1.0001594642002871);  squeeze_dims_180 = None
        mul_tensor_306: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_305, 0.1);  mul_tensor_305 = None
        mul_tensor_307: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
        add_tensor_185: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_306, mul_tensor_307);  mul_tensor_306 = mul_tensor_307 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_349, -1);  primals_349 = None
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_308: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_302, unsqueeze_default_5);  mul_tensor_302 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_350, -1);  primals_350 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_186: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_308, unsqueeze_default_7);  mul_tensor_308 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_tensor_187: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_181, add_tensor_186);  add_tensor_181 = add_tensor_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_187);  add_tensor_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 1408, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        as_strided_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 1408, 1, 1], [1408, 1, 1408, 1408]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 1408]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[1408, 1000]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_8: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_174, 0);  squeeze_dims_174 = None
        unsqueeze_default_9: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_171, 0);  squeeze_dims_171 = None
        unsqueeze_default_12: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_168, 0);  squeeze_dims_168 = None
        unsqueeze_default_15: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 3);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_165, 0);  squeeze_dims_165 = None
        unsqueeze_default_18: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 2);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 3);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_162, 0);  squeeze_dims_162 = None
        unsqueeze_default_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        unsqueeze_default_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_159, 0);  squeeze_dims_159 = None
        unsqueeze_default_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_156, 0);  squeeze_dims_156 = None
        unsqueeze_default_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_153, 0);  squeeze_dims_153 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        unsqueeze_default_32: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_150, 0);  squeeze_dims_150 = None
        unsqueeze_default_33: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 2);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_33, 3);  unsqueeze_default_33 = None
        unsqueeze_default_35: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_147, 0);  squeeze_dims_147 = None
        unsqueeze_default_36: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 2);  unsqueeze_default_35 = None
        unsqueeze_default_37: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 3);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_144, 0);  squeeze_dims_144 = None
        unsqueeze_default_39: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 3);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_141, 0);  squeeze_dims_141 = None
        unsqueeze_default_42: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 2);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 3);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_138, 0);  squeeze_dims_138 = None
        unsqueeze_default_45: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 3);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_135, 0);  squeeze_dims_135 = None
        unsqueeze_default_48: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 2);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 3);  unsqueeze_default_48 = None
        unsqueeze_default_50: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_132, 0);  squeeze_dims_132 = None
        unsqueeze_default_51: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 2);  unsqueeze_default_50 = None
        unsqueeze_default_52: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_51, 3);  unsqueeze_default_51 = None
        unsqueeze_default_53: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_129, 0);  squeeze_dims_129 = None
        unsqueeze_default_54: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 2);  unsqueeze_default_53 = None
        unsqueeze_default_55: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_54, 3);  unsqueeze_default_54 = None
        unsqueeze_default_56: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_126, 0);  squeeze_dims_126 = None
        unsqueeze_default_57: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 2);  unsqueeze_default_56 = None
        unsqueeze_default_58: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_57, 3);  unsqueeze_default_57 = None
        unsqueeze_default_59: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_123, 0);  squeeze_dims_123 = None
        unsqueeze_default_60: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 2);  unsqueeze_default_59 = None
        unsqueeze_default_61: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_60, 3);  unsqueeze_default_60 = None
        unsqueeze_default_62: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_120, 0);  squeeze_dims_120 = None
        unsqueeze_default_63: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 2);  unsqueeze_default_62 = None
        unsqueeze_default_64: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_63, 3);  unsqueeze_default_63 = None
        unsqueeze_default_65: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_117, 0);  squeeze_dims_117 = None
        unsqueeze_default_66: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_65, 2);  unsqueeze_default_65 = None
        unsqueeze_default_67: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_66, 3);  unsqueeze_default_66 = None
        unsqueeze_default_68: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_114, 0);  squeeze_dims_114 = None
        unsqueeze_default_69: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 2);  unsqueeze_default_68 = None
        unsqueeze_default_70: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_69, 3);  unsqueeze_default_69 = None
        unsqueeze_default_71: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_111, 0);  squeeze_dims_111 = None
        unsqueeze_default_72: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_71, 2);  unsqueeze_default_71 = None
        unsqueeze_default_73: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_72, 3);  unsqueeze_default_72 = None
        unsqueeze_default_74: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_108, 0);  squeeze_dims_108 = None
        unsqueeze_default_75: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 2);  unsqueeze_default_74 = None
        unsqueeze_default_76: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_75, 3);  unsqueeze_default_75 = None
        unsqueeze_default_77: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_105, 0);  squeeze_dims_105 = None
        unsqueeze_default_78: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_77, 2);  unsqueeze_default_77 = None
        unsqueeze_default_79: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_78, 3);  unsqueeze_default_78 = None
        unsqueeze_default_80: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_102, 0);  squeeze_dims_102 = None
        unsqueeze_default_81: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_80, 2);  unsqueeze_default_80 = None
        unsqueeze_default_82: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_81, 3);  unsqueeze_default_81 = None
        unsqueeze_default_83: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_99, 0);  squeeze_dims_99 = None
        unsqueeze_default_84: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_83, 2);  unsqueeze_default_83 = None
        unsqueeze_default_85: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_84, 3);  unsqueeze_default_84 = None
        unsqueeze_default_86: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_96, 0);  squeeze_dims_96 = None
        unsqueeze_default_87: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_86, 2);  unsqueeze_default_86 = None
        unsqueeze_default_88: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_87, 3);  unsqueeze_default_87 = None
        unsqueeze_default_89: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_93, 0);  squeeze_dims_93 = None
        unsqueeze_default_90: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_89, 2);  unsqueeze_default_89 = None
        unsqueeze_default_91: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_90, 3);  unsqueeze_default_90 = None
        unsqueeze_default_92: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_90, 0);  squeeze_dims_90 = None
        unsqueeze_default_93: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_92, 2);  unsqueeze_default_92 = None
        unsqueeze_default_94: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_93, 3);  unsqueeze_default_93 = None
        unsqueeze_default_95: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_87, 0);  squeeze_dims_87 = None
        unsqueeze_default_96: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_95, 2);  unsqueeze_default_95 = None
        unsqueeze_default_97: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_96, 3);  unsqueeze_default_96 = None
        unsqueeze_default_98: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_84, 0);  squeeze_dims_84 = None
        unsqueeze_default_99: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_98, 2);  unsqueeze_default_98 = None
        unsqueeze_default_100: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_99, 3);  unsqueeze_default_99 = None
        unsqueeze_default_101: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_81, 0);  squeeze_dims_81 = None
        unsqueeze_default_102: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_101, 2);  unsqueeze_default_101 = None
        unsqueeze_default_103: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_102, 3);  unsqueeze_default_102 = None
        unsqueeze_default_104: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_78, 0);  squeeze_dims_78 = None
        unsqueeze_default_105: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_104, 2);  unsqueeze_default_104 = None
        unsqueeze_default_106: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_105, 3);  unsqueeze_default_105 = None
        unsqueeze_default_107: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_75, 0);  squeeze_dims_75 = None
        unsqueeze_default_108: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_107, 2);  unsqueeze_default_107 = None
        unsqueeze_default_109: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_108, 3);  unsqueeze_default_108 = None
        unsqueeze_default_110: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_72, 0);  squeeze_dims_72 = None
        unsqueeze_default_111: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_110, 2);  unsqueeze_default_110 = None
        unsqueeze_default_112: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_111, 3);  unsqueeze_default_111 = None
        unsqueeze_default_113: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_69, 0);  squeeze_dims_69 = None
        unsqueeze_default_114: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_113, 2);  unsqueeze_default_113 = None
        unsqueeze_default_115: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_114, 3);  unsqueeze_default_114 = None
        unsqueeze_default_116: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_66, 0);  squeeze_dims_66 = None
        unsqueeze_default_117: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_116, 2);  unsqueeze_default_116 = None
        unsqueeze_default_118: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_117, 3);  unsqueeze_default_117 = None
        unsqueeze_default_119: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_63, 0);  squeeze_dims_63 = None
        unsqueeze_default_120: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_119, 2);  unsqueeze_default_119 = None
        unsqueeze_default_121: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_120, 3);  unsqueeze_default_120 = None
        unsqueeze_default_122: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_123: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_122, 2);  unsqueeze_default_122 = None
        unsqueeze_default_124: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_123, 3);  unsqueeze_default_123 = None
        unsqueeze_default_125: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_57, 0);  squeeze_dims_57 = None
        unsqueeze_default_126: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_125, 2);  unsqueeze_default_125 = None
        unsqueeze_default_127: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_126, 3);  unsqueeze_default_126 = None
        unsqueeze_default_128: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_54, 0);  squeeze_dims_54 = None
        unsqueeze_default_129: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_128, 2);  unsqueeze_default_128 = None
        unsqueeze_default_130: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_129, 3);  unsqueeze_default_129 = None
        unsqueeze_default_131: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_51, 0);  squeeze_dims_51 = None
        unsqueeze_default_132: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_131, 2);  unsqueeze_default_131 = None
        unsqueeze_default_133: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_132, 3);  unsqueeze_default_132 = None
        unsqueeze_default_134: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_48, 0);  squeeze_dims_48 = None
        unsqueeze_default_135: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_134, 2);  unsqueeze_default_134 = None
        unsqueeze_default_136: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_135, 3);  unsqueeze_default_135 = None
        unsqueeze_default_137: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_45, 0);  squeeze_dims_45 = None
        unsqueeze_default_138: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_137, 2);  unsqueeze_default_137 = None
        unsqueeze_default_139: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_138, 3);  unsqueeze_default_138 = None
        unsqueeze_default_140: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_42, 0);  squeeze_dims_42 = None
        unsqueeze_default_141: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_140, 2);  unsqueeze_default_140 = None
        unsqueeze_default_142: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_141, 3);  unsqueeze_default_141 = None
        unsqueeze_default_143: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_39, 0);  squeeze_dims_39 = None
        unsqueeze_default_144: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_143, 2);  unsqueeze_default_143 = None
        unsqueeze_default_145: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_144, 3);  unsqueeze_default_144 = None
        unsqueeze_default_146: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_36, 0);  squeeze_dims_36 = None
        unsqueeze_default_147: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_146, 2);  unsqueeze_default_146 = None
        unsqueeze_default_148: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_147, 3);  unsqueeze_default_147 = None
        unsqueeze_default_149: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_33, 0);  squeeze_dims_33 = None
        unsqueeze_default_150: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_149, 2);  unsqueeze_default_149 = None
        unsqueeze_default_151: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_150, 3);  unsqueeze_default_150 = None
        unsqueeze_default_152: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_30, 0);  squeeze_dims_30 = None
        unsqueeze_default_153: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_152, 2);  unsqueeze_default_152 = None
        unsqueeze_default_154: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_153, 3);  unsqueeze_default_153 = None
        unsqueeze_default_155: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_27, 0);  squeeze_dims_27 = None
        unsqueeze_default_156: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_155, 2);  unsqueeze_default_155 = None
        unsqueeze_default_157: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_156, 3);  unsqueeze_default_156 = None
        unsqueeze_default_158: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_24, 0);  squeeze_dims_24 = None
        unsqueeze_default_159: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_158, 2);  unsqueeze_default_158 = None
        unsqueeze_default_160: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_159, 3);  unsqueeze_default_159 = None
        unsqueeze_default_161: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_21, 0);  squeeze_dims_21 = None
        unsqueeze_default_162: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_161, 2);  unsqueeze_default_161 = None
        unsqueeze_default_163: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_162, 3);  unsqueeze_default_162 = None
        unsqueeze_default_164: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_18, 0);  squeeze_dims_18 = None
        unsqueeze_default_165: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_164, 2);  unsqueeze_default_164 = None
        unsqueeze_default_166: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_165, 3);  unsqueeze_default_165 = None
        unsqueeze_default_167: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_15, 0);  squeeze_dims_15 = None
        unsqueeze_default_168: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_167, 2);  unsqueeze_default_167 = None
        unsqueeze_default_169: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_168, 3);  unsqueeze_default_168 = None
        unsqueeze_default_170: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_12, 0);  squeeze_dims_12 = None
        unsqueeze_default_171: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_170, 2);  unsqueeze_default_170 = None
        unsqueeze_default_172: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_171, 3);  unsqueeze_default_171 = None
        unsqueeze_default_173: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_9, 0);  squeeze_dims_9 = None
        unsqueeze_default_174: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_173, 2);  unsqueeze_default_173 = None
        unsqueeze_default_175: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_174, 3);  unsqueeze_default_174 = None
        unsqueeze_default_176: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_177: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_176, 2);  unsqueeze_default_176 = None
        unsqueeze_default_178: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_177, 3);  unsqueeze_default_177 = None
        unsqueeze_default_179: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_180: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_179, 2);  unsqueeze_default_179 = None
        unsqueeze_default_181: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_180, 3);  unsqueeze_default_180 = None
        unsqueeze_default_182: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_183: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_182, 2);  unsqueeze_default_182 = None
        unsqueeze_default_184: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_183, 3);  unsqueeze_default_183 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_3, add_tensor);  primals_3 = add_tensor = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(primals_4, add_tensor_1);  primals_4 = add_tensor_1 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(primals_5, add_tensor_2);  primals_5 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_tensor_3);  primals_9 = add_tensor_3 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(primals_10, add_tensor_4);  primals_10 = add_tensor_4 = None
        copy__default_5: "f32[64]" = torch.ops.aten.copy_.default(primals_11, add_tensor_5);  primals_11 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_tensor_6);  primals_15 = add_tensor_6 = None
        copy__default_7: "f32[96]" = torch.ops.aten.copy_.default(primals_16, add_tensor_7);  primals_16 = add_tensor_7 = None
        copy__default_8: "f32[96]" = torch.ops.aten.copy_.default(primals_17, add_tensor_8);  primals_17 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_tensor_9);  primals_21 = add_tensor_9 = None
        copy__default_10: "f32[96]" = torch.ops.aten.copy_.default(primals_22, add_tensor_10);  primals_22 = add_tensor_10 = None
        copy__default_11: "f32[96]" = torch.ops.aten.copy_.default(primals_23, add_tensor_11);  primals_23 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(primals_26, add_tensor_12);  primals_26 = add_tensor_12 = None
        copy__default_13: "f32[96]" = torch.ops.aten.copy_.default(primals_27, add_tensor_13);  primals_27 = add_tensor_13 = None
        copy__default_14: "f32[96]" = torch.ops.aten.copy_.default(primals_28, add_tensor_14);  primals_28 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(primals_32, add_tensor_15);  primals_32 = add_tensor_15 = None
        copy__default_16: "f32[96]" = torch.ops.aten.copy_.default(primals_33, add_tensor_16);  primals_33 = add_tensor_16 = None
        copy__default_17: "f32[96]" = torch.ops.aten.copy_.default(primals_34, add_tensor_17);  primals_34 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(primals_38, add_tensor_18);  primals_38 = add_tensor_18 = None
        copy__default_19: "f32[96]" = torch.ops.aten.copy_.default(primals_39, add_tensor_19);  primals_39 = add_tensor_19 = None
        copy__default_20: "f32[96]" = torch.ops.aten.copy_.default(primals_40, add_tensor_20);  primals_40 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(primals_44, add_tensor_21);  primals_44 = add_tensor_21 = None
        copy__default_22: "f32[192]" = torch.ops.aten.copy_.default(primals_45, add_tensor_22);  primals_45 = add_tensor_22 = None
        copy__default_23: "f32[192]" = torch.ops.aten.copy_.default(primals_46, add_tensor_23);  primals_46 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(primals_50, add_tensor_24);  primals_50 = add_tensor_24 = None
        copy__default_25: "f32[192]" = torch.ops.aten.copy_.default(primals_51, add_tensor_25);  primals_51 = add_tensor_25 = None
        copy__default_26: "f32[192]" = torch.ops.aten.copy_.default(primals_52, add_tensor_26);  primals_52 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(primals_55, add_tensor_27);  primals_55 = add_tensor_27 = None
        copy__default_28: "f32[192]" = torch.ops.aten.copy_.default(primals_56, add_tensor_28);  primals_56 = add_tensor_28 = None
        copy__default_29: "f32[192]" = torch.ops.aten.copy_.default(primals_57, add_tensor_29);  primals_57 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(primals_61, add_tensor_30);  primals_61 = add_tensor_30 = None
        copy__default_31: "f32[192]" = torch.ops.aten.copy_.default(primals_62, add_tensor_31);  primals_62 = add_tensor_31 = None
        copy__default_32: "f32[192]" = torch.ops.aten.copy_.default(primals_63, add_tensor_32);  primals_63 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(primals_67, add_tensor_33);  primals_67 = add_tensor_33 = None
        copy__default_34: "f32[192]" = torch.ops.aten.copy_.default(primals_68, add_tensor_34);  primals_68 = add_tensor_34 = None
        copy__default_35: "f32[192]" = torch.ops.aten.copy_.default(primals_69, add_tensor_35);  primals_69 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(primals_72, add_tensor_36);  primals_72 = add_tensor_36 = None
        copy__default_37: "f32[192]" = torch.ops.aten.copy_.default(primals_73, add_tensor_37);  primals_73 = add_tensor_37 = None
        copy__default_38: "f32[192]" = torch.ops.aten.copy_.default(primals_74, add_tensor_38);  primals_74 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(primals_78, add_tensor_39);  primals_78 = add_tensor_39 = None
        copy__default_40: "f32[192]" = torch.ops.aten.copy_.default(primals_79, add_tensor_40);  primals_79 = add_tensor_40 = None
        copy__default_41: "f32[192]" = torch.ops.aten.copy_.default(primals_80, add_tensor_41);  primals_80 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(primals_84, add_tensor_42);  primals_84 = add_tensor_42 = None
        copy__default_43: "f32[192]" = torch.ops.aten.copy_.default(primals_85, add_tensor_43);  primals_85 = add_tensor_43 = None
        copy__default_44: "f32[192]" = torch.ops.aten.copy_.default(primals_86, add_tensor_44);  primals_86 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(primals_89, add_tensor_45);  primals_89 = add_tensor_45 = None
        copy__default_46: "f32[192]" = torch.ops.aten.copy_.default(primals_90, add_tensor_46);  primals_90 = add_tensor_46 = None
        copy__default_47: "f32[192]" = torch.ops.aten.copy_.default(primals_91, add_tensor_47);  primals_91 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(primals_95, add_tensor_48);  primals_95 = add_tensor_48 = None
        copy__default_49: "f32[192]" = torch.ops.aten.copy_.default(primals_96, add_tensor_49);  primals_96 = add_tensor_49 = None
        copy__default_50: "f32[192]" = torch.ops.aten.copy_.default(primals_97, add_tensor_50);  primals_97 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(primals_101, add_tensor_51);  primals_101 = add_tensor_51 = None
        copy__default_52: "f32[192]" = torch.ops.aten.copy_.default(primals_102, add_tensor_52);  primals_102 = add_tensor_52 = None
        copy__default_53: "f32[192]" = torch.ops.aten.copy_.default(primals_103, add_tensor_53);  primals_103 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(primals_107, add_tensor_54);  primals_107 = add_tensor_54 = None
        copy__default_55: "f32[384]" = torch.ops.aten.copy_.default(primals_108, add_tensor_55);  primals_108 = add_tensor_55 = None
        copy__default_56: "f32[384]" = torch.ops.aten.copy_.default(primals_109, add_tensor_56);  primals_109 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(primals_113, add_tensor_57);  primals_113 = add_tensor_57 = None
        copy__default_58: "f32[384]" = torch.ops.aten.copy_.default(primals_114, add_tensor_58);  primals_114 = add_tensor_58 = None
        copy__default_59: "f32[384]" = torch.ops.aten.copy_.default(primals_115, add_tensor_59);  primals_115 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(primals_118, add_tensor_60);  primals_118 = add_tensor_60 = None
        copy__default_61: "f32[384]" = torch.ops.aten.copy_.default(primals_119, add_tensor_61);  primals_119 = add_tensor_61 = None
        copy__default_62: "f32[384]" = torch.ops.aten.copy_.default(primals_120, add_tensor_62);  primals_120 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(primals_124, add_tensor_63);  primals_124 = add_tensor_63 = None
        copy__default_64: "f32[384]" = torch.ops.aten.copy_.default(primals_125, add_tensor_64);  primals_125 = add_tensor_64 = None
        copy__default_65: "f32[384]" = torch.ops.aten.copy_.default(primals_126, add_tensor_65);  primals_126 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(primals_130, add_tensor_66);  primals_130 = add_tensor_66 = None
        copy__default_67: "f32[384]" = torch.ops.aten.copy_.default(primals_131, add_tensor_67);  primals_131 = add_tensor_67 = None
        copy__default_68: "f32[384]" = torch.ops.aten.copy_.default(primals_132, add_tensor_68);  primals_132 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(primals_135, add_tensor_69);  primals_135 = add_tensor_69 = None
        copy__default_70: "f32[384]" = torch.ops.aten.copy_.default(primals_136, add_tensor_70);  primals_136 = add_tensor_70 = None
        copy__default_71: "f32[384]" = torch.ops.aten.copy_.default(primals_137, add_tensor_71);  primals_137 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(primals_141, add_tensor_72);  primals_141 = add_tensor_72 = None
        copy__default_73: "f32[384]" = torch.ops.aten.copy_.default(primals_142, add_tensor_73);  primals_142 = add_tensor_73 = None
        copy__default_74: "f32[384]" = torch.ops.aten.copy_.default(primals_143, add_tensor_74);  primals_143 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(primals_147, add_tensor_75);  primals_147 = add_tensor_75 = None
        copy__default_76: "f32[384]" = torch.ops.aten.copy_.default(primals_148, add_tensor_76);  primals_148 = add_tensor_76 = None
        copy__default_77: "f32[384]" = torch.ops.aten.copy_.default(primals_149, add_tensor_77);  primals_149 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(primals_152, add_tensor_78);  primals_152 = add_tensor_78 = None
        copy__default_79: "f32[384]" = torch.ops.aten.copy_.default(primals_153, add_tensor_79);  primals_153 = add_tensor_79 = None
        copy__default_80: "f32[384]" = torch.ops.aten.copy_.default(primals_154, add_tensor_80);  primals_154 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(primals_158, add_tensor_81);  primals_158 = add_tensor_81 = None
        copy__default_82: "f32[384]" = torch.ops.aten.copy_.default(primals_159, add_tensor_82);  primals_159 = add_tensor_82 = None
        copy__default_83: "f32[384]" = torch.ops.aten.copy_.default(primals_160, add_tensor_83);  primals_160 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(primals_164, add_tensor_84);  primals_164 = add_tensor_84 = None
        copy__default_85: "f32[384]" = torch.ops.aten.copy_.default(primals_165, add_tensor_85);  primals_165 = add_tensor_85 = None
        copy__default_86: "f32[384]" = torch.ops.aten.copy_.default(primals_166, add_tensor_86);  primals_166 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(primals_169, add_tensor_87);  primals_169 = add_tensor_87 = None
        copy__default_88: "f32[384]" = torch.ops.aten.copy_.default(primals_170, add_tensor_88);  primals_170 = add_tensor_88 = None
        copy__default_89: "f32[384]" = torch.ops.aten.copy_.default(primals_171, add_tensor_89);  primals_171 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(primals_175, add_tensor_90);  primals_175 = add_tensor_90 = None
        copy__default_91: "f32[384]" = torch.ops.aten.copy_.default(primals_176, add_tensor_91);  primals_176 = add_tensor_91 = None
        copy__default_92: "f32[384]" = torch.ops.aten.copy_.default(primals_177, add_tensor_92);  primals_177 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(primals_181, add_tensor_93);  primals_181 = add_tensor_93 = None
        copy__default_94: "f32[384]" = torch.ops.aten.copy_.default(primals_182, add_tensor_94);  primals_182 = add_tensor_94 = None
        copy__default_95: "f32[384]" = torch.ops.aten.copy_.default(primals_183, add_tensor_95);  primals_183 = add_tensor_95 = None
        copy__default_96: "i64[]" = torch.ops.aten.copy_.default(primals_186, add_tensor_96);  primals_186 = add_tensor_96 = None
        copy__default_97: "f32[384]" = torch.ops.aten.copy_.default(primals_187, add_tensor_97);  primals_187 = add_tensor_97 = None
        copy__default_98: "f32[384]" = torch.ops.aten.copy_.default(primals_188, add_tensor_98);  primals_188 = add_tensor_98 = None
        copy__default_99: "i64[]" = torch.ops.aten.copy_.default(primals_192, add_tensor_99);  primals_192 = add_tensor_99 = None
        copy__default_100: "f32[384]" = torch.ops.aten.copy_.default(primals_193, add_tensor_100);  primals_193 = add_tensor_100 = None
        copy__default_101: "f32[384]" = torch.ops.aten.copy_.default(primals_194, add_tensor_101);  primals_194 = add_tensor_101 = None
        copy__default_102: "i64[]" = torch.ops.aten.copy_.default(primals_198, add_tensor_102);  primals_198 = add_tensor_102 = None
        copy__default_103: "f32[384]" = torch.ops.aten.copy_.default(primals_199, add_tensor_103);  primals_199 = add_tensor_103 = None
        copy__default_104: "f32[384]" = torch.ops.aten.copy_.default(primals_200, add_tensor_104);  primals_200 = add_tensor_104 = None
        copy__default_105: "i64[]" = torch.ops.aten.copy_.default(primals_203, add_tensor_105);  primals_203 = add_tensor_105 = None
        copy__default_106: "f32[384]" = torch.ops.aten.copy_.default(primals_204, add_tensor_106);  primals_204 = add_tensor_106 = None
        copy__default_107: "f32[384]" = torch.ops.aten.copy_.default(primals_205, add_tensor_107);  primals_205 = add_tensor_107 = None
        copy__default_108: "i64[]" = torch.ops.aten.copy_.default(primals_209, add_tensor_108);  primals_209 = add_tensor_108 = None
        copy__default_109: "f32[384]" = torch.ops.aten.copy_.default(primals_210, add_tensor_109);  primals_210 = add_tensor_109 = None
        copy__default_110: "f32[384]" = torch.ops.aten.copy_.default(primals_211, add_tensor_110);  primals_211 = add_tensor_110 = None
        copy__default_111: "i64[]" = torch.ops.aten.copy_.default(primals_215, add_tensor_111);  primals_215 = add_tensor_111 = None
        copy__default_112: "f32[384]" = torch.ops.aten.copy_.default(primals_216, add_tensor_112);  primals_216 = add_tensor_112 = None
        copy__default_113: "f32[384]" = torch.ops.aten.copy_.default(primals_217, add_tensor_113);  primals_217 = add_tensor_113 = None
        copy__default_114: "i64[]" = torch.ops.aten.copy_.default(primals_220, add_tensor_114);  primals_220 = add_tensor_114 = None
        copy__default_115: "f32[384]" = torch.ops.aten.copy_.default(primals_221, add_tensor_115);  primals_221 = add_tensor_115 = None
        copy__default_116: "f32[384]" = torch.ops.aten.copy_.default(primals_222, add_tensor_116);  primals_222 = add_tensor_116 = None
        copy__default_117: "i64[]" = torch.ops.aten.copy_.default(primals_226, add_tensor_117);  primals_226 = add_tensor_117 = None
        copy__default_118: "f32[384]" = torch.ops.aten.copy_.default(primals_227, add_tensor_118);  primals_227 = add_tensor_118 = None
        copy__default_119: "f32[384]" = torch.ops.aten.copy_.default(primals_228, add_tensor_119);  primals_228 = add_tensor_119 = None
        copy__default_120: "i64[]" = torch.ops.aten.copy_.default(primals_232, add_tensor_120);  primals_232 = add_tensor_120 = None
        copy__default_121: "f32[384]" = torch.ops.aten.copy_.default(primals_233, add_tensor_121);  primals_233 = add_tensor_121 = None
        copy__default_122: "f32[384]" = torch.ops.aten.copy_.default(primals_234, add_tensor_122);  primals_234 = add_tensor_122 = None
        copy__default_123: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_tensor_123);  primals_237 = add_tensor_123 = None
        copy__default_124: "f32[384]" = torch.ops.aten.copy_.default(primals_238, add_tensor_124);  primals_238 = add_tensor_124 = None
        copy__default_125: "f32[384]" = torch.ops.aten.copy_.default(primals_239, add_tensor_125);  primals_239 = add_tensor_125 = None
        copy__default_126: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_tensor_126);  primals_243 = add_tensor_126 = None
        copy__default_127: "f32[384]" = torch.ops.aten.copy_.default(primals_244, add_tensor_127);  primals_244 = add_tensor_127 = None
        copy__default_128: "f32[384]" = torch.ops.aten.copy_.default(primals_245, add_tensor_128);  primals_245 = add_tensor_128 = None
        copy__default_129: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_tensor_129);  primals_249 = add_tensor_129 = None
        copy__default_130: "f32[384]" = torch.ops.aten.copy_.default(primals_250, add_tensor_130);  primals_250 = add_tensor_130 = None
        copy__default_131: "f32[384]" = torch.ops.aten.copy_.default(primals_251, add_tensor_131);  primals_251 = add_tensor_131 = None
        copy__default_132: "i64[]" = torch.ops.aten.copy_.default(primals_254, add_tensor_132);  primals_254 = add_tensor_132 = None
        copy__default_133: "f32[384]" = torch.ops.aten.copy_.default(primals_255, add_tensor_133);  primals_255 = add_tensor_133 = None
        copy__default_134: "f32[384]" = torch.ops.aten.copy_.default(primals_256, add_tensor_134);  primals_256 = add_tensor_134 = None
        copy__default_135: "i64[]" = torch.ops.aten.copy_.default(primals_260, add_tensor_135);  primals_260 = add_tensor_135 = None
        copy__default_136: "f32[384]" = torch.ops.aten.copy_.default(primals_261, add_tensor_136);  primals_261 = add_tensor_136 = None
        copy__default_137: "f32[384]" = torch.ops.aten.copy_.default(primals_262, add_tensor_137);  primals_262 = add_tensor_137 = None
        copy__default_138: "i64[]" = torch.ops.aten.copy_.default(primals_266, add_tensor_138);  primals_266 = add_tensor_138 = None
        copy__default_139: "f32[384]" = torch.ops.aten.copy_.default(primals_267, add_tensor_139);  primals_267 = add_tensor_139 = None
        copy__default_140: "f32[384]" = torch.ops.aten.copy_.default(primals_268, add_tensor_140);  primals_268 = add_tensor_140 = None
        copy__default_141: "i64[]" = torch.ops.aten.copy_.default(primals_271, add_tensor_141);  primals_271 = add_tensor_141 = None
        copy__default_142: "f32[384]" = torch.ops.aten.copy_.default(primals_272, add_tensor_142);  primals_272 = add_tensor_142 = None
        copy__default_143: "f32[384]" = torch.ops.aten.copy_.default(primals_273, add_tensor_143);  primals_273 = add_tensor_143 = None
        copy__default_144: "i64[]" = torch.ops.aten.copy_.default(primals_277, add_tensor_144);  primals_277 = add_tensor_144 = None
        copy__default_145: "f32[384]" = torch.ops.aten.copy_.default(primals_278, add_tensor_145);  primals_278 = add_tensor_145 = None
        copy__default_146: "f32[384]" = torch.ops.aten.copy_.default(primals_279, add_tensor_146);  primals_279 = add_tensor_146 = None
        copy__default_147: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_tensor_147);  primals_283 = add_tensor_147 = None
        copy__default_148: "f32[384]" = torch.ops.aten.copy_.default(primals_284, add_tensor_148);  primals_284 = add_tensor_148 = None
        copy__default_149: "f32[384]" = torch.ops.aten.copy_.default(primals_285, add_tensor_149);  primals_285 = add_tensor_149 = None
        copy__default_150: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_tensor_150);  primals_288 = add_tensor_150 = None
        copy__default_151: "f32[384]" = torch.ops.aten.copy_.default(primals_289, add_tensor_151);  primals_289 = add_tensor_151 = None
        copy__default_152: "f32[384]" = torch.ops.aten.copy_.default(primals_290, add_tensor_152);  primals_290 = add_tensor_152 = None
        copy__default_153: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_tensor_153);  primals_294 = add_tensor_153 = None
        copy__default_154: "f32[384]" = torch.ops.aten.copy_.default(primals_295, add_tensor_154);  primals_295 = add_tensor_154 = None
        copy__default_155: "f32[384]" = torch.ops.aten.copy_.default(primals_296, add_tensor_155);  primals_296 = add_tensor_155 = None
        copy__default_156: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_tensor_156);  primals_300 = add_tensor_156 = None
        copy__default_157: "f32[384]" = torch.ops.aten.copy_.default(primals_301, add_tensor_157);  primals_301 = add_tensor_157 = None
        copy__default_158: "f32[384]" = torch.ops.aten.copy_.default(primals_302, add_tensor_158);  primals_302 = add_tensor_158 = None
        copy__default_159: "i64[]" = torch.ops.aten.copy_.default(primals_305, add_tensor_159);  primals_305 = add_tensor_159 = None
        copy__default_160: "f32[384]" = torch.ops.aten.copy_.default(primals_306, add_tensor_160);  primals_306 = add_tensor_160 = None
        copy__default_161: "f32[384]" = torch.ops.aten.copy_.default(primals_307, add_tensor_161);  primals_307 = add_tensor_161 = None
        copy__default_162: "i64[]" = torch.ops.aten.copy_.default(primals_311, add_tensor_162);  primals_311 = add_tensor_162 = None
        copy__default_163: "f32[384]" = torch.ops.aten.copy_.default(primals_312, add_tensor_163);  primals_312 = add_tensor_163 = None
        copy__default_164: "f32[384]" = torch.ops.aten.copy_.default(primals_313, add_tensor_164);  primals_313 = add_tensor_164 = None
        copy__default_165: "i64[]" = torch.ops.aten.copy_.default(primals_317, add_tensor_165);  primals_317 = add_tensor_165 = None
        copy__default_166: "f32[384]" = torch.ops.aten.copy_.default(primals_318, add_tensor_166);  primals_318 = add_tensor_166 = None
        copy__default_167: "f32[384]" = torch.ops.aten.copy_.default(primals_319, add_tensor_167);  primals_319 = add_tensor_167 = None
        copy__default_168: "i64[]" = torch.ops.aten.copy_.default(primals_322, add_tensor_168);  primals_322 = add_tensor_168 = None
        copy__default_169: "f32[384]" = torch.ops.aten.copy_.default(primals_323, add_tensor_169);  primals_323 = add_tensor_169 = None
        copy__default_170: "f32[384]" = torch.ops.aten.copy_.default(primals_324, add_tensor_170);  primals_324 = add_tensor_170 = None
        copy__default_171: "i64[]" = torch.ops.aten.copy_.default(primals_328, add_tensor_171);  primals_328 = add_tensor_171 = None
        copy__default_172: "f32[384]" = torch.ops.aten.copy_.default(primals_329, add_tensor_172);  primals_329 = add_tensor_172 = None
        copy__default_173: "f32[384]" = torch.ops.aten.copy_.default(primals_330, add_tensor_173);  primals_330 = add_tensor_173 = None
        copy__default_174: "i64[]" = torch.ops.aten.copy_.default(primals_334, add_tensor_174);  primals_334 = add_tensor_174 = None
        copy__default_175: "f32[384]" = torch.ops.aten.copy_.default(primals_335, add_tensor_175);  primals_335 = add_tensor_175 = None
        copy__default_176: "f32[384]" = torch.ops.aten.copy_.default(primals_336, add_tensor_176);  primals_336 = add_tensor_176 = None
        copy__default_177: "i64[]" = torch.ops.aten.copy_.default(primals_340, add_tensor_177);  primals_340 = add_tensor_177 = None
        copy__default_178: "f32[1408]" = torch.ops.aten.copy_.default(primals_341, add_tensor_179);  primals_341 = add_tensor_179 = None
        copy__default_179: "f32[1408]" = torch.ops.aten.copy_.default(primals_342, add_tensor_180);  primals_342 = add_tensor_180 = None
        copy__default_180: "i64[]" = torch.ops.aten.copy_.default(primals_346, add_tensor_182);  primals_346 = add_tensor_182 = None
        copy__default_181: "f32[1408]" = torch.ops.aten.copy_.default(primals_347, add_tensor_184);  primals_347 = add_tensor_184 = None
        copy__default_182: "f32[1408]" = torch.ops.aten.copy_.default(primals_348, add_tensor_185);  primals_348 = add_tensor_185 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_7, squeeze_dims_10, squeeze_dims_13, squeeze_dims_16, squeeze_dims_19, squeeze_dims_22, squeeze_dims_25, squeeze_dims_28, squeeze_dims_31, squeeze_dims_34, squeeze_dims_37, squeeze_dims_40, squeeze_dims_43, squeeze_dims_46, squeeze_dims_49, squeeze_dims_52, squeeze_dims_55, squeeze_dims_58, squeeze_dims_61, squeeze_dims_64, squeeze_dims_67, squeeze_dims_70, squeeze_dims_73, squeeze_dims_76, squeeze_dims_79, squeeze_dims_82, squeeze_dims_85, squeeze_dims_88, squeeze_dims_91, squeeze_dims_94, squeeze_dims_97, squeeze_dims_100, squeeze_dims_103, squeeze_dims_106, squeeze_dims_109, squeeze_dims_112, squeeze_dims_115, squeeze_dims_118, squeeze_dims_121, squeeze_dims_124, squeeze_dims_127, squeeze_dims_130, squeeze_dims_133, squeeze_dims_136, squeeze_dims_139, squeeze_dims_142, squeeze_dims_145, squeeze_dims_148, squeeze_dims_151, squeeze_dims_154, squeeze_dims_157, squeeze_dims_160, squeeze_dims_163, squeeze_dims_166, squeeze_dims_169, squeeze_dims_172, squeeze_dims_175, reshape_default, permute_default, unsqueeze_default_10, unsqueeze_default_13, unsqueeze_default_16, unsqueeze_default_19, unsqueeze_default_22, unsqueeze_default_25, unsqueeze_default_28, unsqueeze_default_31, unsqueeze_default_34, unsqueeze_default_37, unsqueeze_default_40, unsqueeze_default_43, unsqueeze_default_46, unsqueeze_default_49, unsqueeze_default_52, unsqueeze_default_55, unsqueeze_default_58, unsqueeze_default_61, unsqueeze_default_64, unsqueeze_default_67, unsqueeze_default_70, unsqueeze_default_73, unsqueeze_default_76, unsqueeze_default_79, unsqueeze_default_82, unsqueeze_default_85, unsqueeze_default_88, unsqueeze_default_91, unsqueeze_default_94, unsqueeze_default_97, unsqueeze_default_100, unsqueeze_default_103, unsqueeze_default_106, unsqueeze_default_109, unsqueeze_default_112, unsqueeze_default_115, unsqueeze_default_118, unsqueeze_default_121, unsqueeze_default_124, unsqueeze_default_127, unsqueeze_default_130, unsqueeze_default_133, unsqueeze_default_136, unsqueeze_default_139, unsqueeze_default_142, unsqueeze_default_145, unsqueeze_default_148, unsqueeze_default_151, unsqueeze_default_154, unsqueeze_default_157, unsqueeze_default_160, unsqueeze_default_163, unsqueeze_default_166, unsqueeze_default_169, unsqueeze_default_172, unsqueeze_default_175, unsqueeze_default_178, unsqueeze_default_181, unsqueeze_default_184, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160, copy__default_161, copy__default_162, copy__default_163, copy__default_164, copy__default_165, copy__default_166, copy__default_167, copy__default_168, copy__default_169, copy__default_170, copy__default_171, copy__default_172, copy__default_173, copy__default_174, copy__default_175, copy__default_176, copy__default_177, copy__default_178, copy__default_179, copy__default_180, copy__default_181, copy__default_182)
