class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "i64[]", getitem_1: "f32[1, 32, 1, 1]", primals_4: "f32[32]", getitem: "f32[1, 32, 1, 1]", primals_5: "f32[32]", primals_9: "i64[]", getitem_3: "f32[1, 32, 1, 1]", primals_10: "f32[32]", getitem_2: "f32[1, 32, 1, 1]", primals_11: "f32[32]", primals_19: "i64[]", getitem_5: "f32[1, 16, 1, 1]", rsqrt_2: "f32[1, 16, 1, 1]", primals_20: "f32[16]", getitem_4: "f32[1, 16, 1, 1]", primals_21: "f32[16]", primals_25: "i64[]", getitem_7: "f32[1, 96, 1, 1]", primals_26: "f32[96]", getitem_6: "f32[1, 96, 1, 1]", primals_27: "f32[96]", primals_31: "i64[]", getitem_9: "f32[1, 96, 1, 1]", primals_32: "f32[96]", getitem_8: "f32[1, 96, 1, 1]", primals_33: "f32[96]", primals_41: "i64[]", getitem_11: "f32[1, 24, 1, 1]", rsqrt_5: "f32[1, 24, 1, 1]", primals_42: "f32[24]", getitem_10: "f32[1, 24, 1, 1]", primals_43: "f32[24]", primals_47: "i64[]", getitem_13: "f32[1, 144, 1, 1]", primals_48: "f32[144]", getitem_12: "f32[1, 144, 1, 1]", primals_49: "f32[144]", primals_53: "i64[]", getitem_15: "f32[1, 144, 1, 1]", primals_54: "f32[144]", getitem_14: "f32[1, 144, 1, 1]", primals_55: "f32[144]", primals_63: "i64[]", getitem_17: "f32[1, 24, 1, 1]", rsqrt_8: "f32[1, 24, 1, 1]", primals_64: "f32[24]", getitem_16: "f32[1, 24, 1, 1]", primals_65: "f32[24]", primals_69: "i64[]", getitem_19: "f32[1, 144, 1, 1]", primals_70: "f32[144]", getitem_18: "f32[1, 144, 1, 1]", primals_71: "f32[144]", primals_75: "i64[]", getitem_21: "f32[1, 144, 1, 1]", primals_76: "f32[144]", getitem_20: "f32[1, 144, 1, 1]", primals_77: "f32[144]", primals_85: "i64[]", getitem_23: "f32[1, 40, 1, 1]", rsqrt_11: "f32[1, 40, 1, 1]", primals_86: "f32[40]", getitem_22: "f32[1, 40, 1, 1]", primals_87: "f32[40]", primals_91: "i64[]", getitem_25: "f32[1, 240, 1, 1]", primals_92: "f32[240]", getitem_24: "f32[1, 240, 1, 1]", primals_93: "f32[240]", primals_97: "i64[]", getitem_27: "f32[1, 240, 1, 1]", primals_98: "f32[240]", getitem_26: "f32[1, 240, 1, 1]", primals_99: "f32[240]", primals_107: "i64[]", getitem_29: "f32[1, 40, 1, 1]", rsqrt_14: "f32[1, 40, 1, 1]", primals_108: "f32[40]", getitem_28: "f32[1, 40, 1, 1]", primals_109: "f32[40]", primals_113: "i64[]", getitem_31: "f32[1, 240, 1, 1]", primals_114: "f32[240]", getitem_30: "f32[1, 240, 1, 1]", primals_115: "f32[240]", primals_119: "i64[]", getitem_33: "f32[1, 240, 1, 1]", primals_120: "f32[240]", getitem_32: "f32[1, 240, 1, 1]", primals_121: "f32[240]", primals_129: "i64[]", getitem_35: "f32[1, 80, 1, 1]", rsqrt_17: "f32[1, 80, 1, 1]", primals_130: "f32[80]", getitem_34: "f32[1, 80, 1, 1]", primals_131: "f32[80]", primals_135: "i64[]", getitem_37: "f32[1, 480, 1, 1]", primals_136: "f32[480]", getitem_36: "f32[1, 480, 1, 1]", primals_137: "f32[480]", primals_141: "i64[]", getitem_39: "f32[1, 480, 1, 1]", primals_142: "f32[480]", getitem_38: "f32[1, 480, 1, 1]", primals_143: "f32[480]", primals_151: "i64[]", getitem_41: "f32[1, 80, 1, 1]", rsqrt_20: "f32[1, 80, 1, 1]", primals_152: "f32[80]", getitem_40: "f32[1, 80, 1, 1]", primals_153: "f32[80]", primals_157: "i64[]", getitem_43: "f32[1, 480, 1, 1]", primals_158: "f32[480]", getitem_42: "f32[1, 480, 1, 1]", primals_159: "f32[480]", primals_163: "i64[]", getitem_45: "f32[1, 480, 1, 1]", primals_164: "f32[480]", getitem_44: "f32[1, 480, 1, 1]", primals_165: "f32[480]", primals_173: "i64[]", getitem_47: "f32[1, 80, 1, 1]", rsqrt_23: "f32[1, 80, 1, 1]", primals_174: "f32[80]", getitem_46: "f32[1, 80, 1, 1]", primals_175: "f32[80]", primals_179: "i64[]", getitem_49: "f32[1, 480, 1, 1]", primals_180: "f32[480]", getitem_48: "f32[1, 480, 1, 1]", primals_181: "f32[480]", primals_185: "i64[]", getitem_51: "f32[1, 480, 1, 1]", primals_186: "f32[480]", getitem_50: "f32[1, 480, 1, 1]", primals_187: "f32[480]", primals_195: "i64[]", getitem_53: "f32[1, 112, 1, 1]", rsqrt_26: "f32[1, 112, 1, 1]", primals_196: "f32[112]", getitem_52: "f32[1, 112, 1, 1]", primals_197: "f32[112]", primals_201: "i64[]", getitem_55: "f32[1, 672, 1, 1]", primals_202: "f32[672]", getitem_54: "f32[1, 672, 1, 1]", primals_203: "f32[672]", primals_207: "i64[]", getitem_57: "f32[1, 672, 1, 1]", primals_208: "f32[672]", getitem_56: "f32[1, 672, 1, 1]", primals_209: "f32[672]", primals_217: "i64[]", getitem_59: "f32[1, 112, 1, 1]", rsqrt_29: "f32[1, 112, 1, 1]", primals_218: "f32[112]", getitem_58: "f32[1, 112, 1, 1]", primals_219: "f32[112]", primals_223: "i64[]", getitem_61: "f32[1, 672, 1, 1]", primals_224: "f32[672]", getitem_60: "f32[1, 672, 1, 1]", primals_225: "f32[672]", primals_229: "i64[]", getitem_63: "f32[1, 672, 1, 1]", primals_230: "f32[672]", getitem_62: "f32[1, 672, 1, 1]", primals_231: "f32[672]", primals_239: "i64[]", getitem_65: "f32[1, 112, 1, 1]", rsqrt_32: "f32[1, 112, 1, 1]", primals_240: "f32[112]", getitem_64: "f32[1, 112, 1, 1]", primals_241: "f32[112]", primals_245: "i64[]", getitem_67: "f32[1, 672, 1, 1]", primals_246: "f32[672]", getitem_66: "f32[1, 672, 1, 1]", primals_247: "f32[672]", primals_251: "i64[]", getitem_69: "f32[1, 672, 1, 1]", primals_252: "f32[672]", getitem_68: "f32[1, 672, 1, 1]", primals_253: "f32[672]", primals_261: "i64[]", getitem_71: "f32[1, 192, 1, 1]", rsqrt_35: "f32[1, 192, 1, 1]", primals_262: "f32[192]", getitem_70: "f32[1, 192, 1, 1]", primals_263: "f32[192]", primals_267: "i64[]", getitem_73: "f32[1, 1152, 1, 1]", primals_268: "f32[1152]", getitem_72: "f32[1, 1152, 1, 1]", primals_269: "f32[1152]", primals_273: "i64[]", getitem_75: "f32[1, 1152, 1, 1]", primals_274: "f32[1152]", getitem_74: "f32[1, 1152, 1, 1]", primals_275: "f32[1152]", primals_283: "i64[]", getitem_77: "f32[1, 192, 1, 1]", rsqrt_38: "f32[1, 192, 1, 1]", primals_284: "f32[192]", getitem_76: "f32[1, 192, 1, 1]", primals_285: "f32[192]", primals_289: "i64[]", getitem_79: "f32[1, 1152, 1, 1]", primals_290: "f32[1152]", getitem_78: "f32[1, 1152, 1, 1]", primals_291: "f32[1152]", primals_295: "i64[]", getitem_81: "f32[1, 1152, 1, 1]", primals_296: "f32[1152]", getitem_80: "f32[1, 1152, 1, 1]", primals_297: "f32[1152]", primals_305: "i64[]", getitem_83: "f32[1, 192, 1, 1]", rsqrt_41: "f32[1, 192, 1, 1]", primals_306: "f32[192]", getitem_82: "f32[1, 192, 1, 1]", primals_307: "f32[192]", primals_311: "i64[]", getitem_85: "f32[1, 1152, 1, 1]", primals_312: "f32[1152]", getitem_84: "f32[1, 1152, 1, 1]", primals_313: "f32[1152]", primals_317: "i64[]", getitem_87: "f32[1, 1152, 1, 1]", primals_318: "f32[1152]", getitem_86: "f32[1, 1152, 1, 1]", primals_319: "f32[1152]", primals_327: "i64[]", getitem_89: "f32[1, 192, 1, 1]", rsqrt_44: "f32[1, 192, 1, 1]", primals_328: "f32[192]", getitem_88: "f32[1, 192, 1, 1]", primals_329: "f32[192]", primals_333: "i64[]", getitem_91: "f32[1, 1152, 1, 1]", primals_334: "f32[1152]", getitem_90: "f32[1, 1152, 1, 1]", primals_335: "f32[1152]", primals_339: "i64[]", getitem_93: "f32[1, 1152, 1, 1]", primals_340: "f32[1152]", getitem_92: "f32[1, 1152, 1, 1]", primals_341: "f32[1152]", primals_349: "i64[]", getitem_95: "f32[1, 320, 1, 1]", rsqrt_47: "f32[1, 320, 1, 1]", primals_350: "f32[320]", getitem_94: "f32[1, 320, 1, 1]", primals_351: "f32[320]", primals_355: "i64[]", convolution_80: "f32[128, 1280, 7, 7]", primals_356: "f32[1280]", primals_357: "f32[1280]", primals_358: "f32[1280]", primals_359: "f32[1280]", primals_360: "f32[1000, 1280]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_1: "f32[32]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_tensor_1: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_1: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000006228081046);  squeeze_dims_1 = None
        mul_tensor_3: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[32]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_5: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_6: "f32[32]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_tensor_4: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0000006228081046);  squeeze_dims_3 = None
        mul_tensor_8: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[32]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_tensor_5: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_19, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_4: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_5: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 0.1)
        mul_tensor_11: "f32[16]" = torch.ops.aten.mul.Tensor(primals_20, 0.9)
        add_tensor_7: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 1.0000006228081046);  squeeze_dims_6 = None
        mul_tensor_13: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[16]" = torch.ops.aten.mul.Tensor(primals_21, 0.9)
        add_tensor_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(primals_25, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_7: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        mul_tensor_15: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 0.1);  squeeze_dims_7 = None
        mul_tensor_16: "f32[96]" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_tensor_10: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_8: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.0000006228081046);  squeeze_dims_8 = None
        mul_tensor_18: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[96]" = torch.ops.aten.mul.Tensor(primals_27, 0.9)
        add_tensor_11: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(primals_31, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_9: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        mul_tensor_20: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 0.1);  squeeze_dims_9 = None
        mul_tensor_21: "f32[96]" = torch.ops.aten.mul.Tensor(primals_32, 0.9)
        add_tensor_13: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_10: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 1.0000024912370735);  squeeze_dims_10 = None
        mul_tensor_23: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[96]" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_tensor_14: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(primals_41, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_11: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_12: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 0.1)
        mul_tensor_26: "f32[24]" = torch.ops.aten.mul.Tensor(primals_42, 0.9)
        add_tensor_16: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_13: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_13, 1.0000024912370735);  squeeze_dims_13 = None
        mul_tensor_28: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[24]" = torch.ops.aten.mul.Tensor(primals_43, 0.9)
        add_tensor_17: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(primals_47, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_14: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        mul_tensor_30: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 0.1);  squeeze_dims_14 = None
        mul_tensor_31: "f32[144]" = torch.ops.aten.mul.Tensor(primals_48, 0.9)
        add_tensor_19: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_15: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_15, 1.0000024912370735);  squeeze_dims_15 = None
        mul_tensor_33: "f32[144]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[144]" = torch.ops.aten.mul.Tensor(primals_49, 0.9)
        add_tensor_20: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(primals_53, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_16: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        mul_tensor_35: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_16, 0.1);  squeeze_dims_16 = None
        mul_tensor_36: "f32[144]" = torch.ops.aten.mul.Tensor(primals_54, 0.9)
        add_tensor_22: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_17: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 1.0000024912370735);  squeeze_dims_17 = None
        mul_tensor_38: "f32[144]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[144]" = torch.ops.aten.mul.Tensor(primals_55, 0.9)
        add_tensor_23: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_18: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_19: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 0.1)
        mul_tensor_41: "f32[24]" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_tensor_25: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_20: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 1.0000024912370735);  squeeze_dims_20 = None
        mul_tensor_43: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[24]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_tensor_26: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_21: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        mul_tensor_45: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_21, 0.1);  squeeze_dims_21 = None
        mul_tensor_46: "f32[144]" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_tensor_28: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_22: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_22, 1.0000024912370735);  squeeze_dims_22 = None
        mul_tensor_48: "f32[144]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[144]" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_tensor_29: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_23: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        mul_tensor_50: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 0.1);  squeeze_dims_23 = None
        mul_tensor_51: "f32[144]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_tensor_31: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_24: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 1.00000996502277);  squeeze_dims_24 = None
        mul_tensor_53: "f32[144]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[144]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_tensor_32: "f32[144]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(primals_85, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_25: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_26: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_25, 0.1)
        mul_tensor_56: "f32[40]" = torch.ops.aten.mul.Tensor(primals_86, 0.9)
        add_tensor_34: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_27: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 1.00000996502277);  squeeze_dims_27 = None
        mul_tensor_58: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[40]" = torch.ops.aten.mul.Tensor(primals_87, 0.9)
        add_tensor_35: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(primals_91, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_28: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        mul_tensor_60: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_28, 0.1);  squeeze_dims_28 = None
        mul_tensor_61: "f32[240]" = torch.ops.aten.mul.Tensor(primals_92, 0.9)
        add_tensor_37: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_29: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 1.00000996502277);  squeeze_dims_29 = None
        mul_tensor_63: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[240]" = torch.ops.aten.mul.Tensor(primals_93, 0.9)
        add_tensor_38: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(primals_97, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_30: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        mul_tensor_65: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 0.1);  squeeze_dims_30 = None
        mul_tensor_66: "f32[240]" = torch.ops.aten.mul.Tensor(primals_98, 0.9)
        add_tensor_40: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_31: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_31, 1.00000996502277);  squeeze_dims_31 = None
        mul_tensor_68: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[240]" = torch.ops.aten.mul.Tensor(primals_99, 0.9)
        add_tensor_41: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(primals_107, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_32: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_33: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 0.1)
        mul_tensor_71: "f32[40]" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
        add_tensor_43: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_34: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_34, 1.00000996502277);  squeeze_dims_34 = None
        mul_tensor_73: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[40]" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        add_tensor_44: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(primals_113, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_35: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        mul_tensor_75: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_35, 0.1);  squeeze_dims_35 = None
        mul_tensor_76: "f32[240]" = torch.ops.aten.mul.Tensor(primals_114, 0.9)
        add_tensor_46: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_36: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 1.00000996502277);  squeeze_dims_36 = None
        mul_tensor_78: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[240]" = torch.ops.aten.mul.Tensor(primals_115, 0.9)
        add_tensor_47: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(primals_119, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_37: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        mul_tensor_80: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_37, 0.1);  squeeze_dims_37 = None
        mul_tensor_81: "f32[240]" = torch.ops.aten.mul.Tensor(primals_120, 0.9)
        add_tensor_49: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_38: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.0000398612827361);  squeeze_dims_38 = None
        mul_tensor_83: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[240]" = torch.ops.aten.mul.Tensor(primals_121, 0.9)
        add_tensor_50: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(primals_129, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_39: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_dims_40: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_85: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.1)
        mul_tensor_86: "f32[80]" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_tensor_52: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_41: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_87: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 1.0000398612827361);  squeeze_dims_41 = None
        mul_tensor_88: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[80]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_tensor_53: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(primals_135, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_42: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        mul_tensor_90: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 0.1);  squeeze_dims_42 = None
        mul_tensor_91: "f32[480]" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_tensor_55: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_43: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_92: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_43, 1.0000398612827361);  squeeze_dims_43 = None
        mul_tensor_93: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[480]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_tensor_56: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_44: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        mul_tensor_95: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 0.1);  squeeze_dims_44 = None
        mul_tensor_96: "f32[480]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_tensor_58: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_45: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_97: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 1.0000398612827361);  squeeze_dims_45 = None
        mul_tensor_98: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[480]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_tensor_59: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(primals_151, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_46: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_47: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_46, 0.1)
        mul_tensor_101: "f32[80]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_tensor_61: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_48: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_102: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 1.0000398612827361);  squeeze_dims_48 = None
        mul_tensor_103: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[80]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
        add_tensor_62: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(primals_157, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_49: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        mul_tensor_105: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_49, 0.1);  squeeze_dims_49 = None
        mul_tensor_106: "f32[480]" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
        add_tensor_64: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_50: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_107: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 1.0000398612827361);  squeeze_dims_50 = None
        mul_tensor_108: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[480]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
        add_tensor_65: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(primals_163, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_51: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        mul_tensor_110: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 0.1);  squeeze_dims_51 = None
        mul_tensor_111: "f32[480]" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
        add_tensor_67: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_52: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_tensor_112: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_52, 1.0000398612827361);  squeeze_dims_52 = None
        mul_tensor_113: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[480]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
        add_tensor_68: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_53: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_54: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 0.1)
        mul_tensor_116: "f32[80]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
        add_tensor_70: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_55: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_117: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_55, 1.0000398612827361);  squeeze_dims_55 = None
        mul_tensor_118: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[80]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
        add_tensor_71: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_56: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        mul_tensor_120: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 0.1);  squeeze_dims_56 = None
        mul_tensor_121: "f32[480]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
        add_tensor_73: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_57: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_122: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 1.0000398612827361);  squeeze_dims_57 = None
        mul_tensor_123: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[480]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
        add_tensor_74: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(primals_185, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_58: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        mul_tensor_125: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_58, 0.1);  squeeze_dims_58 = None
        mul_tensor_126: "f32[480]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
        add_tensor_76: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_59: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_127: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 1.0000398612827361);  squeeze_dims_59 = None
        mul_tensor_128: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[480]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        add_tensor_77: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(primals_195, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_60: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_dims_61: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 0.1)
        mul_tensor_131: "f32[112]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_tensor_79: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_62: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_tensor_132: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 1.0000398612827361);  squeeze_dims_62 = None
        mul_tensor_133: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[112]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_tensor_80: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_63: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        mul_tensor_135: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 0.1);  squeeze_dims_63 = None
        mul_tensor_136: "f32[672]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_tensor_82: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_64: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_tensor_137: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_64, 1.0000398612827361);  squeeze_dims_64 = None
        mul_tensor_138: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.1);  mul_tensor_137 = None
        mul_tensor_139: "f32[672]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_tensor_83: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_65: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        mul_tensor_140: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_65, 0.1);  squeeze_dims_65 = None
        mul_tensor_141: "f32[672]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_tensor_85: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_66: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_tensor_142: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 1.0000398612827361);  squeeze_dims_66 = None
        mul_tensor_143: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.1);  mul_tensor_142 = None
        mul_tensor_144: "f32[672]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_tensor_86: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(primals_217, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_67: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_dims_68: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_tensor_145: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_67, 0.1)
        mul_tensor_146: "f32[112]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
        add_tensor_88: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_69: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_147: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 1.0000398612827361);  squeeze_dims_69 = None
        mul_tensor_148: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.1);  mul_tensor_147 = None
        mul_tensor_149: "f32[112]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
        add_tensor_89: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(primals_223, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_70: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        mul_tensor_150: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_70, 0.1);  squeeze_dims_70 = None
        mul_tensor_151: "f32[672]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
        add_tensor_91: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_71: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_152: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 1.0000398612827361);  squeeze_dims_71 = None
        mul_tensor_153: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.1);  mul_tensor_152 = None
        mul_tensor_154: "f32[672]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
        add_tensor_92: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(primals_229, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_72: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        mul_tensor_155: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_72, 0.1);  squeeze_dims_72 = None
        mul_tensor_156: "f32[672]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
        add_tensor_94: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_155, mul_tensor_156);  mul_tensor_155 = mul_tensor_156 = None
        squeeze_dims_73: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_157: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_73, 1.0000398612827361);  squeeze_dims_73 = None
        mul_tensor_158: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_157, 0.1);  mul_tensor_157 = None
        mul_tensor_159: "f32[672]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
        add_tensor_95: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_96: "i64[]" = torch.ops.aten.add.Tensor(primals_239, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_74: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_75: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_tensor_160: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 0.1)
        mul_tensor_161: "f32[112]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
        add_tensor_97: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        squeeze_dims_76: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_162: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_76, 1.0000398612827361);  squeeze_dims_76 = None
        mul_tensor_163: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_162, 0.1);  mul_tensor_162 = None
        mul_tensor_164: "f32[112]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
        add_tensor_98: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_163, mul_tensor_164);  mul_tensor_163 = mul_tensor_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_99: "i64[]" = torch.ops.aten.add.Tensor(primals_245, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_77: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        mul_tensor_165: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 0.1);  squeeze_dims_77 = None
        mul_tensor_166: "f32[672]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
        add_tensor_100: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_165, mul_tensor_166);  mul_tensor_165 = mul_tensor_166 = None
        squeeze_dims_78: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_167: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_78, 1.0000398612827361);  squeeze_dims_78 = None
        mul_tensor_168: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_167, 0.1);  mul_tensor_167 = None
        mul_tensor_169: "f32[672]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
        add_tensor_101: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_102: "i64[]" = torch.ops.aten.add.Tensor(primals_251, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_79: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        mul_tensor_170: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_79, 0.1);  squeeze_dims_79 = None
        mul_tensor_171: "f32[672]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
        add_tensor_103: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        squeeze_dims_80: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_172: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 1.0001594642002871);  squeeze_dims_80 = None
        mul_tensor_173: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_172, 0.1);  mul_tensor_172 = None
        mul_tensor_174: "f32[672]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
        add_tensor_104: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_173, mul_tensor_174);  mul_tensor_173 = mul_tensor_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_105: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_81: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_dims_82: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_tensor_175: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_81, 0.1)
        mul_tensor_176: "f32[192]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_tensor_106: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_175, mul_tensor_176);  mul_tensor_175 = mul_tensor_176 = None
        squeeze_dims_83: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_tensor_177: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_83, 1.0001594642002871);  squeeze_dims_83 = None
        mul_tensor_178: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_177, 0.1);  mul_tensor_177 = None
        mul_tensor_179: "f32[192]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_tensor_107: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_108: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_84: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        mul_tensor_180: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_84, 0.1);  squeeze_dims_84 = None
        mul_tensor_181: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_tensor_109: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        squeeze_dims_85: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_tensor_182: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_85, 1.0001594642002871);  squeeze_dims_85 = None
        mul_tensor_183: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_182, 0.1);  mul_tensor_182 = None
        mul_tensor_184: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_tensor_110: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_183, mul_tensor_184);  mul_tensor_183 = mul_tensor_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_111: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_86: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        mul_tensor_185: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_86, 0.1);  squeeze_dims_86 = None
        mul_tensor_186: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_tensor_112: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_185, mul_tensor_186);  mul_tensor_185 = mul_tensor_186 = None
        squeeze_dims_87: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_tensor_187: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_87, 1.0001594642002871);  squeeze_dims_87 = None
        mul_tensor_188: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_187, 0.1);  mul_tensor_187 = None
        mul_tensor_189: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
        add_tensor_113: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_114: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_88: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_89: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_tensor_190: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_88, 0.1)
        mul_tensor_191: "f32[192]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
        add_tensor_115: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        squeeze_dims_90: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_192: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_90, 1.0001594642002871);  squeeze_dims_90 = None
        mul_tensor_193: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_192, 0.1);  mul_tensor_192 = None
        mul_tensor_194: "f32[192]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
        add_tensor_116: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_193, mul_tensor_194);  mul_tensor_193 = mul_tensor_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_117: "i64[]" = torch.ops.aten.add.Tensor(primals_289, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_91: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        mul_tensor_195: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_91, 0.1);  squeeze_dims_91 = None
        mul_tensor_196: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
        add_tensor_118: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_195, mul_tensor_196);  mul_tensor_195 = mul_tensor_196 = None
        squeeze_dims_92: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_197: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_92, 1.0001594642002871);  squeeze_dims_92 = None
        mul_tensor_198: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_197, 0.1);  mul_tensor_197 = None
        mul_tensor_199: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
        add_tensor_119: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_120: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_93: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        mul_tensor_200: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_93, 0.1);  squeeze_dims_93 = None
        mul_tensor_201: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
        add_tensor_121: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        squeeze_dims_94: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_tensor_202: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_94, 1.0001594642002871);  squeeze_dims_94 = None
        mul_tensor_203: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_202, 0.1);  mul_tensor_202 = None
        mul_tensor_204: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
        add_tensor_122: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_203, mul_tensor_204);  mul_tensor_203 = mul_tensor_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_123: "i64[]" = torch.ops.aten.add.Tensor(primals_305, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_95: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_dims_96: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_205: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_95, 0.1)
        mul_tensor_206: "f32[192]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
        add_tensor_124: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_205, mul_tensor_206);  mul_tensor_205 = mul_tensor_206 = None
        squeeze_dims_97: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_tensor_207: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_97, 1.0001594642002871);  squeeze_dims_97 = None
        mul_tensor_208: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_207, 0.1);  mul_tensor_207 = None
        mul_tensor_209: "f32[192]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_tensor_125: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_126: "i64[]" = torch.ops.aten.add.Tensor(primals_311, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_98: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        mul_tensor_210: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_98, 0.1);  squeeze_dims_98 = None
        mul_tensor_211: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
        add_tensor_127: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        squeeze_dims_99: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_tensor_212: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_99, 1.0001594642002871);  squeeze_dims_99 = None
        mul_tensor_213: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_212, 0.1);  mul_tensor_212 = None
        mul_tensor_214: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
        add_tensor_128: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_129: "i64[]" = torch.ops.aten.add.Tensor(primals_317, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_100: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        mul_tensor_215: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_100, 0.1);  squeeze_dims_100 = None
        mul_tensor_216: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
        add_tensor_130: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_215, mul_tensor_216);  mul_tensor_215 = mul_tensor_216 = None
        squeeze_dims_101: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_tensor_217: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_101, 1.0001594642002871);  squeeze_dims_101 = None
        mul_tensor_218: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_217, 0.1);  mul_tensor_217 = None
        mul_tensor_219: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
        add_tensor_131: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_132: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_102: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_dims_103: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_tensor_220: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_102, 0.1)
        mul_tensor_221: "f32[192]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_tensor_133: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        squeeze_dims_104: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_tensor_222: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_104, 1.0001594642002871);  squeeze_dims_104 = None
        mul_tensor_223: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_222, 0.1);  mul_tensor_222 = None
        mul_tensor_224: "f32[192]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_tensor_134: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_223, mul_tensor_224);  mul_tensor_223 = mul_tensor_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_135: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_105: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        mul_tensor_225: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_105, 0.1);  squeeze_dims_105 = None
        mul_tensor_226: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_tensor_136: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_225, mul_tensor_226);  mul_tensor_225 = mul_tensor_226 = None
        squeeze_dims_106: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_tensor_227: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_106, 1.0001594642002871);  squeeze_dims_106 = None
        mul_tensor_228: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_227, 0.1);  mul_tensor_227 = None
        mul_tensor_229: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_tensor_137: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_138: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_107: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        mul_tensor_230: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_107, 0.1);  squeeze_dims_107 = None
        mul_tensor_231: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
        add_tensor_139: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        squeeze_dims_108: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_tensor_232: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_108, 1.0001594642002871);  squeeze_dims_108 = None
        mul_tensor_233: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_232, 0.1);  mul_tensor_232 = None
        mul_tensor_234: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_tensor_140: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_233, mul_tensor_234);  mul_tensor_233 = mul_tensor_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_141: "i64[]" = torch.ops.aten.add.Tensor(primals_349, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_109: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_dims_110: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_tensor_235: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_109, 0.1)
        mul_tensor_236: "f32[320]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
        add_tensor_142: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        squeeze_dims_111: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_tensor_237: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_111, 1.0001594642002871);  squeeze_dims_111 = None
        mul_tensor_238: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_237, 0.1);  mul_tensor_237 = None
        mul_tensor_239: "f32[320]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
        add_tensor_143: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_144: "i64[]" = torch.ops.aten.add.Tensor(primals_355, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96: "f32[1, 1280, 1, 1]" = var_mean_correction[0]
        getitem_97: "f32[1, 1280, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_145: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
        rsqrt_default: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_145);  add_tensor_145 = None
        sub_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_97);  convolution_80 = None
        mul_tensor_240: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims_112: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        mul_tensor_241: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_112, 0.1);  squeeze_dims_112 = None
        mul_tensor_242: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
        add_tensor_146: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_241, mul_tensor_242);  mul_tensor_241 = mul_tensor_242 = None
        squeeze_dims_113: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_tensor_243: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_113, 1.0001594642002871);  squeeze_dims_113 = None
        mul_tensor_244: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_243, 0.1);  mul_tensor_243 = None
        mul_tensor_245: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
        add_tensor_147: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_244, mul_tensor_245);  mul_tensor_244 = mul_tensor_245 = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_358, -1);  primals_358 = None
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_246: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_240, unsqueeze_default_1);  mul_tensor_240 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_359, -1);  primals_359 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_148: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_246, unsqueeze_default_3);  mul_tensor_246 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.neg.default(add_tensor_148)
        exp_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_149: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor_148, add_tensor_149);  add_tensor_148 = add_tensor_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 1280, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None
        as_strided_default: "f32[128, 1280, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 1280, 1, 1], [1280, 1, 1280, 1280]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 1280]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:344 in forward_head, code: return x if pre_logits else self.classifier(x)
        permute_default: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_360, [1, 0]);  primals_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_4: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_dims_109, 0);  squeeze_dims_109 = None
        unsqueeze_default_5: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_102, 0);  squeeze_dims_102 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_95, 0);  squeeze_dims_95 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_88, 0);  squeeze_dims_88 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_81, 0);  squeeze_dims_81 = None
        unsqueeze_default_17: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_dims_74, 0);  squeeze_dims_74 = None
        unsqueeze_default_20: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_dims_67, 0);  squeeze_dims_67 = None
        unsqueeze_default_23: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_26: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_53, 0);  squeeze_dims_53 = None
        unsqueeze_default_29: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_46, 0);  squeeze_dims_46 = None
        unsqueeze_default_32: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_39, 0);  squeeze_dims_39 = None
        unsqueeze_default_35: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 2);  unsqueeze_default_34 = None
        unsqueeze_default_36: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 3);  unsqueeze_default_35 = None
        unsqueeze_default_37: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_32, 0);  squeeze_dims_32 = None
        unsqueeze_default_38: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 2);  unsqueeze_default_37 = None
        unsqueeze_default_39: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 3);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_25, 0);  squeeze_dims_25 = None
        unsqueeze_default_41: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 2);  unsqueeze_default_40 = None
        unsqueeze_default_42: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 3);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_18, 0);  squeeze_dims_18 = None
        unsqueeze_default_44: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 2);  unsqueeze_default_43 = None
        unsqueeze_default_45: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 3);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_11, 0);  squeeze_dims_11 = None
        unsqueeze_default_47: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 2);  unsqueeze_default_46 = None
        unsqueeze_default_48: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 3);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims_4, 0);  squeeze_dims_4 = None
        unsqueeze_default_50: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 2);  unsqueeze_default_49 = None
        unsqueeze_default_51: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 3);  unsqueeze_default_50 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_3, add_tensor);  primals_3 = add_tensor = None
        copy__default_1: "f32[32]" = torch.ops.aten.copy_.default(primals_4, add_tensor_1);  primals_4 = add_tensor_1 = None
        copy__default_2: "f32[32]" = torch.ops.aten.copy_.default(primals_5, add_tensor_2);  primals_5 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_tensor_3);  primals_9 = add_tensor_3 = None
        copy__default_4: "f32[32]" = torch.ops.aten.copy_.default(primals_10, add_tensor_4);  primals_10 = add_tensor_4 = None
        copy__default_5: "f32[32]" = torch.ops.aten.copy_.default(primals_11, add_tensor_5);  primals_11 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_19, add_tensor_6);  primals_19 = add_tensor_6 = None
        copy__default_7: "f32[16]" = torch.ops.aten.copy_.default(primals_20, add_tensor_7);  primals_20 = add_tensor_7 = None
        copy__default_8: "f32[16]" = torch.ops.aten.copy_.default(primals_21, add_tensor_8);  primals_21 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(primals_25, add_tensor_9);  primals_25 = add_tensor_9 = None
        copy__default_10: "f32[96]" = torch.ops.aten.copy_.default(primals_26, add_tensor_10);  primals_26 = add_tensor_10 = None
        copy__default_11: "f32[96]" = torch.ops.aten.copy_.default(primals_27, add_tensor_11);  primals_27 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(primals_31, add_tensor_12);  primals_31 = add_tensor_12 = None
        copy__default_13: "f32[96]" = torch.ops.aten.copy_.default(primals_32, add_tensor_13);  primals_32 = add_tensor_13 = None
        copy__default_14: "f32[96]" = torch.ops.aten.copy_.default(primals_33, add_tensor_14);  primals_33 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(primals_41, add_tensor_15);  primals_41 = add_tensor_15 = None
        copy__default_16: "f32[24]" = torch.ops.aten.copy_.default(primals_42, add_tensor_16);  primals_42 = add_tensor_16 = None
        copy__default_17: "f32[24]" = torch.ops.aten.copy_.default(primals_43, add_tensor_17);  primals_43 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(primals_47, add_tensor_18);  primals_47 = add_tensor_18 = None
        copy__default_19: "f32[144]" = torch.ops.aten.copy_.default(primals_48, add_tensor_19);  primals_48 = add_tensor_19 = None
        copy__default_20: "f32[144]" = torch.ops.aten.copy_.default(primals_49, add_tensor_20);  primals_49 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(primals_53, add_tensor_21);  primals_53 = add_tensor_21 = None
        copy__default_22: "f32[144]" = torch.ops.aten.copy_.default(primals_54, add_tensor_22);  primals_54 = add_tensor_22 = None
        copy__default_23: "f32[144]" = torch.ops.aten.copy_.default(primals_55, add_tensor_23);  primals_55 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_tensor_24);  primals_63 = add_tensor_24 = None
        copy__default_25: "f32[24]" = torch.ops.aten.copy_.default(primals_64, add_tensor_25);  primals_64 = add_tensor_25 = None
        copy__default_26: "f32[24]" = torch.ops.aten.copy_.default(primals_65, add_tensor_26);  primals_65 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_tensor_27);  primals_69 = add_tensor_27 = None
        copy__default_28: "f32[144]" = torch.ops.aten.copy_.default(primals_70, add_tensor_28);  primals_70 = add_tensor_28 = None
        copy__default_29: "f32[144]" = torch.ops.aten.copy_.default(primals_71, add_tensor_29);  primals_71 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_tensor_30);  primals_75 = add_tensor_30 = None
        copy__default_31: "f32[144]" = torch.ops.aten.copy_.default(primals_76, add_tensor_31);  primals_76 = add_tensor_31 = None
        copy__default_32: "f32[144]" = torch.ops.aten.copy_.default(primals_77, add_tensor_32);  primals_77 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(primals_85, add_tensor_33);  primals_85 = add_tensor_33 = None
        copy__default_34: "f32[40]" = torch.ops.aten.copy_.default(primals_86, add_tensor_34);  primals_86 = add_tensor_34 = None
        copy__default_35: "f32[40]" = torch.ops.aten.copy_.default(primals_87, add_tensor_35);  primals_87 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(primals_91, add_tensor_36);  primals_91 = add_tensor_36 = None
        copy__default_37: "f32[240]" = torch.ops.aten.copy_.default(primals_92, add_tensor_37);  primals_92 = add_tensor_37 = None
        copy__default_38: "f32[240]" = torch.ops.aten.copy_.default(primals_93, add_tensor_38);  primals_93 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(primals_97, add_tensor_39);  primals_97 = add_tensor_39 = None
        copy__default_40: "f32[240]" = torch.ops.aten.copy_.default(primals_98, add_tensor_40);  primals_98 = add_tensor_40 = None
        copy__default_41: "f32[240]" = torch.ops.aten.copy_.default(primals_99, add_tensor_41);  primals_99 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(primals_107, add_tensor_42);  primals_107 = add_tensor_42 = None
        copy__default_43: "f32[40]" = torch.ops.aten.copy_.default(primals_108, add_tensor_43);  primals_108 = add_tensor_43 = None
        copy__default_44: "f32[40]" = torch.ops.aten.copy_.default(primals_109, add_tensor_44);  primals_109 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(primals_113, add_tensor_45);  primals_113 = add_tensor_45 = None
        copy__default_46: "f32[240]" = torch.ops.aten.copy_.default(primals_114, add_tensor_46);  primals_114 = add_tensor_46 = None
        copy__default_47: "f32[240]" = torch.ops.aten.copy_.default(primals_115, add_tensor_47);  primals_115 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(primals_119, add_tensor_48);  primals_119 = add_tensor_48 = None
        copy__default_49: "f32[240]" = torch.ops.aten.copy_.default(primals_120, add_tensor_49);  primals_120 = add_tensor_49 = None
        copy__default_50: "f32[240]" = torch.ops.aten.copy_.default(primals_121, add_tensor_50);  primals_121 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(primals_129, add_tensor_51);  primals_129 = add_tensor_51 = None
        copy__default_52: "f32[80]" = torch.ops.aten.copy_.default(primals_130, add_tensor_52);  primals_130 = add_tensor_52 = None
        copy__default_53: "f32[80]" = torch.ops.aten.copy_.default(primals_131, add_tensor_53);  primals_131 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(primals_135, add_tensor_54);  primals_135 = add_tensor_54 = None
        copy__default_55: "f32[480]" = torch.ops.aten.copy_.default(primals_136, add_tensor_55);  primals_136 = add_tensor_55 = None
        copy__default_56: "f32[480]" = torch.ops.aten.copy_.default(primals_137, add_tensor_56);  primals_137 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(primals_141, add_tensor_57);  primals_141 = add_tensor_57 = None
        copy__default_58: "f32[480]" = torch.ops.aten.copy_.default(primals_142, add_tensor_58);  primals_142 = add_tensor_58 = None
        copy__default_59: "f32[480]" = torch.ops.aten.copy_.default(primals_143, add_tensor_59);  primals_143 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(primals_151, add_tensor_60);  primals_151 = add_tensor_60 = None
        copy__default_61: "f32[80]" = torch.ops.aten.copy_.default(primals_152, add_tensor_61);  primals_152 = add_tensor_61 = None
        copy__default_62: "f32[80]" = torch.ops.aten.copy_.default(primals_153, add_tensor_62);  primals_153 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(primals_157, add_tensor_63);  primals_157 = add_tensor_63 = None
        copy__default_64: "f32[480]" = torch.ops.aten.copy_.default(primals_158, add_tensor_64);  primals_158 = add_tensor_64 = None
        copy__default_65: "f32[480]" = torch.ops.aten.copy_.default(primals_159, add_tensor_65);  primals_159 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(primals_163, add_tensor_66);  primals_163 = add_tensor_66 = None
        copy__default_67: "f32[480]" = torch.ops.aten.copy_.default(primals_164, add_tensor_67);  primals_164 = add_tensor_67 = None
        copy__default_68: "f32[480]" = torch.ops.aten.copy_.default(primals_165, add_tensor_68);  primals_165 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(primals_173, add_tensor_69);  primals_173 = add_tensor_69 = None
        copy__default_70: "f32[80]" = torch.ops.aten.copy_.default(primals_174, add_tensor_70);  primals_174 = add_tensor_70 = None
        copy__default_71: "f32[80]" = torch.ops.aten.copy_.default(primals_175, add_tensor_71);  primals_175 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_tensor_72);  primals_179 = add_tensor_72 = None
        copy__default_73: "f32[480]" = torch.ops.aten.copy_.default(primals_180, add_tensor_73);  primals_180 = add_tensor_73 = None
        copy__default_74: "f32[480]" = torch.ops.aten.copy_.default(primals_181, add_tensor_74);  primals_181 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(primals_185, add_tensor_75);  primals_185 = add_tensor_75 = None
        copy__default_76: "f32[480]" = torch.ops.aten.copy_.default(primals_186, add_tensor_76);  primals_186 = add_tensor_76 = None
        copy__default_77: "f32[480]" = torch.ops.aten.copy_.default(primals_187, add_tensor_77);  primals_187 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(primals_195, add_tensor_78);  primals_195 = add_tensor_78 = None
        copy__default_79: "f32[112]" = torch.ops.aten.copy_.default(primals_196, add_tensor_79);  primals_196 = add_tensor_79 = None
        copy__default_80: "f32[112]" = torch.ops.aten.copy_.default(primals_197, add_tensor_80);  primals_197 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(primals_201, add_tensor_81);  primals_201 = add_tensor_81 = None
        copy__default_82: "f32[672]" = torch.ops.aten.copy_.default(primals_202, add_tensor_82);  primals_202 = add_tensor_82 = None
        copy__default_83: "f32[672]" = torch.ops.aten.copy_.default(primals_203, add_tensor_83);  primals_203 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_tensor_84);  primals_207 = add_tensor_84 = None
        copy__default_85: "f32[672]" = torch.ops.aten.copy_.default(primals_208, add_tensor_85);  primals_208 = add_tensor_85 = None
        copy__default_86: "f32[672]" = torch.ops.aten.copy_.default(primals_209, add_tensor_86);  primals_209 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(primals_217, add_tensor_87);  primals_217 = add_tensor_87 = None
        copy__default_88: "f32[112]" = torch.ops.aten.copy_.default(primals_218, add_tensor_88);  primals_218 = add_tensor_88 = None
        copy__default_89: "f32[112]" = torch.ops.aten.copy_.default(primals_219, add_tensor_89);  primals_219 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(primals_223, add_tensor_90);  primals_223 = add_tensor_90 = None
        copy__default_91: "f32[672]" = torch.ops.aten.copy_.default(primals_224, add_tensor_91);  primals_224 = add_tensor_91 = None
        copy__default_92: "f32[672]" = torch.ops.aten.copy_.default(primals_225, add_tensor_92);  primals_225 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(primals_229, add_tensor_93);  primals_229 = add_tensor_93 = None
        copy__default_94: "f32[672]" = torch.ops.aten.copy_.default(primals_230, add_tensor_94);  primals_230 = add_tensor_94 = None
        copy__default_95: "f32[672]" = torch.ops.aten.copy_.default(primals_231, add_tensor_95);  primals_231 = add_tensor_95 = None
        copy__default_96: "i64[]" = torch.ops.aten.copy_.default(primals_239, add_tensor_96);  primals_239 = add_tensor_96 = None
        copy__default_97: "f32[112]" = torch.ops.aten.copy_.default(primals_240, add_tensor_97);  primals_240 = add_tensor_97 = None
        copy__default_98: "f32[112]" = torch.ops.aten.copy_.default(primals_241, add_tensor_98);  primals_241 = add_tensor_98 = None
        copy__default_99: "i64[]" = torch.ops.aten.copy_.default(primals_245, add_tensor_99);  primals_245 = add_tensor_99 = None
        copy__default_100: "f32[672]" = torch.ops.aten.copy_.default(primals_246, add_tensor_100);  primals_246 = add_tensor_100 = None
        copy__default_101: "f32[672]" = torch.ops.aten.copy_.default(primals_247, add_tensor_101);  primals_247 = add_tensor_101 = None
        copy__default_102: "i64[]" = torch.ops.aten.copy_.default(primals_251, add_tensor_102);  primals_251 = add_tensor_102 = None
        copy__default_103: "f32[672]" = torch.ops.aten.copy_.default(primals_252, add_tensor_103);  primals_252 = add_tensor_103 = None
        copy__default_104: "f32[672]" = torch.ops.aten.copy_.default(primals_253, add_tensor_104);  primals_253 = add_tensor_104 = None
        copy__default_105: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_tensor_105);  primals_261 = add_tensor_105 = None
        copy__default_106: "f32[192]" = torch.ops.aten.copy_.default(primals_262, add_tensor_106);  primals_262 = add_tensor_106 = None
        copy__default_107: "f32[192]" = torch.ops.aten.copy_.default(primals_263, add_tensor_107);  primals_263 = add_tensor_107 = None
        copy__default_108: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_tensor_108);  primals_267 = add_tensor_108 = None
        copy__default_109: "f32[1152]" = torch.ops.aten.copy_.default(primals_268, add_tensor_109);  primals_268 = add_tensor_109 = None
        copy__default_110: "f32[1152]" = torch.ops.aten.copy_.default(primals_269, add_tensor_110);  primals_269 = add_tensor_110 = None
        copy__default_111: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_tensor_111);  primals_273 = add_tensor_111 = None
        copy__default_112: "f32[1152]" = torch.ops.aten.copy_.default(primals_274, add_tensor_112);  primals_274 = add_tensor_112 = None
        copy__default_113: "f32[1152]" = torch.ops.aten.copy_.default(primals_275, add_tensor_113);  primals_275 = add_tensor_113 = None
        copy__default_114: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_tensor_114);  primals_283 = add_tensor_114 = None
        copy__default_115: "f32[192]" = torch.ops.aten.copy_.default(primals_284, add_tensor_115);  primals_284 = add_tensor_115 = None
        copy__default_116: "f32[192]" = torch.ops.aten.copy_.default(primals_285, add_tensor_116);  primals_285 = add_tensor_116 = None
        copy__default_117: "i64[]" = torch.ops.aten.copy_.default(primals_289, add_tensor_117);  primals_289 = add_tensor_117 = None
        copy__default_118: "f32[1152]" = torch.ops.aten.copy_.default(primals_290, add_tensor_118);  primals_290 = add_tensor_118 = None
        copy__default_119: "f32[1152]" = torch.ops.aten.copy_.default(primals_291, add_tensor_119);  primals_291 = add_tensor_119 = None
        copy__default_120: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_tensor_120);  primals_295 = add_tensor_120 = None
        copy__default_121: "f32[1152]" = torch.ops.aten.copy_.default(primals_296, add_tensor_121);  primals_296 = add_tensor_121 = None
        copy__default_122: "f32[1152]" = torch.ops.aten.copy_.default(primals_297, add_tensor_122);  primals_297 = add_tensor_122 = None
        copy__default_123: "i64[]" = torch.ops.aten.copy_.default(primals_305, add_tensor_123);  primals_305 = add_tensor_123 = None
        copy__default_124: "f32[192]" = torch.ops.aten.copy_.default(primals_306, add_tensor_124);  primals_306 = add_tensor_124 = None
        copy__default_125: "f32[192]" = torch.ops.aten.copy_.default(primals_307, add_tensor_125);  primals_307 = add_tensor_125 = None
        copy__default_126: "i64[]" = torch.ops.aten.copy_.default(primals_311, add_tensor_126);  primals_311 = add_tensor_126 = None
        copy__default_127: "f32[1152]" = torch.ops.aten.copy_.default(primals_312, add_tensor_127);  primals_312 = add_tensor_127 = None
        copy__default_128: "f32[1152]" = torch.ops.aten.copy_.default(primals_313, add_tensor_128);  primals_313 = add_tensor_128 = None
        copy__default_129: "i64[]" = torch.ops.aten.copy_.default(primals_317, add_tensor_129);  primals_317 = add_tensor_129 = None
        copy__default_130: "f32[1152]" = torch.ops.aten.copy_.default(primals_318, add_tensor_130);  primals_318 = add_tensor_130 = None
        copy__default_131: "f32[1152]" = torch.ops.aten.copy_.default(primals_319, add_tensor_131);  primals_319 = add_tensor_131 = None
        copy__default_132: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_tensor_132);  primals_327 = add_tensor_132 = None
        copy__default_133: "f32[192]" = torch.ops.aten.copy_.default(primals_328, add_tensor_133);  primals_328 = add_tensor_133 = None
        copy__default_134: "f32[192]" = torch.ops.aten.copy_.default(primals_329, add_tensor_134);  primals_329 = add_tensor_134 = None
        copy__default_135: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_tensor_135);  primals_333 = add_tensor_135 = None
        copy__default_136: "f32[1152]" = torch.ops.aten.copy_.default(primals_334, add_tensor_136);  primals_334 = add_tensor_136 = None
        copy__default_137: "f32[1152]" = torch.ops.aten.copy_.default(primals_335, add_tensor_137);  primals_335 = add_tensor_137 = None
        copy__default_138: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_tensor_138);  primals_339 = add_tensor_138 = None
        copy__default_139: "f32[1152]" = torch.ops.aten.copy_.default(primals_340, add_tensor_139);  primals_340 = add_tensor_139 = None
        copy__default_140: "f32[1152]" = torch.ops.aten.copy_.default(primals_341, add_tensor_140);  primals_341 = add_tensor_140 = None
        copy__default_141: "i64[]" = torch.ops.aten.copy_.default(primals_349, add_tensor_141);  primals_349 = add_tensor_141 = None
        copy__default_142: "f32[320]" = torch.ops.aten.copy_.default(primals_350, add_tensor_142);  primals_350 = add_tensor_142 = None
        copy__default_143: "f32[320]" = torch.ops.aten.copy_.default(primals_351, add_tensor_143);  primals_351 = add_tensor_143 = None
        copy__default_144: "i64[]" = torch.ops.aten.copy_.default(primals_355, add_tensor_144);  primals_355 = add_tensor_144 = None
        copy__default_145: "f32[1280]" = torch.ops.aten.copy_.default(primals_356, add_tensor_146);  primals_356 = add_tensor_146 = None
        copy__default_146: "f32[1280]" = torch.ops.aten.copy_.default(primals_357, add_tensor_147);  primals_357 = add_tensor_147 = None
        return (squeeze_dims_5, squeeze_dims_12, squeeze_dims_19, squeeze_dims_26, squeeze_dims_33, squeeze_dims_40, squeeze_dims_47, squeeze_dims_54, squeeze_dims_61, squeeze_dims_68, squeeze_dims_75, squeeze_dims_82, squeeze_dims_89, squeeze_dims_96, squeeze_dims_103, squeeze_dims_110, reshape_default, permute_default, unsqueeze_default_6, unsqueeze_default_9, unsqueeze_default_12, unsqueeze_default_15, unsqueeze_default_18, unsqueeze_default_21, unsqueeze_default_24, unsqueeze_default_27, unsqueeze_default_30, unsqueeze_default_33, unsqueeze_default_36, unsqueeze_default_39, unsqueeze_default_42, unsqueeze_default_45, unsqueeze_default_48, unsqueeze_default_51, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146)
