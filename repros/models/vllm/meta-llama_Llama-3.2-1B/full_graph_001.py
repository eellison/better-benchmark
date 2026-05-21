class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 512]", primals_2: "bf16[128256, 2048]", primals_3: "f32[32]", primals_4: "bf16[2048]", primals_5: "bf16[2048, 2048]", primals_6: "bf16[512, 2048]", primals_7: "bf16[512, 2048]", primals_8: "bf16[2048, 2048]", primals_9: "bf16[2048]", primals_10: "bf16[8192, 2048]", primals_11: "bf16[8192, 2048]", primals_12: "bf16[2048, 8192]", primals_13: "bf16[2048]", primals_14: "bf16[2048, 2048]", primals_15: "bf16[512, 2048]", primals_16: "bf16[512, 2048]", primals_17: "bf16[2048, 2048]", primals_18: "bf16[2048]", primals_19: "bf16[8192, 2048]", primals_20: "bf16[8192, 2048]", primals_21: "bf16[2048, 8192]", primals_22: "bf16[2048]", primals_23: "bf16[2048, 2048]", primals_24: "bf16[512, 2048]", primals_25: "bf16[512, 2048]", primals_26: "bf16[2048, 2048]", primals_27: "bf16[2048]", primals_28: "bf16[8192, 2048]", primals_29: "bf16[8192, 2048]", primals_30: "bf16[2048, 8192]", primals_31: "bf16[2048]", primals_32: "bf16[2048, 2048]", primals_33: "bf16[512, 2048]", primals_34: "bf16[512, 2048]", primals_35: "bf16[2048, 2048]", primals_36: "bf16[2048]", primals_37: "bf16[8192, 2048]", primals_38: "bf16[8192, 2048]", primals_39: "bf16[2048, 8192]", primals_40: "bf16[2048]", primals_41: "bf16[2048, 2048]", primals_42: "bf16[512, 2048]", primals_43: "bf16[512, 2048]", primals_44: "bf16[2048, 2048]", primals_45: "bf16[2048]", primals_46: "bf16[8192, 2048]", primals_47: "bf16[8192, 2048]", primals_48: "bf16[2048, 8192]", primals_49: "bf16[2048]", primals_50: "bf16[2048, 2048]", primals_51: "bf16[512, 2048]", primals_52: "bf16[512, 2048]", primals_53: "bf16[2048, 2048]", primals_54: "bf16[2048]", primals_55: "bf16[8192, 2048]", primals_56: "bf16[8192, 2048]", primals_57: "bf16[2048, 8192]", primals_58: "bf16[2048]", primals_59: "bf16[2048, 2048]", primals_60: "bf16[512, 2048]", primals_61: "bf16[512, 2048]", primals_62: "bf16[2048, 2048]", primals_63: "bf16[2048]", primals_64: "bf16[8192, 2048]", primals_65: "bf16[8192, 2048]", primals_66: "bf16[2048, 8192]", primals_67: "bf16[2048]", primals_68: "bf16[2048, 2048]", primals_69: "bf16[512, 2048]", primals_70: "bf16[512, 2048]", primals_71: "bf16[2048, 2048]", primals_72: "bf16[2048]", primals_73: "bf16[8192, 2048]", primals_74: "bf16[8192, 2048]", primals_75: "bf16[2048, 8192]", primals_76: "bf16[2048]", primals_77: "bf16[2048, 2048]", primals_78: "bf16[512, 2048]", primals_79: "bf16[512, 2048]", primals_80: "bf16[2048, 2048]", primals_81: "bf16[2048]", primals_82: "bf16[8192, 2048]", primals_83: "bf16[8192, 2048]", primals_84: "bf16[2048, 8192]", primals_85: "bf16[2048]", primals_86: "bf16[2048, 2048]", primals_87: "bf16[512, 2048]", primals_88: "bf16[512, 2048]", primals_89: "bf16[2048, 2048]", primals_90: "bf16[2048]", primals_91: "bf16[8192, 2048]", primals_92: "bf16[8192, 2048]", primals_93: "bf16[2048, 8192]", primals_94: "bf16[2048]", primals_95: "bf16[2048, 2048]", primals_96: "bf16[512, 2048]", primals_97: "bf16[512, 2048]", primals_98: "bf16[2048, 2048]", primals_99: "bf16[2048]", primals_100: "bf16[8192, 2048]", primals_101: "bf16[8192, 2048]", primals_102: "bf16[2048, 8192]", primals_103: "bf16[2048]", primals_104: "bf16[2048, 2048]", primals_105: "bf16[512, 2048]", primals_106: "bf16[512, 2048]", primals_107: "bf16[2048, 2048]", primals_108: "bf16[2048]", primals_109: "bf16[8192, 2048]", primals_110: "bf16[8192, 2048]", primals_111: "bf16[2048, 8192]", primals_112: "bf16[2048]", primals_113: "bf16[2048, 2048]", primals_114: "bf16[512, 2048]", primals_115: "bf16[512, 2048]", primals_116: "bf16[2048, 2048]", primals_117: "bf16[2048]", primals_118: "bf16[8192, 2048]", primals_119: "bf16[8192, 2048]", primals_120: "bf16[2048, 8192]", primals_121: "bf16[2048]", primals_122: "bf16[2048, 2048]", primals_123: "bf16[512, 2048]", primals_124: "bf16[512, 2048]", primals_125: "bf16[2048, 2048]", primals_126: "bf16[2048]", primals_127: "bf16[8192, 2048]", primals_128: "bf16[8192, 2048]", primals_129: "bf16[2048, 8192]", primals_130: "bf16[2048]", primals_131: "bf16[2048, 2048]", primals_132: "bf16[512, 2048]", primals_133: "bf16[512, 2048]", primals_134: "bf16[2048, 2048]", primals_135: "bf16[2048]", primals_136: "bf16[8192, 2048]", primals_137: "bf16[8192, 2048]", primals_138: "bf16[2048, 8192]", primals_139: "bf16[2048]", primals_140: "bf16[2048, 2048]", primals_141: "bf16[512, 2048]", primals_142: "bf16[512, 2048]", primals_143: "bf16[2048, 2048]", primals_144: "bf16[2048]", primals_145: "bf16[8192, 2048]", primals_146: "bf16[8192, 2048]", primals_147: "bf16[2048, 8192]", primals_148: "bf16[2048]", embedding: "bf16[4, 512, 2048]", rsqrt: "f32[4, 512, 1]", view_4: "bf16[2048, 2048]", add_4: "bf16[4, 32, 512, 64]", view_13: "bf16[4, 32, 512, 64]", view_14: "bf16[4, 32, 512, 64]", where: "bf16[4, 1, 512, 512]", getitem: "bf16[4, 32, 512, 64]", getitem_1: "f32[4, 32, 512, 1]", getitem_6: "i64[]", getitem_7: "i64[]", mm_3: "bf16[2048, 2048]", rsqrt_1: "f32[4, 512, 1]", view_18: "bf16[2048, 2048]", mm_4: "bf16[2048, 8192]", mm_5: "bf16[2048, 8192]", view_22: "bf16[2048, 8192]", add_9: "bf16[4, 512, 2048]", rsqrt_2: "f32[4, 512, 1]", view_24: "bf16[2048, 2048]", add_11: "bf16[4, 32, 512, 64]", view_33: "bf16[4, 32, 512, 64]", view_34: "bf16[4, 32, 512, 64]", getitem_9: "bf16[4, 32, 512, 64]", getitem_10: "f32[4, 32, 512, 1]", getitem_15: "i64[]", getitem_16: "i64[]", add_13: "bf16[4, 512, 2048]", rsqrt_3: "f32[4, 512, 1]", view_38: "bf16[2048, 2048]", mm_11: "bf16[2048, 8192]", mm_12: "bf16[2048, 8192]", view_42: "bf16[2048, 8192]", add_16: "bf16[4, 512, 2048]", rsqrt_4: "f32[4, 512, 1]", view_44: "bf16[2048, 2048]", add_18: "bf16[4, 32, 512, 64]", view_53: "bf16[4, 32, 512, 64]", view_54: "bf16[4, 32, 512, 64]", getitem_18: "bf16[4, 32, 512, 64]", getitem_19: "f32[4, 32, 512, 1]", getitem_24: "i64[]", getitem_25: "i64[]", add_20: "bf16[4, 512, 2048]", rsqrt_5: "f32[4, 512, 1]", view_58: "bf16[2048, 2048]", mm_18: "bf16[2048, 8192]", mm_19: "bf16[2048, 8192]", view_62: "bf16[2048, 8192]", add_23: "bf16[4, 512, 2048]", rsqrt_6: "f32[4, 512, 1]", view_64: "bf16[2048, 2048]", add_25: "bf16[4, 32, 512, 64]", view_73: "bf16[4, 32, 512, 64]", view_74: "bf16[4, 32, 512, 64]", getitem_27: "bf16[4, 32, 512, 64]", getitem_28: "f32[4, 32, 512, 1]", getitem_33: "i64[]", getitem_34: "i64[]", add_27: "bf16[4, 512, 2048]", rsqrt_7: "f32[4, 512, 1]", view_78: "bf16[2048, 2048]", mm_25: "bf16[2048, 8192]", mm_26: "bf16[2048, 8192]", view_82: "bf16[2048, 8192]", add_30: "bf16[4, 512, 2048]", rsqrt_8: "f32[4, 512, 1]", view_84: "bf16[2048, 2048]", add_32: "bf16[4, 32, 512, 64]", view_93: "bf16[4, 32, 512, 64]", view_94: "bf16[4, 32, 512, 64]", getitem_36: "bf16[4, 32, 512, 64]", getitem_37: "f32[4, 32, 512, 1]", getitem_42: "i64[]", getitem_43: "i64[]", add_34: "bf16[4, 512, 2048]", rsqrt_9: "f32[4, 512, 1]", view_98: "bf16[2048, 2048]", mm_32: "bf16[2048, 8192]", mm_33: "bf16[2048, 8192]", view_102: "bf16[2048, 8192]", add_37: "bf16[4, 512, 2048]", rsqrt_10: "f32[4, 512, 1]", view_104: "bf16[2048, 2048]", add_39: "bf16[4, 32, 512, 64]", view_113: "bf16[4, 32, 512, 64]", view_114: "bf16[4, 32, 512, 64]", getitem_45: "bf16[4, 32, 512, 64]", getitem_46: "f32[4, 32, 512, 1]", getitem_51: "i64[]", getitem_52: "i64[]", add_41: "bf16[4, 512, 2048]", rsqrt_11: "f32[4, 512, 1]", view_118: "bf16[2048, 2048]", mm_39: "bf16[2048, 8192]", mm_40: "bf16[2048, 8192]", view_122: "bf16[2048, 8192]", add_44: "bf16[4, 512, 2048]", rsqrt_12: "f32[4, 512, 1]", view_124: "bf16[2048, 2048]", add_46: "bf16[4, 32, 512, 64]", view_133: "bf16[4, 32, 512, 64]", view_134: "bf16[4, 32, 512, 64]", getitem_54: "bf16[4, 32, 512, 64]", getitem_55: "f32[4, 32, 512, 1]", getitem_60: "i64[]", getitem_61: "i64[]", add_48: "bf16[4, 512, 2048]", rsqrt_13: "f32[4, 512, 1]", view_138: "bf16[2048, 2048]", mm_46: "bf16[2048, 8192]", mm_47: "bf16[2048, 8192]", view_142: "bf16[2048, 8192]", add_51: "bf16[4, 512, 2048]", rsqrt_14: "f32[4, 512, 1]", view_144: "bf16[2048, 2048]", add_53: "bf16[4, 32, 512, 64]", view_153: "bf16[4, 32, 512, 64]", view_154: "bf16[4, 32, 512, 64]", getitem_63: "bf16[4, 32, 512, 64]", getitem_64: "f32[4, 32, 512, 1]", getitem_69: "i64[]", getitem_70: "i64[]", add_55: "bf16[4, 512, 2048]", rsqrt_15: "f32[4, 512, 1]", view_158: "bf16[2048, 2048]", mm_53: "bf16[2048, 8192]", mm_54: "bf16[2048, 8192]", view_162: "bf16[2048, 8192]", add_58: "bf16[4, 512, 2048]", rsqrt_16: "f32[4, 512, 1]", view_164: "bf16[2048, 2048]", add_60: "bf16[4, 32, 512, 64]", view_173: "bf16[4, 32, 512, 64]", view_174: "bf16[4, 32, 512, 64]", getitem_72: "bf16[4, 32, 512, 64]", getitem_73: "f32[4, 32, 512, 1]", getitem_78: "i64[]", getitem_79: "i64[]", add_62: "bf16[4, 512, 2048]", rsqrt_17: "f32[4, 512, 1]", view_178: "bf16[2048, 2048]", mm_60: "bf16[2048, 8192]", mm_61: "bf16[2048, 8192]", view_182: "bf16[2048, 8192]", add_65: "bf16[4, 512, 2048]", rsqrt_18: "f32[4, 512, 1]", view_184: "bf16[2048, 2048]", add_67: "bf16[4, 32, 512, 64]", view_193: "bf16[4, 32, 512, 64]", view_194: "bf16[4, 32, 512, 64]", getitem_81: "bf16[4, 32, 512, 64]", getitem_82: "f32[4, 32, 512, 1]", getitem_87: "i64[]", getitem_88: "i64[]", add_69: "bf16[4, 512, 2048]", rsqrt_19: "f32[4, 512, 1]", view_198: "bf16[2048, 2048]", mm_67: "bf16[2048, 8192]", mm_68: "bf16[2048, 8192]", view_202: "bf16[2048, 8192]", add_72: "bf16[4, 512, 2048]", rsqrt_20: "f32[4, 512, 1]", view_204: "bf16[2048, 2048]", add_74: "bf16[4, 32, 512, 64]", view_213: "bf16[4, 32, 512, 64]", view_214: "bf16[4, 32, 512, 64]", getitem_90: "bf16[4, 32, 512, 64]", getitem_91: "f32[4, 32, 512, 1]", getitem_96: "i64[]", getitem_97: "i64[]", add_76: "bf16[4, 512, 2048]", rsqrt_21: "f32[4, 512, 1]", view_218: "bf16[2048, 2048]", mm_74: "bf16[2048, 8192]", mm_75: "bf16[2048, 8192]", view_222: "bf16[2048, 8192]", add_79: "bf16[4, 512, 2048]", rsqrt_22: "f32[4, 512, 1]", view_224: "bf16[2048, 2048]", add_81: "bf16[4, 32, 512, 64]", view_233: "bf16[4, 32, 512, 64]", view_234: "bf16[4, 32, 512, 64]", getitem_99: "bf16[4, 32, 512, 64]", getitem_100: "f32[4, 32, 512, 1]", getitem_105: "i64[]", getitem_106: "i64[]", add_83: "bf16[4, 512, 2048]", rsqrt_23: "f32[4, 512, 1]", view_238: "bf16[2048, 2048]", mm_81: "bf16[2048, 8192]", mm_82: "bf16[2048, 8192]", view_242: "bf16[2048, 8192]", add_86: "bf16[4, 512, 2048]", rsqrt_24: "f32[4, 512, 1]", view_244: "bf16[2048, 2048]", add_88: "bf16[4, 32, 512, 64]", view_253: "bf16[4, 32, 512, 64]", view_254: "bf16[4, 32, 512, 64]", getitem_108: "bf16[4, 32, 512, 64]", getitem_109: "f32[4, 32, 512, 1]", getitem_114: "i64[]", getitem_115: "i64[]", add_90: "bf16[4, 512, 2048]", rsqrt_25: "f32[4, 512, 1]", view_258: "bf16[2048, 2048]", mm_88: "bf16[2048, 8192]", mm_89: "bf16[2048, 8192]", view_262: "bf16[2048, 8192]", add_93: "bf16[4, 512, 2048]", rsqrt_26: "f32[4, 512, 1]", view_264: "bf16[2048, 2048]", add_95: "bf16[4, 32, 512, 64]", view_273: "bf16[4, 32, 512, 64]", view_274: "bf16[4, 32, 512, 64]", getitem_117: "bf16[4, 32, 512, 64]", getitem_118: "f32[4, 32, 512, 1]", getitem_123: "i64[]", getitem_124: "i64[]", add_97: "bf16[4, 512, 2048]", rsqrt_27: "f32[4, 512, 1]", view_278: "bf16[2048, 2048]", mm_95: "bf16[2048, 8192]", mm_96: "bf16[2048, 8192]", view_282: "bf16[2048, 8192]", add_100: "bf16[4, 512, 2048]", rsqrt_28: "f32[4, 512, 1]", view_284: "bf16[2048, 2048]", add_102: "bf16[4, 32, 512, 64]", view_293: "bf16[4, 32, 512, 64]", view_294: "bf16[4, 32, 512, 64]", getitem_126: "bf16[4, 32, 512, 64]", getitem_127: "f32[4, 32, 512, 1]", getitem_132: "i64[]", getitem_133: "i64[]", add_104: "bf16[4, 512, 2048]", rsqrt_29: "f32[4, 512, 1]", view_298: "bf16[2048, 2048]", mm_102: "bf16[2048, 8192]", mm_103: "bf16[2048, 8192]", view_302: "bf16[2048, 8192]", add_107: "bf16[4, 512, 2048]", rsqrt_30: "f32[4, 512, 1]", view_304: "bf16[2048, 2048]", add_109: "bf16[4, 32, 512, 64]", view_313: "bf16[4, 32, 512, 64]", view_314: "bf16[4, 32, 512, 64]", getitem_135: "bf16[4, 32, 512, 64]", getitem_136: "f32[4, 32, 512, 1]", getitem_141: "i64[]", getitem_142: "i64[]", add_111: "bf16[4, 512, 2048]", rsqrt_31: "f32[4, 512, 1]", view_318: "bf16[2048, 2048]", mm_109: "bf16[2048, 8192]", mm_110: "bf16[2048, 8192]", view_322: "bf16[2048, 8192]", add_114: "bf16[4, 512, 2048]", rsqrt_32: "f32[4, 512, 1]", view_324: "bf16[2048, 2048]", view_325: "bf16[4, 512, 128256]", constant_pad_nd: "i64[4, 513]", amax: "f32[2048, 1]", log: "f32[2048, 1]", convert_element_type_328: "f32[]", tangents_1: "f32[]", tangents_2: "bf16[4, 512, 128256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_17: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_328);  tangents_1 = convert_element_type_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_68: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_34: "i64[4, 512]" = torch.ops.aten.clone.default(slice_68, memory_format = torch.contiguous_format);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_327: "i64[2048]" = torch.ops.aten.reshape.default(clone_34, [-1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_79: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(view_327, 1);  view_327 = None
        ne_4: "b8[2048, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_79, -100)
        full_default_33: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "i64[2048, 1]" = torch.ops.aten.where.self(ne_4, unsqueeze_79, full_default_33);  unsqueeze_79 = full_default_33 = None

        # No stacktrace found for following nodes
        iota_default: "i64[128256]" = torch.ops.prims.iota.default(128256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 128256]" = torch.ops.aten.reshape.default(iota_default, [1, 128256]);  iota_default = None
        expand_default: "i64[2048, 128256]" = torch.ops.aten.expand.default(where_18, [2048, 128256]);  where_18 = None
        eq_tensor: "b8[2048, 128256]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[2048, 128256]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_34: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f32[2048, 1]" = torch.ops.aten.where.self(ne_4, div_17, full_default_34);  ne_4 = div_17 = None
        mul_149: "f32[2048, 128256]" = torch.ops.aten.mul.Tensor(where_self, where_19);  where_self = where_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_327: "f32[4, 512, 128256]" = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_326: "f32[2048, 128256]" = torch.ops.aten.reshape.default(convert_element_type_327, [-1, 128256]);  convert_element_type_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_2: "f32[2048, 128256]" = torch.ops.aten.sub.Tensor(view_326, amax);  view_326 = amax = None
        sub_3: "f32[2048, 128256]" = torch.ops.aten.sub.Tensor(sub_2, log);  sub_2 = log = None
        exp_17: "f32[2048, 128256]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [1], True)
        mul_150: "f32[2048, 128256]" = torch.ops.aten.mul.Tensor(exp_17, sum_4);  exp_17 = sum_4 = None
        sub_4: "f32[2048, 128256]" = torch.ops.aten.sub.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_328: "f32[4, 512, 128256]" = torch.ops.aten.reshape.default(sub_4, [4, 512, 128256]);  sub_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_329: "bf16[4, 512, 128256]" = torch.ops.prims.convert_element_type.default(view_328, torch.bfloat16);  view_328 = None
        add_116: "bf16[4, 512, 128256]" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_329);  tangents_2 = convert_element_type_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:487 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_329: "bf16[2048, 128256]" = torch.ops.aten.reshape.default(add_116, [2048, 128256]);  add_116 = None
        permute_178: "bf16[128256, 2048]" = torch.ops.aten.permute.default(view_329, [1, 0])
        mm_113: "bf16[128256, 2048]" = torch.ops.aten.mm.default(permute_178, view_324);  permute_178 = view_324 = None
        permute_177: "bf16[2048, 128256]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_180: "bf16[128256, 2048]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        mm_114: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_329, permute_180);  view_329 = permute_180 = None
        view_330: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_114, [4, 512, 2048]);  mm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_151: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_330, primals_148);  primals_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_323: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_147: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_323, rsqrt_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_324: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None
        mul_152: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_330, convert_element_type_324);  view_330 = convert_element_type_324 = None
        sum_5: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_152, [0, 1], True);  mul_152 = None
        view_331: "bf16[2048]" = torch.ops.aten.reshape.default(sum_5, [2048]);  sum_5 = None
        convert_element_type_334: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_151, torch.float32);  mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_153: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_334, convert_element_type_323)
        mul_154: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_334, rsqrt_32);  convert_element_type_334 = None
        sum_6: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True);  mul_153 = None
        pow_34: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_32, 3);  rsqrt_32 = None
        mul_155: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_6, -0.5);  sum_6 = None
        mul_156: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_155, pow_34);  mul_155 = pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_38: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_156, [4, 512, 2048]);  mul_156 = None
        div_18: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_38, 2048);  expand_38 = None
        pow_35: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_323, 1.0);  convert_element_type_323 = None
        mul_157: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_35, 2.0);  pow_35 = None
        mul_158: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_18, mul_157);  div_18 = mul_157 = None
        add_117: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_154, mul_158);  mul_154 = mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_335: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_332: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(convert_element_type_335, [2048, 2048])
        permute_182: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_115: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_182, view_322);  permute_182 = view_322 = None
        permute_176: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_184: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_116: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_332, permute_184);  view_332 = permute_184 = None
        view_333: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_116, [4, 512, 8192]);  mm_116 = None
        view_319: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_109, [4, 512, 8192]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_317: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        neg_47: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_317)
        exp_15: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_113: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_317, add_113)
        convert_element_type_318: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_159: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_333, convert_element_type_318);  convert_element_type_318 = None
        view_321: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_110, [4, 512, 8192]);  mm_110 = None
        mul_160: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_333, view_321);  view_333 = view_321 = None
        view_334: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_159, [2048, 8192]);  mul_159 = None
        permute_186: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_117: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_186, view_318);  permute_186 = None
        permute_175: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_188: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_118: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_334, permute_188);  view_334 = permute_188 = None
        view_335: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_118, [4, 512, 2048]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_344: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_160, torch.float32);  mul_160 = None
        reciprocal: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_113);  add_113 = None
        mul_161: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_162: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_344, mul_161);  convert_element_type_344 = None
        sub_5: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_161);  mul_161 = None
        mul_163: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_317, sub_5);  convert_element_type_317 = sub_5 = None
        add_119: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_163, 1);  mul_163 = None
        mul_164: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_162, add_119);  mul_162 = add_119 = None
        convert_element_type_346: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_336: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_346, [2048, 8192]);  convert_element_type_346 = None
        permute_190: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_119: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_190, view_318);  permute_190 = view_318 = None
        permute_174: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        permute_192: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_120: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_336, permute_192);  view_336 = permute_192 = None
        view_337: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_120, [4, 512, 2048]);  mm_120 = None
        add_120: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_335, view_337);  view_335 = view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_165: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_120, primals_144);  primals_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_313: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_111, torch.float32);  add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_144: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_313, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_314: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        mul_166: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_120, convert_element_type_314);  add_120 = convert_element_type_314 = None
        sum_7: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1], True);  mul_166 = None
        view_338: "bf16[2048]" = torch.ops.aten.reshape.default(sum_7, [2048]);  sum_7 = None
        convert_element_type_351: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_165, torch.float32);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_167: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_351, convert_element_type_313)
        mul_168: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_351, rsqrt_31);  convert_element_type_351 = None
        sum_8: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True);  mul_167 = None
        pow_36: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
        mul_169: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_8, -0.5);  sum_8 = None
        mul_170: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_169, pow_36);  mul_169 = pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_39: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_170, [4, 512, 2048]);  mul_170 = None
        div_19: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_39, 2048);  expand_39 = None
        pow_37: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_313, 1.0);  convert_element_type_313 = None
        mul_171: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_37, 2.0);  pow_37 = None
        mul_172: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_19, mul_171);  div_19 = mul_171 = None
        add_121: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_168, mul_172);  mul_168 = mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_352: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        add_122: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_335, convert_element_type_352);  convert_element_type_335 = convert_element_type_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_339: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_122, [2048, 2048])
        permute_194: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_339, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_172: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_315: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_172, [4, 512, -1]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_316: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_315, [2048, 2048]);  view_315 = None
        mm_121: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_194, view_316);  permute_194 = view_316 = None
        permute_173: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_196: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_122: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_339, permute_196);  view_339 = permute_196 = None
        view_340: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_122, [4, 512, 2048]);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_341: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_340, [4, 512, 32, 64]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_198: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_198, add_109, view_313, view_314, getitem_135, getitem_136, getitem_141, getitem_142, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_198 = add_109 = view_313 = view_314 = getitem_135 = getitem_136 = getitem_141 = getitem_142 = None
        getitem_144: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_145: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_146: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_342: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_146, [4, 8, 4, 512, 64]);  getitem_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_9: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_342, [2], True);  view_342 = None
        squeeze_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_9, 2);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_343: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_145, [4, 8, 4, 512, 64]);  getitem_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_10: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_343, [2], True);  view_343 = None
        squeeze_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_10, 2);  sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:396 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:397 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:125 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        unsqueeze_11: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        expand_2: "f32[1, 32, 1]" = torch.ops.aten.expand.default(unsqueeze_11, [1, -1, 1]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:126 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        convert_element_type: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_5, torch.float32);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:130 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 32, 1]" = torch.ops.aten.expand.default(expand_2, [1, 32, 1]);  expand_2 = None
        expand_4: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type, [1, 1, 512]);  convert_element_type = None
        mul: "f32[1, 32, 512]" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 512, 32]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:131 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 32]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 512, 2, 32]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 32]);  unsqueeze_13 = None
        clone: "f32[1, 512, 2, 32]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 512, 64]" = torch.ops.aten.reshape.default(clone, [1, 512, 64]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:133 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 64]" = torch.ops.aten.sin.default(view_3)
        mul_2: "f32[1, 512, 64]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 512, 64]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_173: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_2, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_69: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_173, 3, 0, 32)
        slice_70: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_173, 3, 32, 64);  mul_173 = None
        neg_50: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_69);  slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_38: "bf16[4, 8, 512, 64]" = torch.ops.aten.full.default([4, 8, 512, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_50, 3, 32, 9223372036854775807);  neg_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_70, 3, 0, 32);  slice_70 = None
        add_123: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:132 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 64]" = torch.ops.aten.cos.default(view_3);  view_3 = None
        mul_1: "f32[1, 512, 64]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 512, 64]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_174: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_2, unsqueeze_14);  squeeze_2 = None
        add_124: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_123, mul_174);  add_123 = mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_175: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_144, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_71: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_175, 3, 0, 32)
        slice_72: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_175, 3, 32, 64);  mul_175 = None
        neg_51: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_71);  slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_40: "bf16[4, 32, 512, 64]" = torch.ops.aten.full.default([4, 32, 512, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_51, 3, 32, 9223372036854775807);  neg_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_3: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_72, 3, 0, 32);  slice_72 = None
        add_125: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_176: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_144, unsqueeze_14);  getitem_144 = None
        add_126: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_125, mul_176);  add_125 = mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_199: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1, 3]);  squeeze_1 = None
        clone_35: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_344: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_35, [4, 512, 512]);  clone_35 = None
        view_345: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_344, [2048, 512]);  view_344 = None
        permute_200: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_345, [1, 0])
        mm_123: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_200, view_304);  permute_200 = None
        permute_170: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_202: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        mm_124: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_345, permute_202);  view_345 = permute_202 = None
        view_346: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_124, [4, 512, 2048]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_204: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_124, [0, 2, 1, 3]);  add_124 = None
        clone_36: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_347: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_36, [4, 512, 512]);  clone_36 = None
        view_348: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_347, [2048, 512]);  view_347 = None
        permute_205: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_348, [1, 0])
        mm_125: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_205, view_304);  permute_205 = None
        permute_168: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_207: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        mm_126: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_348, permute_207);  view_348 = permute_207 = None
        view_349: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_126, [4, 512, 2048]);  mm_126 = None
        add_127: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_346, view_349);  view_346 = view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_209: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_126, [0, 2, 1, 3]);  add_126 = None
        clone_37: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_350: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_37, [4, 512, 2048]);  clone_37 = None
        view_351: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_350, [2048, 2048]);  view_350 = None
        permute_210: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_351, [1, 0])
        mm_127: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_210, view_304);  permute_210 = view_304 = None
        permute_166: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_212: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_128: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_351, permute_212);  view_351 = permute_212 = None
        view_352: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_128, [4, 512, 2048]);  mm_128 = None
        add_128: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_127, view_352);  add_127 = view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_177: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_128, primals_139);  primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_303: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_138: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_303, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_304: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_138, torch.bfloat16);  mul_138 = None
        mul_178: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_128, convert_element_type_304);  add_128 = convert_element_type_304 = None
        sum_11: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1], True);  mul_178 = None
        view_353: "bf16[2048]" = torch.ops.aten.reshape.default(sum_11, [2048]);  sum_11 = None
        convert_element_type_369: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_177, torch.float32);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_179: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_369, convert_element_type_303)
        mul_180: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_369, rsqrt_30);  convert_element_type_369 = None
        sum_12: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
        pow_38: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_30, 3);  rsqrt_30 = None
        mul_181: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_12, -0.5);  sum_12 = None
        mul_182: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_181, pow_38);  mul_181 = pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_40: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_182, [4, 512, 2048]);  mul_182 = None
        div_20: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_40, 2048);  expand_40 = None
        pow_39: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_303, 1.0);  convert_element_type_303 = None
        mul_183: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_39, 2.0);  pow_39 = None
        mul_184: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_20, mul_183);  div_20 = mul_183 = None
        add_129: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_180, mul_184);  mul_180 = mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_370: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        add_130: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_122, convert_element_type_370);  add_122 = convert_element_type_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_354: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_130, [2048, 2048])
        permute_214: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_129: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_214, view_302);  permute_214 = view_302 = None
        permute_165: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_216: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_130: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_354, permute_216);  view_354 = permute_216 = None
        view_355: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_130, [4, 512, 8192]);  mm_130 = None
        view_299: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_102, [4, 512, 8192]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_297: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        neg_44: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_297)
        exp_14: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_106: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_297, add_106)
        convert_element_type_298: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_185: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_355, convert_element_type_298);  convert_element_type_298 = None
        view_301: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_103, [4, 512, 8192]);  mm_103 = None
        mul_186: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_355, view_301);  view_355 = view_301 = None
        view_356: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_185, [2048, 8192]);  mul_185 = None
        permute_218: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_356, [1, 0])
        mm_131: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_218, view_298);  permute_218 = None
        permute_164: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_220: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_132: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_356, permute_220);  view_356 = permute_220 = None
        view_357: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_132, [4, 512, 2048]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_379: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_186, torch.float32);  mul_186 = None
        reciprocal_1: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_106);  add_106 = None
        mul_187: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_188: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_379, mul_187);  convert_element_type_379 = None
        sub_6: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_187);  mul_187 = None
        mul_189: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_297, sub_6);  convert_element_type_297 = sub_6 = None
        add_132: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_189, 1);  mul_189 = None
        mul_190: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_188, add_132);  mul_188 = add_132 = None
        convert_element_type_381: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_358: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_381, [2048, 8192]);  convert_element_type_381 = None
        permute_222: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_133: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_222, view_298);  permute_222 = view_298 = None
        permute_163: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_224: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_134: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_358, permute_224);  view_358 = permute_224 = None
        view_359: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_134, [4, 512, 2048]);  mm_134 = None
        add_133: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_357, view_359);  view_357 = view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_191: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_133, primals_135);  primals_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_293: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_104, torch.float32);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_135: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_293, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_294: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        mul_192: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_133, convert_element_type_294);  add_133 = convert_element_type_294 = None
        sum_13: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1], True);  mul_192 = None
        view_360: "bf16[2048]" = torch.ops.aten.reshape.default(sum_13, [2048]);  sum_13 = None
        convert_element_type_386: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_191, torch.float32);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_193: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_386, convert_element_type_293)
        mul_194: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_386, rsqrt_29);  convert_element_type_386 = None
        sum_14: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True);  mul_193 = None
        pow_40: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_29, 3);  rsqrt_29 = None
        mul_195: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_14, -0.5);  sum_14 = None
        mul_196: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_195, pow_40);  mul_195 = pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_41: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_196, [4, 512, 2048]);  mul_196 = None
        div_21: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_41, 2048);  expand_41 = None
        pow_41: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_293, 1.0);  convert_element_type_293 = None
        mul_197: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_41, 2.0);  pow_41 = None
        mul_198: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_21, mul_197);  div_21 = mul_197 = None
        add_134: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_194, mul_198);  mul_194 = mul_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_387: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None
        add_135: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_130, convert_element_type_387);  add_130 = convert_element_type_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_361: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_135, [2048, 2048])
        permute_226: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_361, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_295: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_161, [4, 512, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_296: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_295, [2048, 2048]);  view_295 = None
        mm_135: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_226, view_296);  permute_226 = view_296 = None
        permute_162: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_228: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_136: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_361, permute_228);  view_361 = permute_228 = None
        view_362: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_136, [4, 512, 2048]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_363: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_362, [4, 512, 32, 64]);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_230: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_230, add_102, view_293, view_294, getitem_126, getitem_127, getitem_132, getitem_133, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_230 = add_102 = view_293 = view_294 = getitem_126 = getitem_127 = getitem_132 = getitem_133 = None
        getitem_147: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_148: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_149: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_364: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_149, [4, 8, 4, 512, 64]);  getitem_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_15: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_364, [2], True);  view_364 = None
        squeeze_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_15, 2);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_365: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_148, [4, 8, 4, 512, 64]);  getitem_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_16: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_365, [2], True);  view_365 = None
        squeeze_4: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_16, 2);  sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_199: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_4, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_73: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_199, 3, 0, 32)
        slice_74: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_199, 3, 32, 64);  mul_199 = None
        neg_53: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_73);  slice_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_4: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_53, 3, 32, 9223372036854775807);  neg_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_5: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_74, 3, 0, 32);  slice_74 = None
        add_136: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_200: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_4, unsqueeze_14);  squeeze_4 = None
        add_137: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_136, mul_200);  add_136 = mul_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_201: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_147, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_75: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_201, 3, 0, 32)
        slice_76: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_201, 3, 32, 64);  mul_201 = None
        neg_54: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_75);  slice_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_6: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_54, 3, 32, 9223372036854775807);  neg_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_7: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_76, 3, 0, 32);  slice_76 = None
        add_138: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_202: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_147, unsqueeze_14);  getitem_147 = None
        add_139: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_138, mul_202);  add_138 = mul_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_231: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_3, [0, 2, 1, 3]);  squeeze_3 = None
        clone_38: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_366: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_38, [4, 512, 512]);  clone_38 = None
        view_367: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_366, [2048, 512]);  view_366 = None
        permute_232: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_137: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_232, view_284);  permute_232 = None
        permute_159: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_234: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        mm_138: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_367, permute_234);  view_367 = permute_234 = None
        view_368: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_138, [4, 512, 2048]);  mm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_236: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_137, [0, 2, 1, 3]);  add_137 = None
        clone_39: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_369: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_39, [4, 512, 512]);  clone_39 = None
        view_370: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_369, [2048, 512]);  view_369 = None
        permute_237: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_370, [1, 0])
        mm_139: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_237, view_284);  permute_237 = None
        permute_157: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_239: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        mm_140: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_370, permute_239);  view_370 = permute_239 = None
        view_371: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_140, [4, 512, 2048]);  mm_140 = None
        add_140: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_368, view_371);  view_368 = view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_241: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_139, [0, 2, 1, 3]);  add_139 = None
        clone_40: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        view_372: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_40, [4, 512, 2048]);  clone_40 = None
        view_373: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_372, [2048, 2048]);  view_372 = None
        permute_242: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_373, [1, 0])
        mm_141: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_242, view_284);  permute_242 = view_284 = None
        permute_155: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_244: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        mm_142: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_373, permute_244);  view_373 = permute_244 = None
        view_374: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_142, [4, 512, 2048]);  mm_142 = None
        add_141: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_140, view_374);  add_140 = view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_203: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_141, primals_130);  primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_283: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_100, torch.float32);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_129: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_283, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_284: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        mul_204: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_141, convert_element_type_284);  add_141 = convert_element_type_284 = None
        sum_17: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 1], True);  mul_204 = None
        view_375: "bf16[2048]" = torch.ops.aten.reshape.default(sum_17, [2048]);  sum_17 = None
        convert_element_type_404: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_203, torch.float32);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_205: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_404, convert_element_type_283)
        mul_206: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_404, rsqrt_28);  convert_element_type_404 = None
        sum_18: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True);  mul_205 = None
        pow_42: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_28, 3);  rsqrt_28 = None
        mul_207: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_18, -0.5);  sum_18 = None
        mul_208: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_207, pow_42);  mul_207 = pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_42: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_208, [4, 512, 2048]);  mul_208 = None
        div_22: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_42, 2048);  expand_42 = None
        pow_43: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_283, 1.0);  convert_element_type_283 = None
        mul_209: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_43, 2.0);  pow_43 = None
        mul_210: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_22, mul_209);  div_22 = mul_209 = None
        add_142: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_206, mul_210);  mul_206 = mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_405: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None
        add_143: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_135, convert_element_type_405);  add_135 = convert_element_type_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_376: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_143, [2048, 2048])
        permute_246: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_376, [1, 0])
        mm_143: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_246, view_282);  permute_246 = view_282 = None
        permute_154: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_248: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_144: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_376, permute_248);  view_376 = permute_248 = None
        view_377: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_144, [4, 512, 8192]);  mm_144 = None
        view_279: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_95, [4, 512, 8192]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_277: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        neg_41: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_277)
        exp_13: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_99: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_277, add_99)
        convert_element_type_278: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_211: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_377, convert_element_type_278);  convert_element_type_278 = None
        view_281: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_96, [4, 512, 8192]);  mm_96 = None
        mul_212: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_377, view_281);  view_377 = view_281 = None
        view_378: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_211, [2048, 8192]);  mul_211 = None
        permute_250: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_378, [1, 0])
        mm_145: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_250, view_278);  permute_250 = None
        permute_153: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_252: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_146: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_378, permute_252);  view_378 = permute_252 = None
        view_379: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_146, [4, 512, 2048]);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_414: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_212, torch.float32);  mul_212 = None
        reciprocal_2: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_99);  add_99 = None
        mul_213: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_214: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_414, mul_213);  convert_element_type_414 = None
        sub_7: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_213);  mul_213 = None
        mul_215: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_277, sub_7);  convert_element_type_277 = sub_7 = None
        add_145: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_215, 1);  mul_215 = None
        mul_216: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_214, add_145);  mul_214 = add_145 = None
        convert_element_type_416: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_216, torch.bfloat16);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_380: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_416, [2048, 8192]);  convert_element_type_416 = None
        permute_254: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_147: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_254, view_278);  permute_254 = view_278 = None
        permute_152: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_256: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_148: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_380, permute_256);  view_380 = permute_256 = None
        view_381: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_148, [4, 512, 2048]);  mm_148 = None
        add_146: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_379, view_381);  view_379 = view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_217: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_146, primals_126);  primals_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_273: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_97, torch.float32);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_126: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_273, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_274: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        mul_218: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_146, convert_element_type_274);  add_146 = convert_element_type_274 = None
        sum_19: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1], True);  mul_218 = None
        view_382: "bf16[2048]" = torch.ops.aten.reshape.default(sum_19, [2048]);  sum_19 = None
        convert_element_type_421: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_217, torch.float32);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_219: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_421, convert_element_type_273)
        mul_220: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_421, rsqrt_27);  convert_element_type_421 = None
        sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [2], True);  mul_219 = None
        pow_44: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_27, 3);  rsqrt_27 = None
        mul_221: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_20, -0.5);  sum_20 = None
        mul_222: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_221, pow_44);  mul_221 = pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_43: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_222, [4, 512, 2048]);  mul_222 = None
        div_23: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_43, 2048);  expand_43 = None
        pow_45: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_273, 1.0);  convert_element_type_273 = None
        mul_223: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_45, 2.0);  pow_45 = None
        mul_224: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_23, mul_223);  div_23 = mul_223 = None
        add_147: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_422: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_147, torch.bfloat16);  add_147 = None
        add_148: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_143, convert_element_type_422);  add_143 = convert_element_type_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_383: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_148, [2048, 2048])
        permute_258: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_383, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_275: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_150, [4, 512, -1]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_276: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_275, [2048, 2048]);  view_275 = None
        mm_149: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_258, view_276);  permute_258 = view_276 = None
        permute_151: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_260: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_150: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_383, permute_260);  view_383 = permute_260 = None
        view_384: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_150, [4, 512, 2048]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_385: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_384, [4, 512, 32, 64]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_262: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_262, add_95, view_273, view_274, getitem_117, getitem_118, getitem_123, getitem_124, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_262 = add_95 = view_273 = view_274 = getitem_117 = getitem_118 = getitem_123 = getitem_124 = None
        getitem_150: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_151: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_152: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_386: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_152, [4, 8, 4, 512, 64]);  getitem_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_21: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_386, [2], True);  view_386 = None
        squeeze_5: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_21, 2);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_387: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_151, [4, 8, 4, 512, 64]);  getitem_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_22: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_387, [2], True);  view_387 = None
        squeeze_6: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_22, 2);  sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_225: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_6, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_77: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_225, 3, 0, 32)
        slice_78: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_225, 3, 32, 64);  mul_225 = None
        neg_56: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_77);  slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_8: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_56, 3, 32, 9223372036854775807);  neg_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_9: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_78, 3, 0, 32);  slice_78 = None
        add_149: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_226: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_6, unsqueeze_14);  squeeze_6 = None
        add_150: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_149, mul_226);  add_149 = mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_227: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_150, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_79: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_227, 3, 0, 32)
        slice_80: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_227, 3, 32, 64);  mul_227 = None
        neg_57: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_79);  slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_10: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_57, 3, 32, 9223372036854775807);  neg_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_11: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_80, 3, 0, 32);  slice_80 = None
        add_151: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_228: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_150, unsqueeze_14);  getitem_150 = None
        add_152: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_151, mul_228);  add_151 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_263: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_5, [0, 2, 1, 3]);  squeeze_5 = None
        clone_41: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_388: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_41, [4, 512, 512]);  clone_41 = None
        view_389: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_388, [2048, 512]);  view_388 = None
        permute_264: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_151: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_264, view_264);  permute_264 = None
        permute_148: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_266: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        mm_152: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_389, permute_266);  view_389 = permute_266 = None
        view_390: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_152, [4, 512, 2048]);  mm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_268: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_150, [0, 2, 1, 3]);  add_150 = None
        clone_42: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
        view_391: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_42, [4, 512, 512]);  clone_42 = None
        view_392: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_391, [2048, 512]);  view_391 = None
        permute_269: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_153: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_269, view_264);  permute_269 = None
        permute_146: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_271: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        mm_154: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_392, permute_271);  view_392 = permute_271 = None
        view_393: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_154, [4, 512, 2048]);  mm_154 = None
        add_153: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_390, view_393);  view_390 = view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_273: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None
        clone_43: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
        view_394: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_43, [4, 512, 2048]);  clone_43 = None
        view_395: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_394, [2048, 2048]);  view_394 = None
        permute_274: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_155: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_274, view_264);  permute_274 = view_264 = None
        permute_144: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_276: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_156: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_395, permute_276);  view_395 = permute_276 = None
        view_396: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_156, [4, 512, 2048]);  mm_156 = None
        add_154: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_153, view_396);  add_153 = view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_229: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_154, primals_121);  primals_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_263: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_120: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_263, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_264: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        mul_230: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_154, convert_element_type_264);  add_154 = convert_element_type_264 = None
        sum_23: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1], True);  mul_230 = None
        view_397: "bf16[2048]" = torch.ops.aten.reshape.default(sum_23, [2048]);  sum_23 = None
        convert_element_type_439: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_229, torch.float32);  mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_231: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_439, convert_element_type_263)
        mul_232: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_439, rsqrt_26);  convert_element_type_439 = None
        sum_24: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        pow_46: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_26, 3);  rsqrt_26 = None
        mul_233: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
        mul_234: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_233, pow_46);  mul_233 = pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_44: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_234, [4, 512, 2048]);  mul_234 = None
        div_24: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_44, 2048);  expand_44 = None
        pow_47: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_263, 1.0);  convert_element_type_263 = None
        mul_235: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_47, 2.0);  pow_47 = None
        mul_236: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_24, mul_235);  div_24 = mul_235 = None
        add_155: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_232, mul_236);  mul_232 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_440: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None
        add_156: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_148, convert_element_type_440);  add_148 = convert_element_type_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_398: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_156, [2048, 2048])
        permute_278: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_398, [1, 0])
        mm_157: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_278, view_262);  permute_278 = view_262 = None
        permute_143: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_280: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_158: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_398, permute_280);  view_398 = permute_280 = None
        view_399: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_158, [4, 512, 8192]);  mm_158 = None
        view_259: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_88, [4, 512, 8192]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_257: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_259, torch.float32);  view_259 = None
        neg_38: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_257)
        exp_12: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_92: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_257, add_92)
        convert_element_type_258: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_237: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_399, convert_element_type_258);  convert_element_type_258 = None
        view_261: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_89, [4, 512, 8192]);  mm_89 = None
        mul_238: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_399, view_261);  view_399 = view_261 = None
        view_400: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_237, [2048, 8192]);  mul_237 = None
        permute_282: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_400, [1, 0])
        mm_159: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_282, view_258);  permute_282 = None
        permute_142: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_284: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_160: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_400, permute_284);  view_400 = permute_284 = None
        view_401: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_160, [4, 512, 2048]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_449: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_238, torch.float32);  mul_238 = None
        reciprocal_3: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_92);  add_92 = None
        mul_239: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_240: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_449, mul_239);  convert_element_type_449 = None
        sub_8: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_239);  mul_239 = None
        mul_241: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_257, sub_8);  convert_element_type_257 = sub_8 = None
        add_158: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_241, 1);  mul_241 = None
        mul_242: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_240, add_158);  mul_240 = add_158 = None
        convert_element_type_451: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_242, torch.bfloat16);  mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_402: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_451, [2048, 8192]);  convert_element_type_451 = None
        permute_286: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_402, [1, 0])
        mm_161: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_286, view_258);  permute_286 = view_258 = None
        permute_141: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_288: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_162: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_402, permute_288);  view_402 = permute_288 = None
        view_403: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_162, [4, 512, 2048]);  mm_162 = None
        add_159: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_401, view_403);  view_401 = view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_243: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_159, primals_117);  primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_253: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_117: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_253, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_254: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        mul_244: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_159, convert_element_type_254);  add_159 = convert_element_type_254 = None
        sum_25: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 1], True);  mul_244 = None
        view_404: "bf16[2048]" = torch.ops.aten.reshape.default(sum_25, [2048]);  sum_25 = None
        convert_element_type_456: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_243, torch.float32);  mul_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_245: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_456, convert_element_type_253)
        mul_246: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_456, rsqrt_25);  convert_element_type_456 = None
        sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
        pow_48: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
        mul_247: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_26, -0.5);  sum_26 = None
        mul_248: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_247, pow_48);  mul_247 = pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_45: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_248, [4, 512, 2048]);  mul_248 = None
        div_25: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_45, 2048);  expand_45 = None
        pow_49: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_253, 1.0);  convert_element_type_253 = None
        mul_249: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_49, 2.0);  pow_49 = None
        mul_250: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_25, mul_249);  div_25 = mul_249 = None
        add_160: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_246, mul_250);  mul_246 = mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_457: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16);  add_160 = None
        add_161: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_156, convert_element_type_457);  add_156 = convert_element_type_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_405: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_161, [2048, 2048])
        permute_290: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_405, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_255: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_139, [4, 512, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_256: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_255, [2048, 2048]);  view_255 = None
        mm_163: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_290, view_256);  permute_290 = view_256 = None
        permute_140: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_292: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_164: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_405, permute_292);  view_405 = permute_292 = None
        view_406: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_164, [4, 512, 2048]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_407: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_406, [4, 512, 32, 64]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_294: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_294, add_88, view_253, view_254, getitem_108, getitem_109, getitem_114, getitem_115, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_294 = add_88 = view_253 = view_254 = getitem_108 = getitem_109 = getitem_114 = getitem_115 = None
        getitem_153: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_154: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_155: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_408: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_155, [4, 8, 4, 512, 64]);  getitem_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_27: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_408, [2], True);  view_408 = None
        squeeze_7: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_27, 2);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_409: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_154, [4, 8, 4, 512, 64]);  getitem_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_28: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_409, [2], True);  view_409 = None
        squeeze_8: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_28, 2);  sum_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_251: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_8, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_81: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_251, 3, 0, 32)
        slice_82: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_251, 3, 32, 64);  mul_251 = None
        neg_59: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_81);  slice_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_12: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_59, 3, 32, 9223372036854775807);  neg_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_13: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_82, 3, 0, 32);  slice_82 = None
        add_162: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_252: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_8, unsqueeze_14);  squeeze_8 = None
        add_163: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_162, mul_252);  add_162 = mul_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_253: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_153, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_83: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_253, 3, 0, 32)
        slice_84: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_253, 3, 32, 64);  mul_253 = None
        neg_60: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_83);  slice_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_14: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_60, 3, 32, 9223372036854775807);  neg_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_15: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_84, 3, 0, 32);  slice_84 = None
        add_164: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_254: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_153, unsqueeze_14);  getitem_153 = None
        add_165: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_164, mul_254);  add_164 = mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_295: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_7, [0, 2, 1, 3]);  squeeze_7 = None
        clone_44: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
        view_410: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_44, [4, 512, 512]);  clone_44 = None
        view_411: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_410, [2048, 512]);  view_410 = None
        permute_296: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_411, [1, 0])
        mm_165: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_296, view_244);  permute_296 = None
        permute_137: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_298: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        mm_166: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_411, permute_298);  view_411 = permute_298 = None
        view_412: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_166, [4, 512, 2048]);  mm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_300: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_163, [0, 2, 1, 3]);  add_163 = None
        clone_45: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
        view_413: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_45, [4, 512, 512]);  clone_45 = None
        view_414: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_413, [2048, 512]);  view_413 = None
        permute_301: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_167: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_301, view_244);  permute_301 = None
        permute_135: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_303: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
        mm_168: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_414, permute_303);  view_414 = permute_303 = None
        view_415: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_168, [4, 512, 2048]);  mm_168 = None
        add_166: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_412, view_415);  view_412 = view_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_305: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_165, [0, 2, 1, 3]);  add_165 = None
        clone_46: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
        view_416: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_46, [4, 512, 2048]);  clone_46 = None
        view_417: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_416, [2048, 2048]);  view_416 = None
        permute_306: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_169: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_306, view_244);  permute_306 = view_244 = None
        permute_133: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        permute_308: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_170: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_417, permute_308);  view_417 = permute_308 = None
        view_418: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_170, [4, 512, 2048]);  mm_170 = None
        add_167: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_166, view_418);  add_166 = view_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_255: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_167, primals_112);  primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_243: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_86, torch.float32);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_111: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_243, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_244: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None
        mul_256: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_167, convert_element_type_244);  add_167 = convert_element_type_244 = None
        sum_29: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1], True);  mul_256 = None
        view_419: "bf16[2048]" = torch.ops.aten.reshape.default(sum_29, [2048]);  sum_29 = None
        convert_element_type_474: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_255, torch.float32);  mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_257: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_474, convert_element_type_243)
        mul_258: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_474, rsqrt_24);  convert_element_type_474 = None
        sum_30: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
        pow_50: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
        mul_259: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_30, -0.5);  sum_30 = None
        mul_260: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_259, pow_50);  mul_259 = pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_46: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_260, [4, 512, 2048]);  mul_260 = None
        div_26: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_46, 2048);  expand_46 = None
        pow_51: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_243, 1.0);  convert_element_type_243 = None
        mul_261: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_51, 2.0);  pow_51 = None
        mul_262: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_26, mul_261);  div_26 = mul_261 = None
        add_168: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_258, mul_262);  mul_258 = mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_475: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16);  add_168 = None
        add_169: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_161, convert_element_type_475);  add_161 = convert_element_type_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_420: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_169, [2048, 2048])
        permute_310: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_171: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_310, view_242);  permute_310 = view_242 = None
        permute_132: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_312: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_172: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_420, permute_312);  view_420 = permute_312 = None
        view_421: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_172, [4, 512, 8192]);  mm_172 = None
        view_239: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_81, [4, 512, 8192]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_237: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        neg_35: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_237)
        exp_11: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_85: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_237, add_85)
        convert_element_type_238: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_263: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_421, convert_element_type_238);  convert_element_type_238 = None
        view_241: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_82, [4, 512, 8192]);  mm_82 = None
        mul_264: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_421, view_241);  view_421 = view_241 = None
        view_422: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_263, [2048, 8192]);  mul_263 = None
        permute_314: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_173: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_314, view_238);  permute_314 = None
        permute_131: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_316: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_174: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_422, permute_316);  view_422 = permute_316 = None
        view_423: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_174, [4, 512, 2048]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_484: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_264, torch.float32);  mul_264 = None
        reciprocal_4: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_85);  add_85 = None
        mul_265: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_266: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_484, mul_265);  convert_element_type_484 = None
        sub_9: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_265);  mul_265 = None
        mul_267: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_237, sub_9);  convert_element_type_237 = sub_9 = None
        add_171: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_267, 1);  mul_267 = None
        mul_268: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_266, add_171);  mul_266 = add_171 = None
        convert_element_type_486: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16);  mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_424: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_486, [2048, 8192]);  convert_element_type_486 = None
        permute_318: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_424, [1, 0])
        mm_175: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_318, view_238);  permute_318 = view_238 = None
        permute_130: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_320: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_176: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_424, permute_320);  view_424 = permute_320 = None
        view_425: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_176, [4, 512, 2048]);  mm_176 = None
        add_172: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_423, view_425);  view_423 = view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_269: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_172, primals_108);  primals_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_233: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_108: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_234: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None
        mul_270: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_172, convert_element_type_234);  add_172 = convert_element_type_234 = None
        sum_31: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1], True);  mul_270 = None
        view_426: "bf16[2048]" = torch.ops.aten.reshape.default(sum_31, [2048]);  sum_31 = None
        convert_element_type_491: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_269, torch.float32);  mul_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_271: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_491, convert_element_type_233)
        mul_272: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_491, rsqrt_23);  convert_element_type_491 = None
        sum_32: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        pow_52: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_23, 3);  rsqrt_23 = None
        mul_273: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
        mul_274: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_273, pow_52);  mul_273 = pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_47: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_274, [4, 512, 2048]);  mul_274 = None
        div_27: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_47, 2048);  expand_47 = None
        pow_53: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 1.0);  convert_element_type_233 = None
        mul_275: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_53, 2.0);  pow_53 = None
        mul_276: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_27, mul_275);  div_27 = mul_275 = None
        add_173: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_272, mul_276);  mul_272 = mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_492: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None
        add_174: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_169, convert_element_type_492);  add_169 = convert_element_type_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_427: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_174, [2048, 2048])
        permute_322: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_427, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_128, [4, 512, -1]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_236: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_235, [2048, 2048]);  view_235 = None
        mm_177: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_322, view_236);  permute_322 = view_236 = None
        permute_129: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_324: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_178: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_427, permute_324);  view_427 = permute_324 = None
        view_428: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_178, [4, 512, 2048]);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_429: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_428, [4, 512, 32, 64]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_326: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_4 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_326, add_81, view_233, view_234, getitem_99, getitem_100, getitem_105, getitem_106, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_326 = add_81 = view_233 = view_234 = getitem_99 = getitem_100 = getitem_105 = getitem_106 = None
        getitem_156: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_4[0]
        getitem_157: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_4[1]
        getitem_158: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_4[2];  _scaled_dot_product_cudnn_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_430: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_158, [4, 8, 4, 512, 64]);  getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_33: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_430, [2], True);  view_430 = None
        squeeze_9: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_33, 2);  sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_431: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_157, [4, 8, 4, 512, 64]);  getitem_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_34: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_431, [2], True);  view_431 = None
        squeeze_10: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_34, 2);  sum_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_277: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_10, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_85: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_277, 3, 0, 32)
        slice_86: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_277, 3, 32, 64);  mul_277 = None
        neg_62: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_85);  slice_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_16: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_62, 3, 32, 9223372036854775807);  neg_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_17: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_86, 3, 0, 32);  slice_86 = None
        add_175: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_278: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_10, unsqueeze_14);  squeeze_10 = None
        add_176: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_175, mul_278);  add_175 = mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_279: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_156, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_87: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_279, 3, 0, 32)
        slice_88: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_279, 3, 32, 64);  mul_279 = None
        neg_63: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_87);  slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_18: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_63, 3, 32, 9223372036854775807);  neg_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_19: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_88, 3, 0, 32);  slice_88 = None
        add_177: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_280: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_156, unsqueeze_14);  getitem_156 = None
        add_178: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_177, mul_280);  add_177 = mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_327: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_9, [0, 2, 1, 3]);  squeeze_9 = None
        clone_47: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
        view_432: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_47, [4, 512, 512]);  clone_47 = None
        view_433: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_432, [2048, 512]);  view_432 = None
        permute_328: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_179: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_328, view_224);  permute_328 = None
        permute_126: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_330: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        mm_180: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_433, permute_330);  view_433 = permute_330 = None
        view_434: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_180, [4, 512, 2048]);  mm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_332: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_176, [0, 2, 1, 3]);  add_176 = None
        clone_48: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_332, memory_format = torch.contiguous_format);  permute_332 = None
        view_435: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_48, [4, 512, 512]);  clone_48 = None
        view_436: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_435, [2048, 512]);  view_435 = None
        permute_333: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_181: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_333, view_224);  permute_333 = None
        permute_124: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_335: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        mm_182: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_436, permute_335);  view_436 = permute_335 = None
        view_437: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_182, [4, 512, 2048]);  mm_182 = None
        add_179: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_434, view_437);  view_434 = view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_337: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_178, [0, 2, 1, 3]);  add_178 = None
        clone_49: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_438: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_49, [4, 512, 2048]);  clone_49 = None
        view_439: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_438, [2048, 2048]);  view_438 = None
        permute_338: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_183: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_338, view_224);  permute_338 = view_224 = None
        permute_122: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_340: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_184: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_439, permute_340);  view_439 = permute_340 = None
        view_440: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_184, [4, 512, 2048]);  mm_184 = None
        add_180: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_179, view_440);  add_179 = view_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_281: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_180, primals_103);  primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_223: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_102: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_223, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_224: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        mul_282: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_180, convert_element_type_224);  add_180 = convert_element_type_224 = None
        sum_35: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1], True);  mul_282 = None
        view_441: "bf16[2048]" = torch.ops.aten.reshape.default(sum_35, [2048]);  sum_35 = None
        convert_element_type_509: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_281, torch.float32);  mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_283: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_509, convert_element_type_223)
        mul_284: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_509, rsqrt_22);  convert_element_type_509 = None
        sum_36: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True);  mul_283 = None
        pow_54: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_22, 3);  rsqrt_22 = None
        mul_285: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_36, -0.5);  sum_36 = None
        mul_286: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_285, pow_54);  mul_285 = pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_48: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_286, [4, 512, 2048]);  mul_286 = None
        div_28: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_48, 2048);  expand_48 = None
        pow_55: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_223, 1.0);  convert_element_type_223 = None
        mul_287: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_55, 2.0);  pow_55 = None
        mul_288: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_28, mul_287);  div_28 = mul_287 = None
        add_181: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_284, mul_288);  mul_284 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_510: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16);  add_181 = None
        add_182: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_174, convert_element_type_510);  add_174 = convert_element_type_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_442: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_182, [2048, 2048])
        permute_342: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_185: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_342, view_222);  permute_342 = view_222 = None
        permute_121: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_344: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_186: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_442, permute_344);  view_442 = permute_344 = None
        view_443: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_186, [4, 512, 8192]);  mm_186 = None
        view_219: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_74, [4, 512, 8192]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_217: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        neg_32: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_217)
        exp_10: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_78: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_217, add_78)
        convert_element_type_218: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_289: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_443, convert_element_type_218);  convert_element_type_218 = None
        view_221: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_75, [4, 512, 8192]);  mm_75 = None
        mul_290: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_443, view_221);  view_443 = view_221 = None
        view_444: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_289, [2048, 8192]);  mul_289 = None
        permute_346: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_444, [1, 0])
        mm_187: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_346, view_218);  permute_346 = None
        permute_120: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_348: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_188: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_444, permute_348);  view_444 = permute_348 = None
        view_445: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_188, [4, 512, 2048]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_519: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_290, torch.float32);  mul_290 = None
        reciprocal_5: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_78);  add_78 = None
        mul_291: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_292: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_519, mul_291);  convert_element_type_519 = None
        sub_10: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_291);  mul_291 = None
        mul_293: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_217, sub_10);  convert_element_type_217 = sub_10 = None
        add_184: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_293, 1);  mul_293 = None
        mul_294: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_292, add_184);  mul_292 = add_184 = None
        convert_element_type_521: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_294, torch.bfloat16);  mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_446: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_521, [2048, 8192]);  convert_element_type_521 = None
        permute_350: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_446, [1, 0])
        mm_189: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_350, view_218);  permute_350 = view_218 = None
        permute_119: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_352: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_190: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_446, permute_352);  view_446 = permute_352 = None
        view_447: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_190, [4, 512, 2048]);  mm_190 = None
        add_185: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_445, view_447);  view_445 = view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_295: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_185, primals_99);  primals_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_213: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_76, torch.float32);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_99: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_213, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_214: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        mul_296: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_185, convert_element_type_214);  add_185 = convert_element_type_214 = None
        sum_37: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1], True);  mul_296 = None
        view_448: "bf16[2048]" = torch.ops.aten.reshape.default(sum_37, [2048]);  sum_37 = None
        convert_element_type_526: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_295, torch.float32);  mul_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_297: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_526, convert_element_type_213)
        mul_298: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_526, rsqrt_21);  convert_element_type_526 = None
        sum_38: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
        pow_56: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
        mul_299: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_38, -0.5);  sum_38 = None
        mul_300: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_299, pow_56);  mul_299 = pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_49: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_300, [4, 512, 2048]);  mul_300 = None
        div_29: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_49, 2048);  expand_49 = None
        pow_57: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_213, 1.0);  convert_element_type_213 = None
        mul_301: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_57, 2.0);  pow_57 = None
        mul_302: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_29, mul_301);  div_29 = mul_301 = None
        add_186: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_298, mul_302);  mul_298 = mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_527: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16);  add_186 = None
        add_187: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_182, convert_element_type_527);  add_182 = convert_element_type_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_449: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_187, [2048, 2048])
        permute_354: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_449, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_117, [4, 512, -1]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_216: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_215, [2048, 2048]);  view_215 = None
        mm_191: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_354, view_216);  permute_354 = view_216 = None
        permute_118: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_356: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_192: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_449, permute_356);  view_449 = permute_356 = None
        view_450: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_192, [4, 512, 2048]);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_451: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_450, [4, 512, 32, 64]);  view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_358: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_5 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_358, add_74, view_213, view_214, getitem_90, getitem_91, getitem_96, getitem_97, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_358 = add_74 = view_213 = view_214 = getitem_90 = getitem_91 = getitem_96 = getitem_97 = None
        getitem_159: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_5[0]
        getitem_160: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_5[1]
        getitem_161: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_5[2];  _scaled_dot_product_cudnn_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_452: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_161, [4, 8, 4, 512, 64]);  getitem_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_39: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_452, [2], True);  view_452 = None
        squeeze_11: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_39, 2);  sum_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_453: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_160, [4, 8, 4, 512, 64]);  getitem_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_40: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_453, [2], True);  view_453 = None
        squeeze_12: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_40, 2);  sum_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_303: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_12, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_89: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_303, 3, 0, 32)
        slice_90: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_303, 3, 32, 64);  mul_303 = None
        neg_65: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_89);  slice_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_20: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_65, 3, 32, 9223372036854775807);  neg_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_21: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_90, 3, 0, 32);  slice_90 = None
        add_188: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_304: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_12, unsqueeze_14);  squeeze_12 = None
        add_189: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_188, mul_304);  add_188 = mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_305: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_159, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_91: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_305, 3, 0, 32)
        slice_92: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_305, 3, 32, 64);  mul_305 = None
        neg_66: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_91);  slice_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_22: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_66, 3, 32, 9223372036854775807);  neg_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_23: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_92, 3, 0, 32);  slice_92 = None
        add_190: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_306: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_159, unsqueeze_14);  getitem_159 = None
        add_191: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_190, mul_306);  add_190 = mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_359: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_11, [0, 2, 1, 3]);  squeeze_11 = None
        clone_50: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_454: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_50, [4, 512, 512]);  clone_50 = None
        view_455: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_454, [2048, 512]);  view_454 = None
        permute_360: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_193: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_360, view_204);  permute_360 = None
        permute_115: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_362: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        mm_194: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_455, permute_362);  view_455 = permute_362 = None
        view_456: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_194, [4, 512, 2048]);  mm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_364: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_189, [0, 2, 1, 3]);  add_189 = None
        clone_51: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format);  permute_364 = None
        view_457: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_51, [4, 512, 512]);  clone_51 = None
        view_458: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_457, [2048, 512]);  view_457 = None
        permute_365: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_458, [1, 0])
        mm_195: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_365, view_204);  permute_365 = None
        permute_113: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_367: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        mm_196: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_458, permute_367);  view_458 = permute_367 = None
        view_459: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_196, [4, 512, 2048]);  mm_196 = None
        add_192: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_456, view_459);  view_456 = view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_369: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_191, [0, 2, 1, 3]);  add_191 = None
        clone_52: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_460: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_52, [4, 512, 2048]);  clone_52 = None
        view_461: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_460, [2048, 2048]);  view_460 = None
        permute_370: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_461, [1, 0])
        mm_197: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_370, view_204);  permute_370 = view_204 = None
        permute_111: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_372: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_198: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_461, permute_372);  view_461 = permute_372 = None
        view_462: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_198, [4, 512, 2048]);  mm_198 = None
        add_193: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_192, view_462);  add_192 = view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_307: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_193, primals_94);  primals_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_203: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_93: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_203, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_204: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        mul_308: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_193, convert_element_type_204);  add_193 = convert_element_type_204 = None
        sum_41: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1], True);  mul_308 = None
        view_463: "bf16[2048]" = torch.ops.aten.reshape.default(sum_41, [2048]);  sum_41 = None
        convert_element_type_544: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_307, torch.float32);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_309: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_544, convert_element_type_203)
        mul_310: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_544, rsqrt_20);  convert_element_type_544 = None
        sum_42: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
        pow_58: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_20, 3);  rsqrt_20 = None
        mul_311: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
        mul_312: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_311, pow_58);  mul_311 = pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_50: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_312, [4, 512, 2048]);  mul_312 = None
        div_30: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_50, 2048);  expand_50 = None
        pow_59: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_203, 1.0);  convert_element_type_203 = None
        mul_313: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_59, 2.0);  pow_59 = None
        mul_314: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_30, mul_313);  div_30 = mul_313 = None
        add_194: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_310, mul_314);  mul_310 = mul_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_545: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16);  add_194 = None
        add_195: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_187, convert_element_type_545);  add_187 = convert_element_type_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_464: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_195, [2048, 2048])
        permute_374: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_199: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_374, view_202);  permute_374 = view_202 = None
        permute_110: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_376: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_200: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_464, permute_376);  view_464 = permute_376 = None
        view_465: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_200, [4, 512, 8192]);  mm_200 = None
        view_199: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_67, [4, 512, 8192]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_197: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        neg_29: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_197)
        exp_9: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_71: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_197, add_71)
        convert_element_type_198: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_315: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_465, convert_element_type_198);  convert_element_type_198 = None
        view_201: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_68, [4, 512, 8192]);  mm_68 = None
        mul_316: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_465, view_201);  view_465 = view_201 = None
        view_466: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_315, [2048, 8192]);  mul_315 = None
        permute_378: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_466, [1, 0])
        mm_201: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_378, view_198);  permute_378 = None
        permute_109: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_380: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_202: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_466, permute_380);  view_466 = permute_380 = None
        view_467: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_202, [4, 512, 2048]);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_554: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_316, torch.float32);  mul_316 = None
        reciprocal_6: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_71);  add_71 = None
        mul_317: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_318: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_554, mul_317);  convert_element_type_554 = None
        sub_11: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_317);  mul_317 = None
        mul_319: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_197, sub_11);  convert_element_type_197 = sub_11 = None
        add_197: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_319, 1);  mul_319 = None
        mul_320: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_318, add_197);  mul_318 = add_197 = None
        convert_element_type_556: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_320, torch.bfloat16);  mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_468: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_556, [2048, 8192]);  convert_element_type_556 = None
        permute_382: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_468, [1, 0])
        mm_203: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_382, view_198);  permute_382 = view_198 = None
        permute_108: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_384: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_204: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_468, permute_384);  view_468 = permute_384 = None
        view_469: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_204, [4, 512, 2048]);  mm_204 = None
        add_198: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_467, view_469);  view_467 = view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_321: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_198, primals_90);  primals_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_193: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_90: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_193, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_194: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        mul_322: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_198, convert_element_type_194);  add_198 = convert_element_type_194 = None
        sum_43: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1], True);  mul_322 = None
        view_470: "bf16[2048]" = torch.ops.aten.reshape.default(sum_43, [2048]);  sum_43 = None
        convert_element_type_561: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_321, torch.float32);  mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_323: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_561, convert_element_type_193)
        mul_324: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_561, rsqrt_19);  convert_element_type_561 = None
        sum_44: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
        pow_60: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_19, 3);  rsqrt_19 = None
        mul_325: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_44, -0.5);  sum_44 = None
        mul_326: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_325, pow_60);  mul_325 = pow_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_51: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_326, [4, 512, 2048]);  mul_326 = None
        div_31: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_51, 2048);  expand_51 = None
        pow_61: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_193, 1.0);  convert_element_type_193 = None
        mul_327: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_61, 2.0);  pow_61 = None
        mul_328: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_31, mul_327);  div_31 = mul_327 = None
        add_199: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_324, mul_328);  mul_324 = mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_562: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None
        add_200: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_195, convert_element_type_562);  add_195 = convert_element_type_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_471: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_200, [2048, 2048])
        permute_386: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_471, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_195: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_106, [4, 512, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_196: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_195, [2048, 2048]);  view_195 = None
        mm_205: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_386, view_196);  permute_386 = view_196 = None
        permute_107: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_388: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_206: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_471, permute_388);  view_471 = permute_388 = None
        view_472: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_206, [4, 512, 2048]);  mm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_473: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_472, [4, 512, 32, 64]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_390: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_6 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_390, add_67, view_193, view_194, getitem_81, getitem_82, getitem_87, getitem_88, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_390 = add_67 = view_193 = view_194 = getitem_81 = getitem_82 = getitem_87 = getitem_88 = None
        getitem_162: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_6[0]
        getitem_163: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_6[1]
        getitem_164: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_6[2];  _scaled_dot_product_cudnn_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_474: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_164, [4, 8, 4, 512, 64]);  getitem_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_45: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_474, [2], True);  view_474 = None
        squeeze_13: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_45, 2);  sum_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_475: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_163, [4, 8, 4, 512, 64]);  getitem_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_46: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_475, [2], True);  view_475 = None
        squeeze_14: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_46, 2);  sum_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_329: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_14, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_93: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_329, 3, 0, 32)
        slice_94: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_329, 3, 32, 64);  mul_329 = None
        neg_68: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_93);  slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_24: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_68, 3, 32, 9223372036854775807);  neg_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_25: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_94, 3, 0, 32);  slice_94 = None
        add_201: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_25);  slice_scatter_24 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_330: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_14, unsqueeze_14);  squeeze_14 = None
        add_202: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_201, mul_330);  add_201 = mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_331: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_162, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_95: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_331, 3, 0, 32)
        slice_96: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_331, 3, 32, 64);  mul_331 = None
        neg_69: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_95);  slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_26: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_69, 3, 32, 9223372036854775807);  neg_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_27: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_96, 3, 0, 32);  slice_96 = None
        add_203: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_27);  slice_scatter_26 = slice_scatter_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_332: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_162, unsqueeze_14);  getitem_162 = None
        add_204: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_203, mul_332);  add_203 = mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_391: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_13, [0, 2, 1, 3]);  squeeze_13 = None
        clone_53: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_476: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_53, [4, 512, 512]);  clone_53 = None
        view_477: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_476, [2048, 512]);  view_476 = None
        permute_392: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_477, [1, 0])
        mm_207: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_392, view_184);  permute_392 = None
        permute_104: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_394: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
        mm_208: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_477, permute_394);  view_477 = permute_394 = None
        view_478: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_208, [4, 512, 2048]);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_396: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_202, [0, 2, 1, 3]);  add_202 = None
        clone_54: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_396, memory_format = torch.contiguous_format);  permute_396 = None
        view_479: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_54, [4, 512, 512]);  clone_54 = None
        view_480: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_479, [2048, 512]);  view_479 = None
        permute_397: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_480, [1, 0])
        mm_209: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_397, view_184);  permute_397 = None
        permute_102: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_399: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        mm_210: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_480, permute_399);  view_480 = permute_399 = None
        view_481: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_210, [4, 512, 2048]);  mm_210 = None
        add_205: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_478, view_481);  view_478 = view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_401: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_204, [0, 2, 1, 3]);  add_204 = None
        clone_55: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_482: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_55, [4, 512, 2048]);  clone_55 = None
        view_483: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_482, [2048, 2048]);  view_482 = None
        permute_402: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_483, [1, 0])
        mm_211: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_402, view_184);  permute_402 = view_184 = None
        permute_100: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_404: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_212: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_483, permute_404);  view_483 = permute_404 = None
        view_484: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_212, [4, 512, 2048]);  mm_212 = None
        add_206: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_205, view_484);  add_205 = view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_333: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_206, primals_85);  primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_183: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_84: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_183, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_184: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None
        mul_334: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_206, convert_element_type_184);  add_206 = convert_element_type_184 = None
        sum_47: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1], True);  mul_334 = None
        view_485: "bf16[2048]" = torch.ops.aten.reshape.default(sum_47, [2048]);  sum_47 = None
        convert_element_type_579: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_333, torch.float32);  mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_335: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_579, convert_element_type_183)
        mul_336: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_579, rsqrt_18);  convert_element_type_579 = None
        sum_48: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
        pow_62: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
        mul_337: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
        mul_338: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_337, pow_62);  mul_337 = pow_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_52: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_338, [4, 512, 2048]);  mul_338 = None
        div_32: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_52, 2048);  expand_52 = None
        pow_63: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 1.0);  convert_element_type_183 = None
        mul_339: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_63, 2.0);  pow_63 = None
        mul_340: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_32, mul_339);  div_32 = mul_339 = None
        add_207: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_336, mul_340);  mul_336 = mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_580: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_207, torch.bfloat16);  add_207 = None
        add_208: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_200, convert_element_type_580);  add_200 = convert_element_type_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_486: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_208, [2048, 2048])
        permute_406: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_486, [1, 0])
        mm_213: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_406, view_182);  permute_406 = view_182 = None
        permute_99: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_408: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_214: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_486, permute_408);  view_486 = permute_408 = None
        view_487: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_214, [4, 512, 8192]);  mm_214 = None
        view_179: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 8192]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_177: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        neg_26: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_177)
        exp_8: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_64: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_177, add_64)
        convert_element_type_178: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_341: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_487, convert_element_type_178);  convert_element_type_178 = None
        view_181: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_61, [4, 512, 8192]);  mm_61 = None
        mul_342: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_487, view_181);  view_487 = view_181 = None
        view_488: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_341, [2048, 8192]);  mul_341 = None
        permute_410: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_488, [1, 0])
        mm_215: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_410, view_178);  permute_410 = None
        permute_98: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_412: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_216: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_488, permute_412);  view_488 = permute_412 = None
        view_489: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_216, [4, 512, 2048]);  mm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_589: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_342, torch.float32);  mul_342 = None
        reciprocal_7: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_64);  add_64 = None
        mul_343: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_344: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_589, mul_343);  convert_element_type_589 = None
        sub_12: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_343);  mul_343 = None
        mul_345: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_177, sub_12);  convert_element_type_177 = sub_12 = None
        add_210: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_345, 1);  mul_345 = None
        mul_346: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_344, add_210);  mul_344 = add_210 = None
        convert_element_type_591: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_490: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_591, [2048, 8192]);  convert_element_type_591 = None
        permute_414: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_217: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_414, view_178);  permute_414 = view_178 = None
        permute_97: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_416: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_218: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_490, permute_416);  view_490 = permute_416 = None
        view_491: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_218, [4, 512, 2048]);  mm_218 = None
        add_211: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_489, view_491);  view_489 = view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_347: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_211, primals_81);  primals_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_173: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_62, torch.float32);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_81: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_173, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_174: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        mul_348: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_211, convert_element_type_174);  add_211 = convert_element_type_174 = None
        sum_49: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 1], True);  mul_348 = None
        view_492: "bf16[2048]" = torch.ops.aten.reshape.default(sum_49, [2048]);  sum_49 = None
        convert_element_type_596: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_347, torch.float32);  mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_349: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_596, convert_element_type_173)
        mul_350: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_596, rsqrt_17);  convert_element_type_596 = None
        sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
        pow_64: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
        mul_351: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_50, -0.5);  sum_50 = None
        mul_352: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_351, pow_64);  mul_351 = pow_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_53: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_352, [4, 512, 2048]);  mul_352 = None
        div_33: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_53, 2048);  expand_53 = None
        pow_65: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_173, 1.0);  convert_element_type_173 = None
        mul_353: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_65, 2.0);  pow_65 = None
        mul_354: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_33, mul_353);  div_33 = mul_353 = None
        add_212: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_350, mul_354);  mul_350 = mul_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_597: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_212, torch.bfloat16);  add_212 = None
        add_213: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_208, convert_element_type_597);  add_208 = convert_element_type_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_493: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_213, [2048, 2048])
        permute_418: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_493, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_175: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_95, [4, 512, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_176: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_175, [2048, 2048]);  view_175 = None
        mm_219: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_418, view_176);  permute_418 = view_176 = None
        permute_96: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_420: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_220: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_493, permute_420);  view_493 = permute_420 = None
        view_494: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_220, [4, 512, 2048]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_495: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_494, [4, 512, 32, 64]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_422: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_7 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_422, add_60, view_173, view_174, getitem_72, getitem_73, getitem_78, getitem_79, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_422 = add_60 = view_173 = view_174 = getitem_72 = getitem_73 = getitem_78 = getitem_79 = None
        getitem_165: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_7[0]
        getitem_166: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_7[1]
        getitem_167: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_7[2];  _scaled_dot_product_cudnn_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_496: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_167, [4, 8, 4, 512, 64]);  getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_51: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_496, [2], True);  view_496 = None
        squeeze_15: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_51, 2);  sum_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_497: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_166, [4, 8, 4, 512, 64]);  getitem_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_52: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_497, [2], True);  view_497 = None
        squeeze_16: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_52, 2);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_355: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_16, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_97: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_355, 3, 0, 32)
        slice_98: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_355, 3, 32, 64);  mul_355 = None
        neg_71: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_97);  slice_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_28: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_71, 3, 32, 9223372036854775807);  neg_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_29: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_98, 3, 0, 32);  slice_98 = None
        add_214: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_29);  slice_scatter_28 = slice_scatter_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_356: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_16, unsqueeze_14);  squeeze_16 = None
        add_215: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_214, mul_356);  add_214 = mul_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_357: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_165, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_99: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_357, 3, 0, 32)
        slice_100: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_357, 3, 32, 64);  mul_357 = None
        neg_72: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_99);  slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_30: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_72, 3, 32, 9223372036854775807);  neg_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_31: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_100, 3, 0, 32);  slice_100 = None
        add_216: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_30, slice_scatter_31);  slice_scatter_30 = slice_scatter_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_358: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_165, unsqueeze_14);  getitem_165 = None
        add_217: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_216, mul_358);  add_216 = mul_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_423: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_15, [0, 2, 1, 3]);  squeeze_15 = None
        clone_56: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
        view_498: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_56, [4, 512, 512]);  clone_56 = None
        view_499: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_498, [2048, 512]);  view_498 = None
        permute_424: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_499, [1, 0])
        mm_221: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_424, view_164);  permute_424 = None
        permute_93: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_426: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        mm_222: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_499, permute_426);  view_499 = permute_426 = None
        view_500: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_222, [4, 512, 2048]);  mm_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_428: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_215, [0, 2, 1, 3]);  add_215 = None
        clone_57: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
        view_501: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_57, [4, 512, 512]);  clone_57 = None
        view_502: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_501, [2048, 512]);  view_501 = None
        permute_429: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_502, [1, 0])
        mm_223: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_429, view_164);  permute_429 = None
        permute_91: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_431: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        mm_224: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_502, permute_431);  view_502 = permute_431 = None
        view_503: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_224, [4, 512, 2048]);  mm_224 = None
        add_218: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_500, view_503);  view_500 = view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_433: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_217, [0, 2, 1, 3]);  add_217 = None
        clone_58: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        view_504: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_58, [4, 512, 2048]);  clone_58 = None
        view_505: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_504, [2048, 2048]);  view_504 = None
        permute_434: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_505, [1, 0])
        mm_225: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_434, view_164);  permute_434 = view_164 = None
        permute_89: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_436: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_226: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_505, permute_436);  view_505 = permute_436 = None
        view_506: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_226, [4, 512, 2048]);  mm_226 = None
        add_219: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_218, view_506);  add_218 = view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_359: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_219, primals_76);  primals_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_163: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_75: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_164: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        mul_360: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_219, convert_element_type_164);  add_219 = convert_element_type_164 = None
        sum_53: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1], True);  mul_360 = None
        view_507: "bf16[2048]" = torch.ops.aten.reshape.default(sum_53, [2048]);  sum_53 = None
        convert_element_type_614: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_359, torch.float32);  mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_361: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_614, convert_element_type_163)
        mul_362: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_614, rsqrt_16);  convert_element_type_614 = None
        sum_54: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True);  mul_361 = None
        pow_66: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_363: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_54, -0.5);  sum_54 = None
        mul_364: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_363, pow_66);  mul_363 = pow_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_54: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_364, [4, 512, 2048]);  mul_364 = None
        div_34: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_54, 2048);  expand_54 = None
        pow_67: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_163, 1.0);  convert_element_type_163 = None
        mul_365: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_67, 2.0);  pow_67 = None
        mul_366: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_34, mul_365);  div_34 = mul_365 = None
        add_220: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_362, mul_366);  mul_362 = mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_615: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16);  add_220 = None
        add_221: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_213, convert_element_type_615);  add_213 = convert_element_type_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_508: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_221, [2048, 2048])
        permute_438: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_508, [1, 0])
        mm_227: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_438, view_162);  permute_438 = view_162 = None
        permute_88: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_440: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_228: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_508, permute_440);  view_508 = permute_440 = None
        view_509: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_228, [4, 512, 8192]);  mm_228 = None
        view_159: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_53, [4, 512, 8192]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_157: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        neg_23: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_157)
        exp_7: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_57: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_157, add_57)
        convert_element_type_158: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_367: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_509, convert_element_type_158);  convert_element_type_158 = None
        view_161: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_54, [4, 512, 8192]);  mm_54 = None
        mul_368: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_509, view_161);  view_509 = view_161 = None
        view_510: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_367, [2048, 8192]);  mul_367 = None
        permute_442: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_510, [1, 0])
        mm_229: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_442, view_158);  permute_442 = None
        permute_87: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_444: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_230: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_510, permute_444);  view_510 = permute_444 = None
        view_511: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_230, [4, 512, 2048]);  mm_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_624: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_368, torch.float32);  mul_368 = None
        reciprocal_8: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_57);  add_57 = None
        mul_369: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        mul_370: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_624, mul_369);  convert_element_type_624 = None
        sub_13: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_369);  mul_369 = None
        mul_371: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_157, sub_13);  convert_element_type_157 = sub_13 = None
        add_223: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_371, 1);  mul_371 = None
        mul_372: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_370, add_223);  mul_370 = add_223 = None
        convert_element_type_626: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_372, torch.bfloat16);  mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_512: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_626, [2048, 8192]);  convert_element_type_626 = None
        permute_446: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_512, [1, 0])
        mm_231: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_446, view_158);  permute_446 = view_158 = None
        permute_86: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_448: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_232: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_512, permute_448);  view_512 = permute_448 = None
        view_513: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_232, [4, 512, 2048]);  mm_232 = None
        add_224: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_511, view_513);  view_511 = view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_373: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_224, primals_72);  primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_153: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_72: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_154: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        mul_374: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_224, convert_element_type_154);  add_224 = convert_element_type_154 = None
        sum_55: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1], True);  mul_374 = None
        view_514: "bf16[2048]" = torch.ops.aten.reshape.default(sum_55, [2048]);  sum_55 = None
        convert_element_type_631: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_373, torch.float32);  mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_375: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_631, convert_element_type_153)
        mul_376: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_631, rsqrt_15);  convert_element_type_631 = None
        sum_56: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
        pow_68: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_377: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
        mul_378: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_377, pow_68);  mul_377 = pow_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_55: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_378, [4, 512, 2048]);  mul_378 = None
        div_35: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_55, 2048);  expand_55 = None
        pow_69: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 1.0);  convert_element_type_153 = None
        mul_379: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_69, 2.0);  pow_69 = None
        mul_380: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_35, mul_379);  div_35 = mul_379 = None
        add_225: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_632: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_225, torch.bfloat16);  add_225 = None
        add_226: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_221, convert_element_type_632);  add_221 = convert_element_type_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_515: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_226, [2048, 2048])
        permute_450: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_515, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_84, [4, 512, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_156: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_155, [2048, 2048]);  view_155 = None
        mm_233: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_450, view_156);  permute_450 = view_156 = None
        permute_85: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_452: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_234: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_515, permute_452);  view_515 = permute_452 = None
        view_516: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_234, [4, 512, 2048]);  mm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_517: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_516, [4, 512, 32, 64]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_454: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_8 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_454, add_53, view_153, view_154, getitem_63, getitem_64, getitem_69, getitem_70, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_454 = add_53 = view_153 = view_154 = getitem_63 = getitem_64 = getitem_69 = getitem_70 = None
        getitem_168: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_8[0]
        getitem_169: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_8[1]
        getitem_170: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_8[2];  _scaled_dot_product_cudnn_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_518: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_170, [4, 8, 4, 512, 64]);  getitem_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_57: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_518, [2], True);  view_518 = None
        squeeze_17: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_57, 2);  sum_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_519: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_169, [4, 8, 4, 512, 64]);  getitem_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_58: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_519, [2], True);  view_519 = None
        squeeze_18: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_58, 2);  sum_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_381: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_18, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_101: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_381, 3, 0, 32)
        slice_102: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_381, 3, 32, 64);  mul_381 = None
        neg_74: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_101);  slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_32: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_74, 3, 32, 9223372036854775807);  neg_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_33: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_102, 3, 0, 32);  slice_102 = None
        add_227: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_33);  slice_scatter_32 = slice_scatter_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_382: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_18, unsqueeze_14);  squeeze_18 = None
        add_228: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_227, mul_382);  add_227 = mul_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_383: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_168, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_103: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_383, 3, 0, 32)
        slice_104: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_383, 3, 32, 64);  mul_383 = None
        neg_75: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_103);  slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_34: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_75, 3, 32, 9223372036854775807);  neg_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_35: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_104, 3, 0, 32);  slice_104 = None
        add_229: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_35);  slice_scatter_34 = slice_scatter_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_384: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_168, unsqueeze_14);  getitem_168 = None
        add_230: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_229, mul_384);  add_229 = mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_455: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_17, [0, 2, 1, 3]);  squeeze_17 = None
        clone_59: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        view_520: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_59, [4, 512, 512]);  clone_59 = None
        view_521: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_520, [2048, 512]);  view_520 = None
        permute_456: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_521, [1, 0])
        mm_235: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_456, view_144);  permute_456 = None
        permute_82: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_458: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        mm_236: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_521, permute_458);  view_521 = permute_458 = None
        view_522: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_236, [4, 512, 2048]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_460: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_228, [0, 2, 1, 3]);  add_228 = None
        clone_60: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
        view_523: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_60, [4, 512, 512]);  clone_60 = None
        view_524: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_523, [2048, 512]);  view_523 = None
        permute_461: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_524, [1, 0])
        mm_237: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_461, view_144);  permute_461 = None
        permute_80: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_463: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        mm_238: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_524, permute_463);  view_524 = permute_463 = None
        view_525: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_238, [4, 512, 2048]);  mm_238 = None
        add_231: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_522, view_525);  view_522 = view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_465: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_230, [0, 2, 1, 3]);  add_230 = None
        clone_61: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
        view_526: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_61, [4, 512, 2048]);  clone_61 = None
        view_527: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_526, [2048, 2048]);  view_526 = None
        permute_466: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_527, [1, 0])
        mm_239: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_466, view_144);  permute_466 = view_144 = None
        permute_78: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_468: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_240: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_527, permute_468);  view_527 = permute_468 = None
        view_528: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_240, [4, 512, 2048]);  mm_240 = None
        add_232: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_231, view_528);  add_231 = view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_385: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_232, primals_67);  primals_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_143: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_66: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_143, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_144: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        mul_386: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_232, convert_element_type_144);  add_232 = convert_element_type_144 = None
        sum_59: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1], True);  mul_386 = None
        view_529: "bf16[2048]" = torch.ops.aten.reshape.default(sum_59, [2048]);  sum_59 = None
        convert_element_type_649: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_385, torch.float32);  mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_387: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_649, convert_element_type_143)
        mul_388: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_649, rsqrt_14);  convert_element_type_649 = None
        sum_60: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True);  mul_387 = None
        pow_70: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_389: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_60, -0.5);  sum_60 = None
        mul_390: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_389, pow_70);  mul_389 = pow_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_56: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_390, [4, 512, 2048]);  mul_390 = None
        div_36: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_56, 2048);  expand_56 = None
        pow_71: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_143, 1.0);  convert_element_type_143 = None
        mul_391: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_71, 2.0);  pow_71 = None
        mul_392: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_36, mul_391);  div_36 = mul_391 = None
        add_233: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_388, mul_392);  mul_388 = mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_650: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_233, torch.bfloat16);  add_233 = None
        add_234: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_226, convert_element_type_650);  add_226 = convert_element_type_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_530: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_234, [2048, 2048])
        permute_470: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_530, [1, 0])
        mm_241: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_470, view_142);  permute_470 = view_142 = None
        permute_77: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_472: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_242: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_530, permute_472);  view_530 = permute_472 = None
        view_531: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_242, [4, 512, 8192]);  mm_242 = None
        view_139: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 8192]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_137: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        neg_20: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_137)
        exp_6: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_50: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_137, add_50)
        convert_element_type_138: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_393: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_531, convert_element_type_138);  convert_element_type_138 = None
        view_141: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_47, [4, 512, 8192]);  mm_47 = None
        mul_394: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_531, view_141);  view_531 = view_141 = None
        view_532: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_393, [2048, 8192]);  mul_393 = None
        permute_474: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_243: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_474, view_138);  permute_474 = None
        permute_76: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_476: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_244: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_532, permute_476);  view_532 = permute_476 = None
        view_533: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_244, [4, 512, 2048]);  mm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_659: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_394, torch.float32);  mul_394 = None
        reciprocal_9: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_50);  add_50 = None
        mul_395: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        mul_396: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_659, mul_395);  convert_element_type_659 = None
        sub_14: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_395);  mul_395 = None
        mul_397: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_137, sub_14);  convert_element_type_137 = sub_14 = None
        add_236: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_397, 1);  mul_397 = None
        mul_398: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_396, add_236);  mul_396 = add_236 = None
        convert_element_type_661: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_398, torch.bfloat16);  mul_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_534: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_661, [2048, 8192]);  convert_element_type_661 = None
        permute_478: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_245: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_478, view_138);  permute_478 = view_138 = None
        permute_75: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_480: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_246: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_534, permute_480);  view_534 = permute_480 = None
        view_535: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_246, [4, 512, 2048]);  mm_246 = None
        add_237: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_533, view_535);  view_533 = view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_399: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_237, primals_63);  primals_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_133: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_63: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_134: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        mul_400: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_237, convert_element_type_134);  add_237 = convert_element_type_134 = None
        sum_61: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1], True);  mul_400 = None
        view_536: "bf16[2048]" = torch.ops.aten.reshape.default(sum_61, [2048]);  sum_61 = None
        convert_element_type_666: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_399, torch.float32);  mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_401: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_666, convert_element_type_133)
        mul_402: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_666, rsqrt_13);  convert_element_type_666 = None
        sum_62: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True);  mul_401 = None
        pow_72: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_403: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_62, -0.5);  sum_62 = None
        mul_404: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_403, pow_72);  mul_403 = pow_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_57: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_404, [4, 512, 2048]);  mul_404 = None
        div_37: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_57, 2048);  expand_57 = None
        pow_73: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 1.0);  convert_element_type_133 = None
        mul_405: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_73, 2.0);  pow_73 = None
        mul_406: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_37, mul_405);  div_37 = mul_405 = None
        add_238: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_402, mul_406);  mul_402 = mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_667: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16);  add_238 = None
        add_239: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_234, convert_element_type_667);  add_234 = convert_element_type_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_537: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_239, [2048, 2048])
        permute_482: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_537, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_135: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_73, [4, 512, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_136: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_135, [2048, 2048]);  view_135 = None
        mm_247: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_482, view_136);  permute_482 = view_136 = None
        permute_74: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_484: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_248: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_537, permute_484);  view_537 = permute_484 = None
        view_538: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_248, [4, 512, 2048]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_539: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_538, [4, 512, 32, 64]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_486: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_9 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_486, add_46, view_133, view_134, getitem_54, getitem_55, getitem_60, getitem_61, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_486 = add_46 = view_133 = view_134 = getitem_54 = getitem_55 = getitem_60 = getitem_61 = None
        getitem_171: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_9[0]
        getitem_172: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_9[1]
        getitem_173: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_9[2];  _scaled_dot_product_cudnn_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_540: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_173, [4, 8, 4, 512, 64]);  getitem_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_63: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_540, [2], True);  view_540 = None
        squeeze_19: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_63, 2);  sum_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_541: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_172, [4, 8, 4, 512, 64]);  getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_64: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_541, [2], True);  view_541 = None
        squeeze_20: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_64, 2);  sum_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_407: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_20, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_105: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_407, 3, 0, 32)
        slice_106: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_407, 3, 32, 64);  mul_407 = None
        neg_77: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_105);  slice_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_36: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_77, 3, 32, 9223372036854775807);  neg_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_37: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_106, 3, 0, 32);  slice_106 = None
        add_240: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_37);  slice_scatter_36 = slice_scatter_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_408: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_20, unsqueeze_14);  squeeze_20 = None
        add_241: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_240, mul_408);  add_240 = mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_409: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_171, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_107: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_409, 3, 0, 32)
        slice_108: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_409, 3, 32, 64);  mul_409 = None
        neg_78: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_107);  slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_38: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_78, 3, 32, 9223372036854775807);  neg_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_39: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_108, 3, 0, 32);  slice_108 = None
        add_242: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_38, slice_scatter_39);  slice_scatter_38 = slice_scatter_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_410: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_171, unsqueeze_14);  getitem_171 = None
        add_243: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_242, mul_410);  add_242 = mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_487: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_19, [0, 2, 1, 3]);  squeeze_19 = None
        clone_62: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
        view_542: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_62, [4, 512, 512]);  clone_62 = None
        view_543: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_542, [2048, 512]);  view_542 = None
        permute_488: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_543, [1, 0])
        mm_249: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_488, view_124);  permute_488 = None
        permute_71: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_490: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_250: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_543, permute_490);  view_543 = permute_490 = None
        view_544: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_250, [4, 512, 2048]);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_492: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_241, [0, 2, 1, 3]);  add_241 = None
        clone_63: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_492, memory_format = torch.contiguous_format);  permute_492 = None
        view_545: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_63, [4, 512, 512]);  clone_63 = None
        view_546: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_545, [2048, 512]);  view_545 = None
        permute_493: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_251: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_493, view_124);  permute_493 = None
        permute_69: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_495: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_252: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_546, permute_495);  view_546 = permute_495 = None
        view_547: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_252, [4, 512, 2048]);  mm_252 = None
        add_244: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_544, view_547);  view_544 = view_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_497: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_243, [0, 2, 1, 3]);  add_243 = None
        clone_64: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_497, memory_format = torch.contiguous_format);  permute_497 = None
        view_548: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_64, [4, 512, 2048]);  clone_64 = None
        view_549: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_548, [2048, 2048]);  view_548 = None
        permute_498: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_549, [1, 0])
        mm_253: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_498, view_124);  permute_498 = view_124 = None
        permute_67: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_500: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_254: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_549, permute_500);  view_549 = permute_500 = None
        view_550: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_254, [4, 512, 2048]);  mm_254 = None
        add_245: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_244, view_550);  add_244 = view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_411: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_245, primals_58);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_123: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_57: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_124: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        mul_412: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_245, convert_element_type_124);  add_245 = convert_element_type_124 = None
        sum_65: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 1], True);  mul_412 = None
        view_551: "bf16[2048]" = torch.ops.aten.reshape.default(sum_65, [2048]);  sum_65 = None
        convert_element_type_684: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_411, torch.float32);  mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_413: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_684, convert_element_type_123)
        mul_414: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_684, rsqrt_12);  convert_element_type_684 = None
        sum_66: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
        pow_74: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_415: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
        mul_416: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_415, pow_74);  mul_415 = pow_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_58: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_416, [4, 512, 2048]);  mul_416 = None
        div_38: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_58, 2048);  expand_58 = None
        pow_75: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 1.0);  convert_element_type_123 = None
        mul_417: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_75, 2.0);  pow_75 = None
        mul_418: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_38, mul_417);  div_38 = mul_417 = None
        add_246: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_414, mul_418);  mul_414 = mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_685: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16);  add_246 = None
        add_247: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_239, convert_element_type_685);  add_239 = convert_element_type_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_552: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_247, [2048, 2048])
        permute_502: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_552, [1, 0])
        mm_255: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_502, view_122);  permute_502 = view_122 = None
        permute_66: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_504: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_256: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_552, permute_504);  view_552 = permute_504 = None
        view_553: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_256, [4, 512, 8192]);  mm_256 = None
        view_119: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_39, [4, 512, 8192]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_117: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        neg_17: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_117)
        exp_5: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_43: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_117, add_43)
        convert_element_type_118: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_419: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_553, convert_element_type_118);  convert_element_type_118 = None
        view_121: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 8192]);  mm_40 = None
        mul_420: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_553, view_121);  view_553 = view_121 = None
        view_554: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_419, [2048, 8192]);  mul_419 = None
        permute_506: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_257: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_506, view_118);  permute_506 = None
        permute_65: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_508: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_258: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_554, permute_508);  view_554 = permute_508 = None
        view_555: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_258, [4, 512, 2048]);  mm_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_694: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_420, torch.float32);  mul_420 = None
        reciprocal_10: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_43);  add_43 = None
        mul_421: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        mul_422: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_694, mul_421);  convert_element_type_694 = None
        sub_15: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_421);  mul_421 = None
        mul_423: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_117, sub_15);  convert_element_type_117 = sub_15 = None
        add_249: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_423, 1);  mul_423 = None
        mul_424: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_422, add_249);  mul_422 = add_249 = None
        convert_element_type_696: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_424, torch.bfloat16);  mul_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_556: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_696, [2048, 8192]);  convert_element_type_696 = None
        permute_510: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_259: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_510, view_118);  permute_510 = view_118 = None
        permute_64: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_512: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_260: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_556, permute_512);  view_556 = permute_512 = None
        view_557: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_260, [4, 512, 2048]);  mm_260 = None
        add_250: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_555, view_557);  view_555 = view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_425: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_250, primals_54);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_113: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_54: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_114: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_426: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_250, convert_element_type_114);  add_250 = convert_element_type_114 = None
        sum_67: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 1], True);  mul_426 = None
        view_558: "bf16[2048]" = torch.ops.aten.reshape.default(sum_67, [2048]);  sum_67 = None
        convert_element_type_701: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_425, torch.float32);  mul_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_427: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_701, convert_element_type_113)
        mul_428: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_701, rsqrt_11);  convert_element_type_701 = None
        sum_68: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
        pow_76: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
        mul_429: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_68, -0.5);  sum_68 = None
        mul_430: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_429, pow_76);  mul_429 = pow_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_59: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_430, [4, 512, 2048]);  mul_430 = None
        div_39: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_59, 2048);  expand_59 = None
        pow_77: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 1.0);  convert_element_type_113 = None
        mul_431: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_77, 2.0);  pow_77 = None
        mul_432: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_39, mul_431);  div_39 = mul_431 = None
        add_251: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_428, mul_432);  mul_428 = mul_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_702: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_251, torch.bfloat16);  add_251 = None
        add_252: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_247, convert_element_type_702);  add_247 = convert_element_type_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_559: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_252, [2048, 2048])
        permute_514: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_559, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_115: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_62, [4, 512, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_116: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_115, [2048, 2048]);  view_115 = None
        mm_261: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_514, view_116);  permute_514 = view_116 = None
        permute_63: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_516: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_262: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_559, permute_516);  view_559 = permute_516 = None
        view_560: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_262, [4, 512, 2048]);  mm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_561: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_560, [4, 512, 32, 64]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_518: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_10 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_518, add_39, view_113, view_114, getitem_45, getitem_46, getitem_51, getitem_52, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_518 = add_39 = view_113 = view_114 = getitem_45 = getitem_46 = getitem_51 = getitem_52 = None
        getitem_174: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_10[0]
        getitem_175: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_10[1]
        getitem_176: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_10[2];  _scaled_dot_product_cudnn_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_562: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_176, [4, 8, 4, 512, 64]);  getitem_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_69: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_562, [2], True);  view_562 = None
        squeeze_21: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_69, 2);  sum_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_563: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_175, [4, 8, 4, 512, 64]);  getitem_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_70: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_563, [2], True);  view_563 = None
        squeeze_22: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_70, 2);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_433: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_22, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_109: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_433, 3, 0, 32)
        slice_110: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_433, 3, 32, 64);  mul_433 = None
        neg_80: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_109);  slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_40: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_80, 3, 32, 9223372036854775807);  neg_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_41: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_110, 3, 0, 32);  slice_110 = None
        add_253: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_41);  slice_scatter_40 = slice_scatter_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_434: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_22, unsqueeze_14);  squeeze_22 = None
        add_254: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_253, mul_434);  add_253 = mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_435: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_174, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_111: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_435, 3, 0, 32)
        slice_112: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_435, 3, 32, 64);  mul_435 = None
        neg_81: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_111);  slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_42: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_81, 3, 32, 9223372036854775807);  neg_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_43: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_112, 3, 0, 32);  slice_112 = None
        add_255: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_43);  slice_scatter_42 = slice_scatter_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_436: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_174, unsqueeze_14);  getitem_174 = None
        add_256: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_255, mul_436);  add_255 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_519: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_21, [0, 2, 1, 3]);  squeeze_21 = None
        clone_65: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
        view_564: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_65, [4, 512, 512]);  clone_65 = None
        view_565: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_564, [2048, 512]);  view_564 = None
        permute_520: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_263: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_520, view_104);  permute_520 = None
        permute_60: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_522: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_264: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_565, permute_522);  view_565 = permute_522 = None
        view_566: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_264, [4, 512, 2048]);  mm_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_524: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_254, [0, 2, 1, 3]);  add_254 = None
        clone_66: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
        view_567: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_66, [4, 512, 512]);  clone_66 = None
        view_568: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_567, [2048, 512]);  view_567 = None
        permute_525: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_265: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_525, view_104);  permute_525 = None
        permute_58: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_527: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_266: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_568, permute_527);  view_568 = permute_527 = None
        view_569: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_266, [4, 512, 2048]);  mm_266 = None
        add_257: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_566, view_569);  view_566 = view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_529: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_256, [0, 2, 1, 3]);  add_256 = None
        clone_67: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_529, memory_format = torch.contiguous_format);  permute_529 = None
        view_570: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_67, [4, 512, 2048]);  clone_67 = None
        view_571: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_570, [2048, 2048]);  view_570 = None
        permute_530: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_571, [1, 0])
        mm_267: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_530, view_104);  permute_530 = view_104 = None
        permute_56: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_532: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_268: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_571, permute_532);  view_571 = permute_532 = None
        view_572: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_268, [4, 512, 2048]);  mm_268 = None
        add_258: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_257, view_572);  add_257 = view_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_437: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_258, primals_49);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_103: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_48: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_104: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        mul_438: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_258, convert_element_type_104);  add_258 = convert_element_type_104 = None
        sum_71: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 1], True);  mul_438 = None
        view_573: "bf16[2048]" = torch.ops.aten.reshape.default(sum_71, [2048]);  sum_71 = None
        convert_element_type_719: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_437, torch.float32);  mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_439: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_719, convert_element_type_103)
        mul_440: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_719, rsqrt_10);  convert_element_type_719 = None
        sum_72: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
        pow_78: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
        mul_441: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_72, -0.5);  sum_72 = None
        mul_442: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_441, pow_78);  mul_441 = pow_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_60: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_442, [4, 512, 2048]);  mul_442 = None
        div_40: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_60, 2048);  expand_60 = None
        pow_79: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 1.0);  convert_element_type_103 = None
        mul_443: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_79, 2.0);  pow_79 = None
        mul_444: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_40, mul_443);  div_40 = mul_443 = None
        add_259: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_440, mul_444);  mul_440 = mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_720: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        add_260: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_252, convert_element_type_720);  add_252 = convert_element_type_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_574: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_260, [2048, 2048])
        permute_534: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_269: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_534, view_102);  permute_534 = view_102 = None
        permute_55: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_536: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_270: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_574, permute_536);  view_574 = permute_536 = None
        view_575: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_270, [4, 512, 8192]);  mm_270 = None
        view_99: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 8192]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_97: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        neg_14: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_97)
        exp_4: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_36: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_97, add_36)
        convert_element_type_98: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_445: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_575, convert_element_type_98);  convert_element_type_98 = None
        view_101: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_33, [4, 512, 8192]);  mm_33 = None
        mul_446: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_575, view_101);  view_575 = view_101 = None
        view_576: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_445, [2048, 8192]);  mul_445 = None
        permute_538: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_271: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_538, view_98);  permute_538 = None
        permute_54: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_540: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_272: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_576, permute_540);  view_576 = permute_540 = None
        view_577: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_272, [4, 512, 2048]);  mm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_729: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_446, torch.float32);  mul_446 = None
        reciprocal_11: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_36);  add_36 = None
        mul_447: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        mul_448: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_729, mul_447);  convert_element_type_729 = None
        sub_16: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_447);  mul_447 = None
        mul_449: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_97, sub_16);  convert_element_type_97 = sub_16 = None
        add_262: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_449, 1);  mul_449 = None
        mul_450: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_448, add_262);  mul_448 = add_262 = None
        convert_element_type_731: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16);  mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_578: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_731, [2048, 8192]);  convert_element_type_731 = None
        permute_542: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_273: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_542, view_98);  permute_542 = view_98 = None
        permute_53: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_544: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_274: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_578, permute_544);  view_578 = permute_544 = None
        view_579: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_274, [4, 512, 2048]);  mm_274 = None
        add_263: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_577, view_579);  view_577 = view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_451: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_263, primals_45);  primals_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_93: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_45: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_94: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_452: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_263, convert_element_type_94);  add_263 = convert_element_type_94 = None
        sum_73: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1], True);  mul_452 = None
        view_580: "bf16[2048]" = torch.ops.aten.reshape.default(sum_73, [2048]);  sum_73 = None
        convert_element_type_736: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_451, torch.float32);  mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_453: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_736, convert_element_type_93)
        mul_454: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_736, rsqrt_9);  convert_element_type_736 = None
        sum_74: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True);  mul_453 = None
        pow_80: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
        mul_455: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_74, -0.5);  sum_74 = None
        mul_456: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_455, pow_80);  mul_455 = pow_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_61: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_456, [4, 512, 2048]);  mul_456 = None
        div_41: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_61, 2048);  expand_61 = None
        pow_81: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 1.0);  convert_element_type_93 = None
        mul_457: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_81, 2.0);  pow_81 = None
        mul_458: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_41, mul_457);  div_41 = mul_457 = None
        add_264: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_454, mul_458);  mul_454 = mul_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_737: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16);  add_264 = None
        add_265: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_260, convert_element_type_737);  add_260 = convert_element_type_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_581: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_265, [2048, 2048])
        permute_546: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_581, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_51, [4, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_96: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_95, [2048, 2048]);  view_95 = None
        mm_275: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_546, view_96);  permute_546 = view_96 = None
        permute_52: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_548: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_276: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_581, permute_548);  view_581 = permute_548 = None
        view_582: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_276, [4, 512, 2048]);  mm_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_583: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_582, [4, 512, 32, 64]);  view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_550: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_583, [0, 2, 1, 3]);  view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_11 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_550, add_32, view_93, view_94, getitem_36, getitem_37, getitem_42, getitem_43, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_550 = add_32 = view_93 = view_94 = getitem_36 = getitem_37 = getitem_42 = getitem_43 = None
        getitem_177: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_11[0]
        getitem_178: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_11[1]
        getitem_179: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_11[2];  _scaled_dot_product_cudnn_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_584: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_179, [4, 8, 4, 512, 64]);  getitem_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_75: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_584, [2], True);  view_584 = None
        squeeze_23: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_75, 2);  sum_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_585: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_178, [4, 8, 4, 512, 64]);  getitem_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_76: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_585, [2], True);  view_585 = None
        squeeze_24: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_76, 2);  sum_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_459: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_24, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_113: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_459, 3, 0, 32)
        slice_114: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_459, 3, 32, 64);  mul_459 = None
        neg_83: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_113);  slice_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_44: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_83, 3, 32, 9223372036854775807);  neg_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_45: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_114, 3, 0, 32);  slice_114 = None
        add_266: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_44, slice_scatter_45);  slice_scatter_44 = slice_scatter_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_460: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_24, unsqueeze_14);  squeeze_24 = None
        add_267: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_266, mul_460);  add_266 = mul_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_461: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_177, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_115: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_461, 3, 0, 32)
        slice_116: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_461, 3, 32, 64);  mul_461 = None
        neg_84: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_115);  slice_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_46: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_84, 3, 32, 9223372036854775807);  neg_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_47: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_116, 3, 0, 32);  slice_116 = None
        add_268: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_46, slice_scatter_47);  slice_scatter_46 = slice_scatter_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_462: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_177, unsqueeze_14);  getitem_177 = None
        add_269: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_268, mul_462);  add_268 = mul_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_551: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_23, [0, 2, 1, 3]);  squeeze_23 = None
        clone_68: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_586: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_68, [4, 512, 512]);  clone_68 = None
        view_587: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_586, [2048, 512]);  view_586 = None
        permute_552: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_277: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_552, view_84);  permute_552 = None
        permute_49: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_554: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_278: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_587, permute_554);  view_587 = permute_554 = None
        view_588: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_278, [4, 512, 2048]);  mm_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_556: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_267, [0, 2, 1, 3]);  add_267 = None
        clone_69: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
        view_589: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_69, [4, 512, 512]);  clone_69 = None
        view_590: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_589, [2048, 512]);  view_589 = None
        permute_557: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_279: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_557, view_84);  permute_557 = None
        permute_47: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_559: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_280: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_590, permute_559);  view_590 = permute_559 = None
        view_591: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_280, [4, 512, 2048]);  mm_280 = None
        add_270: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_588, view_591);  view_588 = view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_561: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_269, [0, 2, 1, 3]);  add_269 = None
        clone_70: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_561, memory_format = torch.contiguous_format);  permute_561 = None
        view_592: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_70, [4, 512, 2048]);  clone_70 = None
        view_593: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_592, [2048, 2048]);  view_592 = None
        permute_562: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_281: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_562, view_84);  permute_562 = view_84 = None
        permute_45: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_564: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_282: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_593, permute_564);  view_593 = permute_564 = None
        view_594: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_282, [4, 512, 2048]);  mm_282 = None
        add_271: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_270, view_594);  add_270 = view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_463: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_271, primals_40);  primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_83: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_39: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_84: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        mul_464: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_271, convert_element_type_84);  add_271 = convert_element_type_84 = None
        sum_77: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1], True);  mul_464 = None
        view_595: "bf16[2048]" = torch.ops.aten.reshape.default(sum_77, [2048]);  sum_77 = None
        convert_element_type_754: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_463, torch.float32);  mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_465: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_754, convert_element_type_83)
        mul_466: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_754, rsqrt_8);  convert_element_type_754 = None
        sum_78: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_465, [2], True);  mul_465 = None
        pow_82: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
        mul_467: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_78, -0.5);  sum_78 = None
        mul_468: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_467, pow_82);  mul_467 = pow_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_62: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_468, [4, 512, 2048]);  mul_468 = None
        div_42: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_62, 2048);  expand_62 = None
        pow_83: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 1.0);  convert_element_type_83 = None
        mul_469: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_83, 2.0);  pow_83 = None
        mul_470: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_42, mul_469);  div_42 = mul_469 = None
        add_272: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_466, mul_470);  mul_466 = mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_755: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_272, torch.bfloat16);  add_272 = None
        add_273: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_265, convert_element_type_755);  add_265 = convert_element_type_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_596: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_273, [2048, 2048])
        permute_566: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_596, [1, 0])
        mm_283: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_566, view_82);  permute_566 = view_82 = None
        permute_44: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_568: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_284: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_596, permute_568);  view_596 = permute_568 = None
        view_597: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_284, [4, 512, 8192]);  mm_284 = None
        view_79: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_25, [4, 512, 8192]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_77: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        neg_11: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_77)
        exp_3: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_29: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_77, add_29)
        convert_element_type_78: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_471: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_597, convert_element_type_78);  convert_element_type_78 = None
        view_81: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 8192]);  mm_26 = None
        mul_472: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_597, view_81);  view_597 = view_81 = None
        view_598: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_471, [2048, 8192]);  mul_471 = None
        permute_570: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_285: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_570, view_78);  permute_570 = None
        permute_43: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_572: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_286: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_598, permute_572);  view_598 = permute_572 = None
        view_599: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_286, [4, 512, 2048]);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_764: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_472, torch.float32);  mul_472 = None
        reciprocal_12: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_29);  add_29 = None
        mul_473: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        mul_474: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_764, mul_473);  convert_element_type_764 = None
        sub_17: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_473);  mul_473 = None
        mul_475: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_77, sub_17);  convert_element_type_77 = sub_17 = None
        add_275: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_475, 1);  mul_475 = None
        mul_476: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_474, add_275);  mul_474 = add_275 = None
        convert_element_type_766: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_476, torch.bfloat16);  mul_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_600: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_766, [2048, 8192]);  convert_element_type_766 = None
        permute_574: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_287: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_574, view_78);  permute_574 = view_78 = None
        permute_42: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_576: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_288: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_600, permute_576);  view_600 = permute_576 = None
        view_601: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_288, [4, 512, 2048]);  mm_288 = None
        add_276: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_599, view_601);  view_599 = view_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_477: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_276, primals_36);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_73: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_36: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_74: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        mul_478: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_276, convert_element_type_74);  add_276 = convert_element_type_74 = None
        sum_79: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 1], True);  mul_478 = None
        view_602: "bf16[2048]" = torch.ops.aten.reshape.default(sum_79, [2048]);  sum_79 = None
        convert_element_type_771: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_477, torch.float32);  mul_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_479: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_771, convert_element_type_73)
        mul_480: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_771, rsqrt_7);  convert_element_type_771 = None
        sum_80: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        pow_84: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
        mul_481: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_80, -0.5);  sum_80 = None
        mul_482: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_481, pow_84);  mul_481 = pow_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_63: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_482, [4, 512, 2048]);  mul_482 = None
        div_43: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_63, 2048);  expand_63 = None
        pow_85: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 1.0);  convert_element_type_73 = None
        mul_483: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_85, 2.0);  pow_85 = None
        mul_484: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_43, mul_483);  div_43 = mul_483 = None
        add_277: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_480, mul_484);  mul_480 = mul_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_772: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_277, torch.bfloat16);  add_277 = None
        add_278: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_273, convert_element_type_772);  add_273 = convert_element_type_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_603: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_278, [2048, 2048])
        permute_578: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_603, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_40, [4, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_75, [2048, 2048]);  view_75 = None
        mm_289: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_578, view_76);  permute_578 = view_76 = None
        permute_41: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_580: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_290: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_603, permute_580);  view_603 = permute_580 = None
        view_604: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_290, [4, 512, 2048]);  mm_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_605: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_604, [4, 512, 32, 64]);  view_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_582: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_12 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_582, add_25, view_73, view_74, getitem_27, getitem_28, getitem_33, getitem_34, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_582 = add_25 = view_73 = view_74 = getitem_27 = getitem_28 = getitem_33 = getitem_34 = None
        getitem_180: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_12[0]
        getitem_181: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_12[1]
        getitem_182: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_12[2];  _scaled_dot_product_cudnn_attention_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_606: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_182, [4, 8, 4, 512, 64]);  getitem_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_81: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_606, [2], True);  view_606 = None
        squeeze_25: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_81, 2);  sum_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_607: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_181, [4, 8, 4, 512, 64]);  getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_82: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_607, [2], True);  view_607 = None
        squeeze_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_82, 2);  sum_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_485: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_26, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_117: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_485, 3, 0, 32)
        slice_118: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_485, 3, 32, 64);  mul_485 = None
        neg_86: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_117);  slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_48: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_86, 3, 32, 9223372036854775807);  neg_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_49: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_118, 3, 0, 32);  slice_118 = None
        add_279: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_48, slice_scatter_49);  slice_scatter_48 = slice_scatter_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_486: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_26, unsqueeze_14);  squeeze_26 = None
        add_280: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_279, mul_486);  add_279 = mul_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_487: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_180, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_119: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_487, 3, 0, 32)
        slice_120: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_487, 3, 32, 64);  mul_487 = None
        neg_87: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_119);  slice_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_50: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_87, 3, 32, 9223372036854775807);  neg_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_51: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_120, 3, 0, 32);  slice_120 = None
        add_281: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_51);  slice_scatter_50 = slice_scatter_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_488: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_180, unsqueeze_14);  getitem_180 = None
        add_282: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_281, mul_488);  add_281 = mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_583: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_25, [0, 2, 1, 3]);  squeeze_25 = None
        clone_71: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_583, memory_format = torch.contiguous_format);  permute_583 = None
        view_608: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_71, [4, 512, 512]);  clone_71 = None
        view_609: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_608, [2048, 512]);  view_608 = None
        permute_584: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_609, [1, 0])
        mm_291: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_584, view_64);  permute_584 = None
        permute_38: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_586: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_292: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_609, permute_586);  view_609 = permute_586 = None
        view_610: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_292, [4, 512, 2048]);  mm_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_588: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_280, [0, 2, 1, 3]);  add_280 = None
        clone_72: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_588, memory_format = torch.contiguous_format);  permute_588 = None
        view_611: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_72, [4, 512, 512]);  clone_72 = None
        view_612: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_611, [2048, 512]);  view_611 = None
        permute_589: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_293: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_589, view_64);  permute_589 = None
        permute_36: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_591: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_294: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_612, permute_591);  view_612 = permute_591 = None
        view_613: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_294, [4, 512, 2048]);  mm_294 = None
        add_283: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_610, view_613);  view_610 = view_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_593: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_282, [0, 2, 1, 3]);  add_282 = None
        clone_73: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_593, memory_format = torch.contiguous_format);  permute_593 = None
        view_614: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_73, [4, 512, 2048]);  clone_73 = None
        view_615: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_614, [2048, 2048]);  view_614 = None
        permute_594: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_295: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_594, view_64);  permute_594 = view_64 = None
        permute_34: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_596: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_296: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_615, permute_596);  view_615 = permute_596 = None
        view_616: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_296, [4, 512, 2048]);  mm_296 = None
        add_284: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_283, view_616);  add_283 = view_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_489: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_284, primals_31);  primals_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_63: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_30: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_64: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        mul_490: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_284, convert_element_type_64);  add_284 = convert_element_type_64 = None
        sum_83: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1], True);  mul_490 = None
        view_617: "bf16[2048]" = torch.ops.aten.reshape.default(sum_83, [2048]);  sum_83 = None
        convert_element_type_789: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_489, torch.float32);  mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_491: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_789, convert_element_type_63)
        mul_492: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_789, rsqrt_6);  convert_element_type_789 = None
        sum_84: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_491, [2], True);  mul_491 = None
        pow_86: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
        mul_493: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_84, -0.5);  sum_84 = None
        mul_494: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_493, pow_86);  mul_493 = pow_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_64: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_494, [4, 512, 2048]);  mul_494 = None
        div_44: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_64, 2048);  expand_64 = None
        pow_87: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 1.0);  convert_element_type_63 = None
        mul_495: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_87, 2.0);  pow_87 = None
        mul_496: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_44, mul_495);  div_44 = mul_495 = None
        add_285: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_492, mul_496);  mul_492 = mul_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_790: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16);  add_285 = None
        add_286: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_278, convert_element_type_790);  add_278 = convert_element_type_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_618: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_286, [2048, 2048])
        permute_598: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_297: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_598, view_62);  permute_598 = view_62 = None
        permute_33: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_600: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_298: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_618, permute_600);  view_618 = permute_600 = None
        view_619: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_298, [4, 512, 8192]);  mm_298 = None
        view_59: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 8192]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_57: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        neg_8: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_57)
        exp_2: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_22: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_57, add_22)
        convert_element_type_58: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_497: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_619, convert_element_type_58);  convert_element_type_58 = None
        view_61: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_19, [4, 512, 8192]);  mm_19 = None
        mul_498: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_619, view_61);  view_619 = view_61 = None
        view_620: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_497, [2048, 8192]);  mul_497 = None
        permute_602: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_620, [1, 0])
        mm_299: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_602, view_58);  permute_602 = None
        permute_32: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_604: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_300: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_620, permute_604);  view_620 = permute_604 = None
        view_621: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_300, [4, 512, 2048]);  mm_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_799: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_498, torch.float32);  mul_498 = None
        reciprocal_13: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_22);  add_22 = None
        mul_499: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        mul_500: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_799, mul_499);  convert_element_type_799 = None
        sub_18: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_499);  mul_499 = None
        mul_501: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_57, sub_18);  convert_element_type_57 = sub_18 = None
        add_288: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_501, 1);  mul_501 = None
        mul_502: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_500, add_288);  mul_500 = add_288 = None
        convert_element_type_801: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_502, torch.bfloat16);  mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_622: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_801, [2048, 8192]);  convert_element_type_801 = None
        permute_606: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_301: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_606, view_58);  permute_606 = view_58 = None
        permute_31: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_608: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_302: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_622, permute_608);  view_622 = permute_608 = None
        view_623: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_302, [4, 512, 2048]);  mm_302 = None
        add_289: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_621, view_623);  view_621 = view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_503: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_289, primals_27);  primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_53: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_27: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_54: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_504: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_289, convert_element_type_54);  add_289 = convert_element_type_54 = None
        sum_85: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1], True);  mul_504 = None
        view_624: "bf16[2048]" = torch.ops.aten.reshape.default(sum_85, [2048]);  sum_85 = None
        convert_element_type_806: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_503, torch.float32);  mul_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_505: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_806, convert_element_type_53)
        mul_506: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_806, rsqrt_5);  convert_element_type_806 = None
        sum_86: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True);  mul_505 = None
        pow_88: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
        mul_507: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_86, -0.5);  sum_86 = None
        mul_508: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_507, pow_88);  mul_507 = pow_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_65: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_508, [4, 512, 2048]);  mul_508 = None
        div_45: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_65, 2048);  expand_65 = None
        pow_89: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 1.0);  convert_element_type_53 = None
        mul_509: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_89, 2.0);  pow_89 = None
        mul_510: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_45, mul_509);  div_45 = mul_509 = None
        add_290: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_506, mul_510);  mul_506 = mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_807: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_290, torch.bfloat16);  add_290 = None
        add_291: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_286, convert_element_type_807);  add_286 = convert_element_type_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_625: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_291, [2048, 2048])
        permute_610: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_625, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_29, [4, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_55, [2048, 2048]);  view_55 = None
        mm_303: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_610, view_56);  permute_610 = view_56 = None
        permute_30: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_612: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_304: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_625, permute_612);  view_625 = permute_612 = None
        view_626: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_304, [4, 512, 2048]);  mm_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_627: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_626, [4, 512, 32, 64]);  view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_614: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_13 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_614, add_18, view_53, view_54, getitem_18, getitem_19, getitem_24, getitem_25, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_614 = add_18 = view_53 = view_54 = getitem_18 = getitem_19 = getitem_24 = getitem_25 = None
        getitem_183: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_13[0]
        getitem_184: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_13[1]
        getitem_185: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_13[2];  _scaled_dot_product_cudnn_attention_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_628: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_185, [4, 8, 4, 512, 64]);  getitem_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_87: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_628, [2], True);  view_628 = None
        squeeze_27: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_87, 2);  sum_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_629: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_184, [4, 8, 4, 512, 64]);  getitem_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_88: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_629, [2], True);  view_629 = None
        squeeze_28: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_88, 2);  sum_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_511: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_28, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_121: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_511, 3, 0, 32)
        slice_122: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_511, 3, 32, 64);  mul_511 = None
        neg_89: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_121);  slice_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_52: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_89, 3, 32, 9223372036854775807);  neg_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_53: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_122, 3, 0, 32);  slice_122 = None
        add_292: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_52, slice_scatter_53);  slice_scatter_52 = slice_scatter_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_512: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_28, unsqueeze_14);  squeeze_28 = None
        add_293: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_292, mul_512);  add_292 = mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_513: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_183, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_123: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_513, 3, 0, 32)
        slice_124: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_513, 3, 32, 64);  mul_513 = None
        neg_90: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_123);  slice_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_54: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_90, 3, 32, 9223372036854775807);  neg_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_55: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_124, 3, 0, 32);  slice_124 = None
        add_294: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_55);  slice_scatter_54 = slice_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_514: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_183, unsqueeze_14);  getitem_183 = None
        add_295: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_294, mul_514);  add_294 = mul_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_615: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_27, [0, 2, 1, 3]);  squeeze_27 = None
        clone_74: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
        view_630: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_74, [4, 512, 512]);  clone_74 = None
        view_631: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_630, [2048, 512]);  view_630 = None
        permute_616: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_305: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_616, view_44);  permute_616 = None
        permute_27: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_618: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_306: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_631, permute_618);  view_631 = permute_618 = None
        view_632: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_306, [4, 512, 2048]);  mm_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_620: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_293, [0, 2, 1, 3]);  add_293 = None
        clone_75: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
        view_633: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_75, [4, 512, 512]);  clone_75 = None
        view_634: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_633, [2048, 512]);  view_633 = None
        permute_621: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_634, [1, 0])
        mm_307: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_621, view_44);  permute_621 = None
        permute_25: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_623: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_308: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_634, permute_623);  view_634 = permute_623 = None
        view_635: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_308, [4, 512, 2048]);  mm_308 = None
        add_296: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_632, view_635);  view_632 = view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_625: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_295, [0, 2, 1, 3]);  add_295 = None
        clone_76: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_625, memory_format = torch.contiguous_format);  permute_625 = None
        view_636: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_76, [4, 512, 2048]);  clone_76 = None
        view_637: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_636, [2048, 2048]);  view_636 = None
        permute_626: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_637, [1, 0])
        mm_309: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_626, view_44);  permute_626 = view_44 = None
        permute_23: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_628: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_310: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_637, permute_628);  view_637 = permute_628 = None
        view_638: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_310, [4, 512, 2048]);  mm_310 = None
        add_297: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_296, view_638);  add_296 = view_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_515: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_297, primals_22);  primals_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_43: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_21: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_43, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_44: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_516: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_297, convert_element_type_44);  add_297 = convert_element_type_44 = None
        sum_89: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 1], True);  mul_516 = None
        view_639: "bf16[2048]" = torch.ops.aten.reshape.default(sum_89, [2048]);  sum_89 = None
        convert_element_type_824: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_515, torch.float32);  mul_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_517: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_824, convert_element_type_43)
        mul_518: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_824, rsqrt_4);  convert_element_type_824 = None
        sum_90: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [2], True);  mul_517 = None
        pow_90: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_519: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_90, -0.5);  sum_90 = None
        mul_520: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_519, pow_90);  mul_519 = pow_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_66: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_520, [4, 512, 2048]);  mul_520 = None
        div_46: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_66, 2048);  expand_66 = None
        pow_91: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_43, 1.0);  convert_element_type_43 = None
        mul_521: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_91, 2.0);  pow_91 = None
        mul_522: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_46, mul_521);  div_46 = mul_521 = None
        add_298: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_518, mul_522);  mul_518 = mul_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_825: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_298, torch.bfloat16);  add_298 = None
        add_299: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_291, convert_element_type_825);  add_291 = convert_element_type_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_640: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_299, [2048, 2048])
        permute_630: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_311: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_630, view_42);  permute_630 = view_42 = None
        permute_22: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_632: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_312: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_640, permute_632);  view_640 = permute_632 = None
        view_641: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_312, [4, 512, 8192]);  mm_312 = None
        view_39: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 8192]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_37: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        neg_5: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_37)
        exp_1: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_15: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_37, add_15)
        convert_element_type_38: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_523: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_641, convert_element_type_38);  convert_element_type_38 = None
        view_41: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 8192]);  mm_12 = None
        mul_524: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_641, view_41);  view_641 = view_41 = None
        view_642: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_523, [2048, 8192]);  mul_523 = None
        permute_634: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_313: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_634, view_38);  permute_634 = None
        permute_21: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_636: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_314: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_642, permute_636);  view_642 = permute_636 = None
        view_643: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_314, [4, 512, 2048]);  mm_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_834: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_524, torch.float32);  mul_524 = None
        reciprocal_14: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_15);  add_15 = None
        mul_525: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        mul_526: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_834, mul_525);  convert_element_type_834 = None
        sub_19: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_525);  mul_525 = None
        mul_527: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_37, sub_19);  convert_element_type_37 = sub_19 = None
        add_301: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_527, 1);  mul_527 = None
        mul_528: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_526, add_301);  mul_526 = add_301 = None
        convert_element_type_836: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_528, torch.bfloat16);  mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_644: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_836, [2048, 8192]);  convert_element_type_836 = None
        permute_638: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_644, [1, 0])
        mm_315: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_638, view_38);  permute_638 = view_38 = None
        permute_20: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_640: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_316: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_644, permute_640);  view_644 = permute_640 = None
        view_645: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_316, [4, 512, 2048]);  mm_316 = None
        add_302: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_643, view_645);  view_643 = view_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_529: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_302, primals_18);  primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_33: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_18: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_33, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_34: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        mul_530: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_302, convert_element_type_34);  add_302 = convert_element_type_34 = None
        sum_91: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 1], True);  mul_530 = None
        view_646: "bf16[2048]" = torch.ops.aten.reshape.default(sum_91, [2048]);  sum_91 = None
        convert_element_type_841: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_529, torch.float32);  mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_531: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_841, convert_element_type_33)
        mul_532: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_841, rsqrt_3);  convert_element_type_841 = None
        sum_92: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True);  mul_531 = None
        pow_92: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_533: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_92, -0.5);  sum_92 = None
        mul_534: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_533, pow_92);  mul_533 = pow_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_67: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_534, [4, 512, 2048]);  mul_534 = None
        div_47: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_67, 2048);  expand_67 = None
        pow_93: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 1.0);  convert_element_type_33 = None
        mul_535: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_93, 2.0);  pow_93 = None
        mul_536: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_47, mul_535);  div_47 = mul_535 = None
        add_303: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_532, mul_536);  mul_532 = mul_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_842: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16);  add_303 = None
        add_304: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_299, convert_element_type_842);  add_299 = convert_element_type_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_647: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_304, [2048, 2048])
        permute_642: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_647, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_18, [4, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_35, [2048, 2048]);  view_35 = None
        mm_317: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_642, view_36);  permute_642 = view_36 = None
        permute_19: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_644: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_318: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_647, permute_644);  view_647 = permute_644 = None
        view_648: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_318, [4, 512, 2048]);  mm_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_649: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_648, [4, 512, 32, 64]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_646: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_14 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_646, add_11, view_33, view_34, getitem_9, getitem_10, getitem_15, getitem_16, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_646 = add_11 = view_33 = view_34 = getitem_9 = getitem_10 = getitem_15 = getitem_16 = None
        getitem_186: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_14[0]
        getitem_187: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_14[1]
        getitem_188: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_14[2];  _scaled_dot_product_cudnn_attention_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_650: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_188, [4, 8, 4, 512, 64]);  getitem_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_93: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_650, [2], True);  view_650 = None
        squeeze_29: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_93, 2);  sum_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_651: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_187, [4, 8, 4, 512, 64]);  getitem_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_94: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_651, [2], True);  view_651 = None
        squeeze_30: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_94, 2);  sum_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_537: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_30, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_125: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_537, 3, 0, 32)
        slice_126: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_537, 3, 32, 64);  mul_537 = None
        neg_92: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_125);  slice_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_56: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_92, 3, 32, 9223372036854775807);  neg_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_57: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_126, 3, 0, 32);  slice_126 = None
        add_305: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_56, slice_scatter_57);  slice_scatter_56 = slice_scatter_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_538: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_30, unsqueeze_14);  squeeze_30 = None
        add_306: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_305, mul_538);  add_305 = mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_539: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_186, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_127: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_539, 3, 0, 32)
        slice_128: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_539, 3, 32, 64);  mul_539 = None
        neg_93: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_127);  slice_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_58: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_93, 3, 32, 9223372036854775807);  neg_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_59: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_128, 3, 0, 32);  slice_128 = None
        add_307: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_59);  slice_scatter_58 = slice_scatter_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_540: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_186, unsqueeze_14);  getitem_186 = None
        add_308: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_307, mul_540);  add_307 = mul_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_647: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_29, [0, 2, 1, 3]);  squeeze_29 = None
        clone_77: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_647, memory_format = torch.contiguous_format);  permute_647 = None
        view_652: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_77, [4, 512, 512]);  clone_77 = None
        view_653: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_652, [2048, 512]);  view_652 = None
        permute_648: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_653, [1, 0])
        mm_319: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_648, view_24);  permute_648 = None
        permute_16: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_650: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_320: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_653, permute_650);  view_653 = permute_650 = None
        view_654: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_320, [4, 512, 2048]);  mm_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_652: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_306, [0, 2, 1, 3]);  add_306 = None
        clone_78: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
        view_655: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_78, [4, 512, 512]);  clone_78 = None
        view_656: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_655, [2048, 512]);  view_655 = None
        permute_653: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_321: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_653, view_24);  permute_653 = None
        permute_14: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_655: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_322: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_656, permute_655);  view_656 = permute_655 = None
        view_657: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_322, [4, 512, 2048]);  mm_322 = None
        add_309: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_654, view_657);  view_654 = view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_657: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_308, [0, 2, 1, 3]);  add_308 = None
        clone_79: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_657, memory_format = torch.contiguous_format);  permute_657 = None
        view_658: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_79, [4, 512, 2048]);  clone_79 = None
        view_659: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_658, [2048, 2048]);  view_658 = None
        permute_658: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_323: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_658, view_24);  permute_658 = view_24 = None
        permute_12: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_660: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_324: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_659, permute_660);  view_659 = permute_660 = None
        view_660: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_324, [4, 512, 2048]);  mm_324 = None
        add_310: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_309, view_660);  add_309 = view_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_541: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_310, primals_13);  primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_23: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_12: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_23, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_24: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_542: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_310, convert_element_type_24);  add_310 = convert_element_type_24 = None
        sum_95: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 1], True);  mul_542 = None
        view_661: "bf16[2048]" = torch.ops.aten.reshape.default(sum_95, [2048]);  sum_95 = None
        convert_element_type_859: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_541, torch.float32);  mul_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_543: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_859, convert_element_type_23)
        mul_544: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_859, rsqrt_2);  convert_element_type_859 = None
        sum_96: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True);  mul_543 = None
        pow_94: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_545: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_96, -0.5);  sum_96 = None
        mul_546: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_545, pow_94);  mul_545 = pow_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_68: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_546, [4, 512, 2048]);  mul_546 = None
        div_48: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_68, 2048);  expand_68 = None
        pow_95: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 1.0);  convert_element_type_23 = None
        mul_547: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_95, 2.0);  pow_95 = None
        mul_548: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_48, mul_547);  div_48 = mul_547 = None
        add_311: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_544, mul_548);  mul_544 = mul_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_860: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_311, torch.bfloat16);  add_311 = None
        add_312: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_304, convert_element_type_860);  add_304 = convert_element_type_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_662: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_312, [2048, 2048])
        permute_662: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_662, [1, 0])
        mm_325: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_662, view_22);  permute_662 = view_22 = None
        permute_11: "bf16[8192, 2048]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_664: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_326: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_662, permute_664);  view_662 = permute_664 = None
        view_663: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_326, [4, 512, 8192]);  mm_326 = None
        view_19: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 8192]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_17: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        neg_2: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_17)
        exp: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_8: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_17, add_8)
        convert_element_type_18: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_549: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_663, convert_element_type_18);  convert_element_type_18 = None
        view_21: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 8192]);  mm_5 = None
        mul_550: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_663, view_21);  view_663 = view_21 = None
        view_664: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_549, [2048, 8192]);  mul_549 = None
        permute_666: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_664, [1, 0])
        mm_327: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_666, view_18);  permute_666 = None
        permute_10: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_668: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_328: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_664, permute_668);  view_664 = permute_668 = None
        view_665: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_328, [4, 512, 2048]);  mm_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_869: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_550, torch.float32);  mul_550 = None
        reciprocal_15: "f32[4, 512, 8192]" = torch.ops.aten.reciprocal.default(add_8);  add_8 = None
        mul_551: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        mul_552: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_869, mul_551);  convert_element_type_869 = None
        sub_20: "f32[4, 512, 8192]" = torch.ops.aten.sub.Tensor(1, mul_551);  mul_551 = None
        mul_553: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_17, sub_20);  convert_element_type_17 = sub_20 = None
        add_314: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(mul_553, 1);  mul_553 = None
        mul_554: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_552, add_314);  mul_552 = add_314 = None
        convert_element_type_871: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(mul_554, torch.bfloat16);  mul_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_666: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(convert_element_type_871, [2048, 8192]);  convert_element_type_871 = None
        permute_670: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_666, [1, 0])
        mm_329: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_670, view_18);  permute_670 = view_18 = None
        permute_9: "bf16[2048, 8192]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_672: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_330: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_666, permute_672);  view_666 = permute_672 = None
        view_667: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_330, [4, 512, 2048]);  mm_330 = None
        add_315: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_665, view_667);  view_665 = view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_555: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_315, primals_9);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_17: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 2048]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_6: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(embedding, view_17);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_13: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_9: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_13, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_14: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_556: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_315, convert_element_type_14);  add_315 = convert_element_type_14 = None
        sum_97: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1], True);  mul_556 = None
        view_668: "bf16[2048]" = torch.ops.aten.reshape.default(sum_97, [2048]);  sum_97 = None
        convert_element_type_876: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_555, torch.float32);  mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_557: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_876, convert_element_type_13)
        mul_558: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_876, rsqrt_1);  convert_element_type_876 = None
        sum_98: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True);  mul_557 = None
        pow_96: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_559: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_98, -0.5);  sum_98 = None
        mul_560: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_559, pow_96);  mul_559 = pow_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_69: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_560, [4, 512, 2048]);  mul_560 = None
        div_49: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_69, 2048);  expand_69 = None
        pow_97: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 1.0);  convert_element_type_13 = None
        mul_561: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_97, 2.0);  pow_97 = None
        mul_562: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_49, mul_561);  div_49 = mul_561 = None
        add_316: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_558, mul_562);  mul_558 = mul_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_877: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_316, torch.bfloat16);  add_316 = None
        add_317: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_312, convert_element_type_877);  add_312 = convert_element_type_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_669: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_317, [2048, 2048])
        permute_674: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_669, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_15, [2048, 2048]);  view_15 = None
        mm_331: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_674, view_16);  permute_674 = view_16 = None
        permute_8: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_676: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_332: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_669, permute_676);  view_669 = permute_676 = None
        view_670: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_332, [4, 512, 2048]);  mm_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_671: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_670, [4, 512, 32, 64]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_678: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_15 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_678, add_4, view_13, view_14, getitem, getitem_1, getitem_6, getitem_7, where, None, None, 512, 512, 0.0, False, scale = 0.125);  permute_678 = add_4 = view_13 = view_14 = getitem = getitem_1 = getitem_6 = getitem_7 = where = None
        getitem_189: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_15[0]
        getitem_190: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_15[1]
        getitem_191: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_backward_15[2];  _scaled_dot_product_cudnn_attention_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_672: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_191, [4, 8, 4, 512, 64]);  getitem_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_99: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_672, [2], True);  view_672 = None
        squeeze_31: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_99, 2);  sum_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_673: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_190, [4, 8, 4, 512, 64]);  getitem_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_100: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(view_673, [2], True);  view_673 = None
        squeeze_32: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_100, 2);  sum_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_563: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_32, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_129: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_563, 3, 0, 32)
        slice_130: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_563, 3, 32, 64);  mul_563 = None
        neg_95: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_129);  slice_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_60: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_95, 3, 32, 9223372036854775807);  neg_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_61: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_130, 3, 0, 32);  full_default_38 = slice_130 = None
        add_318: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_60, slice_scatter_61);  slice_scatter_60 = slice_scatter_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_564: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_32, unsqueeze_14);  squeeze_32 = None
        add_319: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_318, mul_564);  add_318 = mul_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_565: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_189, unsqueeze_15);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_131: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_565, 3, 0, 32)
        slice_132: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_565, 3, 32, 64);  mul_565 = None
        neg_96: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_131);  slice_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_62: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_96, 3, 32, 9223372036854775807);  neg_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_63: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_132, 3, 0, 32);  full_default_40 = slice_132 = None
        add_320: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_62, slice_scatter_63);  slice_scatter_62 = slice_scatter_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_566: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_189, unsqueeze_14);  getitem_189 = unsqueeze_14 = None
        add_321: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_320, mul_566);  add_320 = mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_679: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_31, [0, 2, 1, 3]);  squeeze_31 = None
        clone_80: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_679, memory_format = torch.contiguous_format);  permute_679 = None
        view_674: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_80, [4, 512, 512]);  clone_80 = None
        view_675: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_674, [2048, 512]);  view_674 = None
        permute_680: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_675, [1, 0])
        mm_333: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_680, view_4);  permute_680 = None
        permute_5: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_682: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_334: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_675, permute_682);  view_675 = permute_682 = None
        view_676: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_334, [4, 512, 2048]);  mm_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_684: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_319, [0, 2, 1, 3]);  add_319 = None
        clone_81: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
        view_677: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_81, [4, 512, 512]);  clone_81 = None
        view_678: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_677, [2048, 512]);  view_677 = None
        permute_685: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_678, [1, 0])
        mm_335: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_685, view_4);  permute_685 = None
        permute_3: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_687: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_336: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_678, permute_687);  view_678 = permute_687 = None
        view_679: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_336, [4, 512, 2048]);  mm_336 = None
        add_322: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_676, view_679);  view_676 = view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_689: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_321, [0, 2, 1, 3]);  add_321 = None
        clone_82: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
        view_680: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_82, [4, 512, 2048]);  clone_82 = None
        view_681: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_680, [2048, 2048]);  view_680 = None
        permute_690: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_337: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_690, view_4);  permute_690 = view_4 = None
        permute_1: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_692: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_338: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_681, permute_692);  view_681 = permute_692 = None
        view_682: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_338, [4, 512, 2048]);  mm_338 = None
        add_323: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_322, view_682);  add_322 = view_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_567: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_323, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_3: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_4: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_568: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_323, convert_element_type_4);  add_323 = convert_element_type_4 = None
        sum_101: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 1], True);  mul_568 = None
        view_683: "bf16[2048]" = torch.ops.aten.reshape.default(sum_101, [2048]);  sum_101 = None
        convert_element_type_894: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_567, torch.float32);  mul_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_569: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_894, convert_element_type_3)
        mul_570: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_894, rsqrt);  convert_element_type_894 = None
        sum_102: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True);  mul_569 = None
        pow_98: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_571: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_102, -0.5);  sum_102 = None
        mul_572: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_571, pow_98);  mul_571 = pow_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_70: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_572, [4, 512, 2048]);  mul_572 = None
        div_50: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_70, 2048);  expand_70 = None
        pow_99: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 1.0);  convert_element_type_3 = None
        mul_573: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_99, 2.0);  pow_99 = None
        mul_574: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_50, mul_573);  div_50 = mul_573 = None
        add_324: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_570, mul_574);  mul_570 = mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_895: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16);  add_324 = None
        add_325: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_317, convert_element_type_895);  add_317 = convert_element_type_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:389 in forward, code: inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        convert_element_type_896: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_325, torch.float32);  add_325 = None
        eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_80: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_20: "f32[4, 512, 2048]" = torch.ops.aten.where.self(unsqueeze_80, full_default_34, convert_element_type_896);  unsqueeze_80 = full_default_34 = convert_element_type_896 = None
        full_default_103: "f32[128256, 2048]" = torch.ops.aten.full.default([128256, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[128256, 2048]" = torch.ops.aten.index_put.default(full_default_103, [primals_1], where_20, True);  full_default_103 = primals_1 = where_20 = None
        convert_element_type_897: "bf16[128256, 2048]" = torch.ops.prims.convert_element_type.default(index_put, torch.bfloat16);  index_put = None
        add_326: "bf16[128256, 2048]" = torch.ops.aten.add.Tensor(mm_113, convert_element_type_897);  mm_113 = convert_element_type_897 = None
        return (None, add_326, None, view_683, mm_337, mm_335, mm_333, mm_331, view_668, mm_329, mm_327, mm_325, view_661, mm_323, mm_321, mm_319, mm_317, view_646, mm_315, mm_313, mm_311, view_639, mm_309, mm_307, mm_305, mm_303, view_624, mm_301, mm_299, mm_297, view_617, mm_295, mm_293, mm_291, mm_289, view_602, mm_287, mm_285, mm_283, view_595, mm_281, mm_279, mm_277, mm_275, view_580, mm_273, mm_271, mm_269, view_573, mm_267, mm_265, mm_263, mm_261, view_558, mm_259, mm_257, mm_255, view_551, mm_253, mm_251, mm_249, mm_247, view_536, mm_245, mm_243, mm_241, view_529, mm_239, mm_237, mm_235, mm_233, view_514, mm_231, mm_229, mm_227, view_507, mm_225, mm_223, mm_221, mm_219, view_492, mm_217, mm_215, mm_213, view_485, mm_211, mm_209, mm_207, mm_205, view_470, mm_203, mm_201, mm_199, view_463, mm_197, mm_195, mm_193, mm_191, view_448, mm_189, mm_187, mm_185, view_441, mm_183, mm_181, mm_179, mm_177, view_426, mm_175, mm_173, mm_171, view_419, mm_169, mm_167, mm_165, mm_163, view_404, mm_161, mm_159, mm_157, view_397, mm_155, mm_153, mm_151, mm_149, view_382, mm_147, mm_145, mm_143, view_375, mm_141, mm_139, mm_137, mm_135, view_360, mm_133, mm_131, mm_129, view_353, mm_127, mm_125, mm_123, mm_121, view_338, mm_119, mm_117, mm_115, view_331, None)
