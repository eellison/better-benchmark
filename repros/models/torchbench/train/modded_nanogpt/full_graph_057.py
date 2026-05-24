import torch
from torch import device
from math import inf, nan

# Register nanogpt custom ops (from modded_nanogpt benchmark model)
if not hasattr(torch.ops, 'nanogpt') or not hasattr(torch.ops.nanogpt, 'mm'):
    @torch.library.custom_op("nanogpt::mm", mutates_args=())
    def _mm_op(x: torch.Tensor, w: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(x_f8, w_f8.T, out_dtype=torch.bfloat16,
                               scale_a=x.new_tensor(x_s, dtype=torch.float32),
                               scale_b=x.new_tensor(w_s, dtype=torch.float32), use_fast_accum=True)
        return out, x_f8, w_f8

    @_mm_op.register_fake
    def _mm_fake(x: torch.Tensor, w: torch.Tensor, *_):
        return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

    @torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
    def _mm_backward_op(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        grad_f8 = g.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(grad_f8, w_f8.T.contiguous().T, out_dtype=torch.bfloat16,
                                   scale_a=g.new_tensor(grad_s, dtype=torch.float32),
                                   scale_b=g.new_tensor(w_s, dtype=torch.float32), use_fast_accum=False)
        grad_w = torch._scaled_mm(x_f8.T.contiguous(), grad_f8.T.contiguous().T, out_dtype=torch.float32,
                                  scale_a=g.new_tensor(x_s, dtype=torch.float32),
                                  scale_b=g.new_tensor(grad_s, dtype=torch.float32), use_fast_accum=False)
        return grad_x, grad_w.T

    @_mm_backward_op.register_fake
    def _mm_backward_fake(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, *_):
        return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)


class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i32[6144]", primals_7: "f32[64]", primals_9: "f32[262144, 64]", primals_10: "f32[262144, 64]", primals_15: "f32[262144, 64]", primals_16: "f32[262144, 64]", primals_21: "f32[262144, 64]", primals_22: "f32[262144, 64]", primals_27: "f32[262144, 64]", primals_28: "f32[262144, 64]", primals_33: "f32[262144, 64]", primals_34: "f32[262144, 64]", primals_39: "f32[262144, 64]", primals_40: "f32[262144, 64]", primals_45: "f32[262144, 64]", primals_46: "f32[262144, 64]", primals_53: "f32[262144, 64]", primals_54: "f32[262144, 64]", primals_59: "f32[262144, 64]", primals_60: "f32[262144, 64]", primals_65: "f32[262144, 64]", primals_66: "f32[262144, 64]", primals_71: "f32[262144, 64]", primals_72: "f32[262144, 64]", primals_77: "i64[6144]", embedding: "bf16[6144, 768]", embedding_1: "bf16[6144, 768]", embedding_2: "bf16[6144, 768]", cumsum: "i64[6144]", unsqueeze_9: "i32[1, 1, 48, 48]", unsqueeze_13: "i32[1, 1, 48, 48]", clamp_max: "i32[1, 1, 48]", clamp_max_1: "i32[1, 1, 48]", convert_element_type_2: "i32[1, 1, 48]", clone_4: "i32[1, 1, 48, 48]", convert_element_type_4: "i32[1, 1, 48]", clone_7: "i32[1, 1, 48, 48]", clamp_max_2: "i32[1, 1, 48]", clamp_max_3: "i32[1, 1, 48]", convert_element_type_6: "i32[1, 1, 48]", clone_10: "i32[1, 1, 48, 48]", convert_element_type_8: "i32[1, 1, 48]", clone_13: "i32[1, 1, 48, 48]", embedding_3: "bf16[6144, 768]", rsqrt: "f32[1, 6144, 1]", rsqrt_1: "f32[1, 6144, 1]", view_9: "bf16[6144, 768]", mm: "bf16[6144, 2304]", rsqrt_2: "f32[1, 6144, 6, 1]", rsqrt_3: "f32[1, 6144, 6, 1]", permute_5: "bf16[1, 6, 6144, 128]", permute_6: "bf16[1, 6, 6144, 128]", permute_7: "bf16[1, 6, 6144, 128]", getitem_19: "bf16[1, 6, 6144, 128]", getitem_20: "f32[1, 6, 6144]", view_14: "bf16[6144, 12]", mm_1: "bf16[6144, 6]", view_18: "bf16[6144, 768]", add_10: "bf16[1, 6144, 768]", rsqrt_4: "f32[1, 6144, 1]", mm_3: "bf16[6144, 3072]", view_22: "bf16[6144, 3072]", add_12: "bf16[1, 6144, 768]", add_13: "bf16[1, 6144, 768]", rsqrt_5: "f32[1, 6144, 1]", view_25: "bf16[6144, 768]", mm_5: "bf16[6144, 2304]", rsqrt_6: "f32[1, 6144, 6, 1]", rsqrt_7: "f32[1, 6144, 6, 1]", permute_15: "bf16[1, 6, 6144, 128]", permute_16: "bf16[1, 6, 6144, 128]", permute_17: "bf16[1, 6, 6144, 128]", getitem_29: "bf16[1, 6, 6144, 128]", getitem_30: "f32[1, 6, 6144]", view_30: "bf16[6144, 12]", mm_6: "bf16[6144, 6]", view_34: "bf16[6144, 768]", add_22: "bf16[1, 6144, 768]", rsqrt_8: "f32[1, 6144, 1]", mm_8: "bf16[6144, 3072]", view_38: "bf16[6144, 3072]", add_24: "bf16[1, 6144, 768]", add_25: "bf16[1, 6144, 768]", rsqrt_9: "f32[1, 6144, 1]", view_41: "bf16[6144, 768]", mm_10: "bf16[6144, 2304]", rsqrt_10: "f32[1, 6144, 6, 1]", rsqrt_11: "f32[1, 6144, 6, 1]", permute_25: "bf16[1, 6, 6144, 128]", permute_26: "bf16[1, 6, 6144, 128]", permute_27: "bf16[1, 6, 6144, 128]", getitem_39: "bf16[1, 6, 6144, 128]", getitem_40: "f32[1, 6, 6144]", view_46: "bf16[6144, 12]", mm_11: "bf16[6144, 6]", view_50: "bf16[6144, 768]", add_34: "bf16[1, 6144, 768]", rsqrt_12: "f32[1, 6144, 1]", mm_13: "bf16[6144, 3072]", view_54: "bf16[6144, 3072]", add_36: "bf16[1, 6144, 768]", add_37: "bf16[1, 6144, 768]", rsqrt_13: "f32[1, 6144, 1]", view_57: "bf16[6144, 768]", mm_15: "bf16[6144, 2304]", rsqrt_14: "f32[1, 6144, 6, 1]", rsqrt_15: "f32[1, 6144, 6, 1]", permute_35: "bf16[1, 6, 6144, 128]", permute_36: "bf16[1, 6, 6144, 128]", permute_37: "bf16[1, 6, 6144, 128]", getitem_49: "bf16[1, 6, 6144, 128]", getitem_50: "f32[1, 6, 6144]", view_61: "bf16[6144, 12]", mm_16: "bf16[6144, 6]", view_65: "bf16[6144, 768]", add_45: "bf16[1, 6144, 768]", rsqrt_16: "f32[1, 6144, 1]", mm_18: "bf16[6144, 3072]", view_69: "bf16[6144, 3072]", add_47: "bf16[1, 6144, 768]", add_48: "bf16[1, 6144, 768]", rsqrt_17: "f32[1, 6144, 1]", view_72: "bf16[6144, 768]", mm_20: "bf16[6144, 2304]", rsqrt_18: "f32[1, 6144, 6, 1]", rsqrt_19: "f32[1, 6144, 6, 1]", permute_45: "bf16[1, 6, 6144, 128]", permute_46: "bf16[1, 6, 6144, 128]", permute_47: "bf16[1, 6, 6144, 128]", getitem_59: "bf16[1, 6, 6144, 128]", getitem_60: "f32[1, 6, 6144]", view_76: "bf16[6144, 12]", mm_21: "bf16[6144, 6]", view_80: "bf16[6144, 768]", add_56: "bf16[1, 6144, 768]", rsqrt_20: "f32[1, 6144, 1]", mm_23: "bf16[6144, 3072]", view_84: "bf16[6144, 3072]", add_58: "bf16[1, 6144, 768]", add_59: "bf16[1, 6144, 768]", rsqrt_21: "f32[1, 6144, 1]", view_87: "bf16[6144, 768]", mm_25: "bf16[6144, 2304]", rsqrt_22: "f32[1, 6144, 6, 1]", rsqrt_23: "f32[1, 6144, 6, 1]", permute_55: "bf16[1, 6, 6144, 128]", permute_56: "bf16[1, 6, 6144, 128]", permute_57: "bf16[1, 6, 6144, 128]", getitem_69: "bf16[1, 6, 6144, 128]", getitem_70: "f32[1, 6, 6144]", view_91: "bf16[6144, 12]", mm_26: "bf16[6144, 6]", view_95: "bf16[6144, 768]", add_67: "bf16[1, 6144, 768]", rsqrt_24: "f32[1, 6144, 1]", mm_28: "bf16[6144, 3072]", view_99: "bf16[6144, 3072]", add_69: "bf16[1, 6144, 768]", add_71: "bf16[1, 6144, 768]", rsqrt_25: "f32[1, 6144, 1]", view_102: "bf16[6144, 768]", mm_30: "bf16[6144, 2304]", rsqrt_26: "f32[1, 6144, 6, 1]", rsqrt_27: "f32[1, 6144, 6, 1]", permute_65: "bf16[1, 6, 6144, 128]", permute_66: "bf16[1, 6, 6144, 128]", permute_67: "bf16[1, 6, 6144, 128]", getitem_79: "bf16[1, 6, 6144, 128]", getitem_80: "f32[1, 6, 6144]", view_106: "bf16[6144, 12]", mm_31: "bf16[6144, 6]", view_110: "bf16[6144, 768]", add_79: "bf16[1, 6144, 768]", rsqrt_28: "f32[1, 6144, 1]", mm_33: "bf16[6144, 3072]", view_114: "bf16[6144, 3072]", add_82: "bf16[1, 6144, 768]", add_83: "bf16[1, 6144, 768]", rsqrt_29: "f32[1, 6144, 1]", mm_35: "bf16[6144, 3072]", view_118: "bf16[6144, 3072]", add_86: "bf16[1, 6144, 768]", add_87: "bf16[1, 6144, 768]", rsqrt_30: "f32[1, 6144, 1]", view_121: "bf16[6144, 768]", mm_37: "bf16[6144, 2304]", rsqrt_31: "f32[1, 6144, 6, 1]", rsqrt_32: "f32[1, 6144, 6, 1]", permute_78: "bf16[1, 6, 6144, 128]", permute_79: "bf16[1, 6, 6144, 128]", permute_80: "bf16[1, 6, 6144, 128]", getitem_89: "bf16[1, 6, 6144, 128]", getitem_90: "f32[1, 6, 6144]", view_125: "bf16[6144, 12]", mm_38: "bf16[6144, 6]", view_129: "bf16[6144, 768]", add_95: "bf16[1, 6144, 768]", rsqrt_33: "f32[1, 6144, 1]", mm_40: "bf16[6144, 3072]", view_133: "bf16[6144, 3072]", add_98: "bf16[1, 6144, 768]", add_99: "bf16[1, 6144, 768]", rsqrt_34: "f32[1, 6144, 1]", view_136: "bf16[6144, 768]", mm_42: "bf16[6144, 2304]", rsqrt_35: "f32[1, 6144, 6, 1]", rsqrt_36: "f32[1, 6144, 6, 1]", permute_88: "bf16[1, 6, 6144, 128]", permute_89: "bf16[1, 6, 6144, 128]", permute_90: "bf16[1, 6, 6144, 128]", getitem_99: "bf16[1, 6, 6144, 128]", getitem_100: "f32[1, 6, 6144]", view_141: "bf16[6144, 12]", mm_43: "bf16[6144, 6]", view_145: "bf16[6144, 768]", add_108: "bf16[1, 6144, 768]", rsqrt_37: "f32[1, 6144, 1]", mm_45: "bf16[6144, 3072]", view_149: "bf16[6144, 3072]", add_111: "bf16[1, 6144, 768]", add_112: "bf16[1, 6144, 768]", rsqrt_38: "f32[1, 6144, 1]", view_152: "bf16[6144, 768]", mm_47: "bf16[6144, 2304]", rsqrt_39: "f32[1, 6144, 6, 1]", rsqrt_40: "f32[1, 6144, 6, 1]", permute_98: "bf16[1, 6, 6144, 128]", permute_99: "bf16[1, 6, 6144, 128]", permute_100: "bf16[1, 6, 6144, 128]", getitem_109: "bf16[1, 6, 6144, 128]", getitem_110: "f32[1, 6, 6144]", view_157: "bf16[6144, 12]", mm_48: "bf16[6144, 6]", view_161: "bf16[6144, 768]", add_121: "bf16[1, 6144, 768]", rsqrt_41: "f32[1, 6144, 1]", mm_50: "bf16[6144, 3072]", view_165: "bf16[6144, 3072]", add_124: "bf16[1, 6144, 768]", add_125: "bf16[1, 6144, 768]", rsqrt_42: "f32[1, 6144, 1]", view_168: "bf16[6144, 768]", mm_52: "bf16[6144, 2304]", rsqrt_43: "f32[1, 6144, 6, 1]", rsqrt_44: "f32[1, 6144, 6, 1]", permute_108: "bf16[1, 6, 6144, 128]", permute_109: "bf16[1, 6, 6144, 128]", permute_110: "bf16[1, 6, 6144, 128]", getitem_119: "bf16[1, 6, 6144, 128]", getitem_120: "f32[1, 6, 6144]", view_173: "bf16[6144, 12]", mm_53: "bf16[6144, 6]", view_177: "bf16[6144, 768]", add_134: "bf16[1, 6144, 768]", rsqrt_45: "f32[1, 6144, 1]", mm_55: "bf16[6144, 3072]", view_181: "bf16[6144, 3072]", add_136: "bf16[1, 6144, 768]", rsqrt_46: "f32[1, 6144, 1]", getitem_123: "f8e4m3fn[6144, 768]", getitem_124: "f8e4m3fn[50304, 768]", view_184: "bf16[1, 6144, 50304]", amax: "f32[6144, 1]", log: "f32[6144, 1]", permute_119: "bf16[768, 3072]", permute_121: "bf16[768, 6144]", permute_122: "bf16[3072, 768]", permute_127: "bf16[768, 768]", permute_131: "bf16[6, 12]", permute_139: "bf16[2304, 768]", permute_143: "bf16[768, 3072]", permute_145: "bf16[768, 6144]", permute_146: "bf16[3072, 768]", permute_151: "bf16[768, 768]", permute_155: "bf16[6, 12]", permute_163: "bf16[2304, 768]", permute_167: "bf16[768, 3072]", permute_169: "bf16[768, 6144]", permute_170: "bf16[3072, 768]", permute_175: "bf16[768, 768]", permute_179: "bf16[6, 12]", permute_187: "bf16[2304, 768]", permute_191: "bf16[768, 3072]", permute_193: "bf16[768, 6144]", permute_194: "bf16[3072, 768]", permute_199: "bf16[768, 768]", permute_203: "bf16[6, 12]", permute_211: "bf16[2304, 768]", permute_215: "bf16[768, 3072]", permute_217: "bf16[768, 6144]", permute_218: "bf16[3072, 768]", permute_223: "bf16[768, 3072]", permute_225: "bf16[768, 6144]", permute_226: "bf16[3072, 768]", permute_231: "bf16[768, 768]", permute_235: "bf16[6, 12]", permute_243: "bf16[2304, 768]", permute_247: "bf16[768, 3072]", permute_249: "bf16[768, 6144]", permute_250: "bf16[3072, 768]", permute_255: "bf16[768, 768]", permute_259: "bf16[6, 12]", permute_267: "bf16[2304, 768]", permute_271: "bf16[768, 3072]", permute_273: "bf16[768, 6144]", permute_274: "bf16[3072, 768]", permute_279: "bf16[768, 768]", permute_283: "bf16[6, 12]", permute_291: "bf16[2304, 768]", permute_295: "bf16[768, 3072]", permute_297: "bf16[768, 6144]", permute_298: "bf16[3072, 768]", permute_303: "bf16[768, 768]", permute_307: "bf16[6, 12]", permute_315: "bf16[2304, 768]", permute_319: "bf16[768, 3072]", permute_321: "bf16[768, 6144]", permute_322: "bf16[3072, 768]", permute_327: "bf16[768, 768]", permute_331: "bf16[6, 12]", permute_339: "bf16[2304, 768]", permute_343: "bf16[768, 3072]", permute_345: "bf16[768, 6144]", permute_346: "bf16[3072, 768]", permute_351: "bf16[768, 768]", permute_355: "bf16[6, 12]", permute_363: "bf16[2304, 768]", permute_367: "bf16[768, 3072]", permute_369: "bf16[768, 6144]", permute_370: "bf16[3072, 768]", permute_375: "bf16[768, 768]", permute_379: "bf16[6, 12]", permute_387: "bf16[2304, 768]", tangents_1: "f32[]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_132: "i64[6144, 1]" = torch.ops.aten.unsqueeze.default(primals_77, 1);  primals_77 = None
        ne_3: "b8[6144, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_132, -100)
        where_7: "f32[6144, 1]" = torch.ops.aten.where.self(ne_3, tangents_1, full_default_13);  tangents_1 = None
        full_default_12: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "i64[6144, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_132, full_default_12);  ne_3 = unsqueeze_132 = full_default_12 = None

        # No stacktrace found for following nodes
        iota_default: "i64[50304]" = torch.ops.prims.iota.default(50304, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50304]" = torch.ops.aten.reshape.default(iota_default, [1, 50304]);  iota_default = None
        expand_default: "i64[6144, 50304]" = torch.ops.aten.expand.default(where_6, [6144, 50304]);  where_6 = None
        eq_tensor: "b8[6144, 50304]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        where_self: "f32[6144, 50304]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        mul_194: "f32[6144, 50304]" = torch.ops.aten.mul.Tensor(where_self, where_7);  where_self = where_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:929 in forward, code: logits = self.lm_head(x).float()
        convert_element_type_319: "f32[1, 6144, 50304]" = torch.ops.prims.convert_element_type.default(view_184, torch.float32);  view_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:931 in forward, code: logits = 30 * torch.sigmoid(logits / 7.5)
        div_1: "f32[1, 6144, 50304]" = torch.ops.aten.div.Tensor(convert_element_type_319, 7.5);  convert_element_type_319 = None
        sigmoid_11: "f32[1, 6144, 50304]" = torch.ops.aten.sigmoid.default(div_1);  div_1 = None
        mul_193: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(sigmoid_11, 30)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:933 in forward, code: logits.view(-1, logits.size(-1)),
        view_185: "f32[6144, 50304]" = torch.ops.aten.reshape.default(mul_193, [-1, 50304]);  mul_193 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        sub_4: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(view_185, amax);  view_185 = amax = None
        sub_5: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(sub_4, log);  sub_4 = log = None
        exp_1: "f32[6144, 50304]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_10: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [1], True)
        mul_195: "f32[6144, 50304]" = torch.ops.aten.mul.Tensor(exp_1, sum_10);  exp_1 = sum_10 = None
        sub_6: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:933 in forward, code: logits.view(-1, logits.size(-1)),
        view_186: "f32[1, 6144, 50304]" = torch.ops.aten.reshape.default(sub_6, [1, 6144, 50304]);  sub_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:931 in forward, code: logits = 30 * torch.sigmoid(logits / 7.5)
        mul_196: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(view_186, 30);  view_186 = None
        sub_7: "f32[1, 6144, 50304]" = torch.ops.aten.sub.Tensor(1, sigmoid_11)
        mul_197: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(sigmoid_11, sub_7);  sigmoid_11 = sub_7 = None
        mul_198: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        div_2: "f32[1, 6144, 50304]" = torch.ops.aten.div.Tensor(mul_198, 7.5);  mul_198 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:929 in forward, code: logits = self.lm_head(x).float()
        convert_element_type_321: "bf16[1, 6144, 50304]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:616 in forward, code: return out.reshape(*x.shape[:-1], -1)
        view_187: "bf16[6144, 50304]" = torch.ops.aten.reshape.default(convert_element_type_321, [6144, 50304]);  convert_element_type_321 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:613 in forward, code: out: Tensor = torch.ops.nanogpt.mm(
        mm_backward = torch.ops.nanogpt.mm_backward.default(view_187, getitem_123, getitem_124, 0.06185895741317419, 0.001953125, 0.002232142857142857);  view_187 = getitem_123 = getitem_124 = None
        getitem_125: "bf16[6144, 768]" = mm_backward[0]
        getitem_126: "f32[50304, 768]" = mm_backward[1];  mm_backward = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:612 in forward, code: _x = x.flatten(0, -2)
        view_188: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(getitem_125, [1, 6144, 768]);  getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_322: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        convert_element_type_317: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_136, torch.float32);  add_136 = None
        mul_192: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_317, rsqrt_46);  convert_element_type_317 = None
        mul_200: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_192, convert_element_type_322)
        sum_11: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
        div_3: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_192, 768);  mul_192 = None
        mul_201: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_3, sum_11);  div_3 = sum_11 = None
        sub_8: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_322, mul_201);  convert_element_type_322 = mul_201 = None
        mul_202: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_46);  sub_8 = rsqrt_46 = None
        convert_element_type_324: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_189: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_324, [6144, 768])
        permute_117: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_58: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_117, view_181);  permute_117 = view_181 = None
        mm_59: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_189, permute_119);  view_189 = permute_119 = None
        view_190: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_59, [1, 6144, 3072]);  mm_59 = None
        convert_element_type_329: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_180: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_55, [1, 6144, 3072]);  mm_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_11: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_180);  view_180 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_60: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_11, 1.0)
        mul_203: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
        mul_204: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_190, mul_203);  view_190 = mul_203 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_1: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        full_default_17: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_1, full_default_17, mul_204);  le_1 = mul_204 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_191: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_8, [6144, 3072]);  where_8 = None
        mm_60: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_121, view_191);  permute_121 = None
        mm_61: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_191, permute_122);  view_191 = permute_122 = None
        view_192: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_61, [1, 6144, 768]);  mm_61 = None
        permute_123: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
        convert_element_type_334: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_123, torch.float32);  permute_123 = None
        permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_335: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_192, torch.float32);  view_192 = None
        convert_element_type_309: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_134, torch.float32);  add_134 = None
        mul_191: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_309, rsqrt_45);  convert_element_type_309 = None
        mul_206: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_191, convert_element_type_335)
        sum_12: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True);  mul_206 = None
        div_4: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_191, 768);  mul_191 = None
        mul_207: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_4, sum_12);  div_4 = sum_12 = None
        sub_9: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_335, mul_207);  convert_element_type_335 = mul_207 = None
        mul_208: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_45);  sub_9 = rsqrt_45 = None
        convert_element_type_337: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None
        add_138: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(convert_element_type_324, convert_element_type_337);  convert_element_type_324 = convert_element_type_337 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_193: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_138, [6144, 768])
        permute_125: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_193, [1, 0])
        mm_62: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_125, view_177);  permute_125 = view_177 = None
        mm_63: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_193, permute_127);  view_193 = permute_127 = None
        view_194: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_63, [1, 6144, 768]);  mm_63 = None
        convert_element_type_342: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        full_default_18: "f32[4, 768, 768]" = torch.ops.aten.full.default([4, 768, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_342, 0, 3);  convert_element_type_342 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_195: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_194, [1, 6144, 6, 128]);  view_194 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_111: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_209: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_195, permute_111);  permute_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_174: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_53, [1, 6144, 6]);  mm_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_10: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_174);  view_174 = None
        view_175: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_10, [1, 6144, 6, 1])
        mul_210: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_195, view_175);  view_195 = view_175 = None
        sum_13: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [3], True);  mul_209 = None
        view_196: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_13, [1, 6144, 6]);  sum_13 = None
        convert_element_type_343: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_196, torch.float32);  view_196 = None
        convert_element_type_344: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_10, torch.float32);  sigmoid_10 = None
        sub_10: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_344)
        mul_211: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_344, sub_10);  convert_element_type_344 = sub_10 = None
        mul_212: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_343, mul_211);  convert_element_type_343 = mul_211 = None
        convert_element_type_345: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_197: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_345, [6144, 6]);  convert_element_type_345 = None
        permute_129: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_197, [1, 0])
        mm_64: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_129, view_173);  permute_129 = view_173 = None
        mm_65: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_197, permute_131);  view_197 = permute_131 = None
        view_198: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_65, [1, 6144, 12]);  mm_65 = None
        convert_element_type_350: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        full_default_19: "bf16[1, 6144, 768]" = torch.ops.aten.full.default([1, 6144, 768], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_198, 2, 0, 12);  view_198 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_133: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_210, [0, 2, 1, 3]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph0 = self.fw_graph0
        joint_graph0 = self.joint_graph0
        mask_graph0 = self.mask_graph0
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(permute_108, permute_109, permute_110, getitem_119, getitem_120, permute_133, None, fw_graph0, joint_graph0, (6144, 6144, clamp_max, unsqueeze_9, clamp_max_1, unsqueeze_13, convert_element_type_2, clone_4, convert_element_type_4, clone_7, 128, 128, mask_graph0), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_108 = permute_109 = permute_110 = getitem_119 = getitem_120 = permute_133 = fw_graph0 = joint_graph0 = mask_graph0 = None
        getitem_127: "bf16[1, 6, 6144, 128]" = flex_attention_backward[0]
        getitem_128: "bf16[1, 6, 6144, 128]" = flex_attention_backward[1]
        getitem_129: "bf16[1, 6, 6144, 128]" = flex_attention_backward[2];  flex_attention_backward = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_134: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1, 3]);  getitem_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_135: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_136: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:915 in forward, code: sa_lambdas = self.scalars[3 * len(self.blocks) : 5 * len(self.blocks)].view(
        slice_11: "f32[24]" = torch.ops.aten.slice.Tensor(primals_7, 0, 36, 60)
        view_7: "f32[12, 2]" = torch.ops.aten.reshape.default(slice_11, [-1, 2]);  slice_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_77: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_81: "f32[]" = torch.ops.aten.select.int(select_77, 0, 1)
        mul_213: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_134, select_81);  select_81 = None
        view_44: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_2, [1, 6144, 6, 128]);  embedding_2 = None
        mul_214: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_134, view_44)
        sum_14: "bf16[]" = torch.ops.aten.sum.default(mul_214);  mul_214 = None
        convert_element_type_351: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        view_200: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_213, [6144, 768]);  mul_213 = None
        full_default_20: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_1: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_351, 0, 1);  convert_element_type_351 = None
        select_80: "f32[]" = torch.ops.aten.select.int(select_77, 0, 0);  select_77 = None
        mul_215: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_134, select_80);  select_80 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_169: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_52, [1, 6144, 2304]);  mm_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_170: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_169, [1, 6144, 18, 128]);  view_169 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_30 = torch.ops.aten.split.Tensor(view_170, 6, -2);  view_170 = None
        getitem_114: "bf16[1, 6144, 6, 128]" = split_30[2]
        getitem_113: "bf16[1, 6144, 6, 128]" = split_30[1]
        getitem_112: "bf16[1, 6144, 6, 128]" = split_30[0];  split_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        mul_216: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_134, getitem_114);  permute_134 = getitem_114 = None
        sum_15: "bf16[]" = torch.ops.aten.sum.default(mul_216);  mul_216 = None
        convert_element_type_352: "f32[]" = torch.ops.prims.convert_element_type.default(sum_15, torch.float32);  sum_15 = None
        select_scatter_2: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_352, 0, 0);  convert_element_type_352 = None
        add_139: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_1, select_scatter_2);  select_scatter_1 = select_scatter_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_353: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_135, torch.float32);  permute_135 = None
        slice_78: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_353, 3, 0, 64)
        slice_79: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_353, 3, 64, 128);  convert_element_type_353 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_123: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_71, 0);  primals_71 = None
        slice_73: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_123, 1, 0, 6144);  unsqueeze_123 = None
        unsqueeze_124: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_73, 2);  slice_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_217: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_79, unsqueeze_124)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_125: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_72, 0);  primals_72 = None
        slice_74: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_125, 1, 0, 6144);  unsqueeze_125 = None
        unsqueeze_126: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_74, 2);  slice_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_20: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_126)
        mul_218: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_79, neg_20);  slice_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_219: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_78, unsqueeze_126)
        add_140: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_217, mul_219);  mul_217 = mul_219 = None
        mul_220: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_78, unsqueeze_124);  slice_78 = None
        add_141: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_218, mul_220);  mul_218 = mul_220 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_22: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_141, add_140], 3);  add_141 = add_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_355: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_136, torch.float32);  permute_136 = None
        slice_80: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_355, 3, 0, 64)
        slice_81: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_355, 3, 64, 128);  convert_element_type_355 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_221: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_81, unsqueeze_124)
        mul_222: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_81, neg_20);  slice_81 = neg_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_223: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_80, unsqueeze_126);  unsqueeze_126 = None
        add_142: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_221, mul_223);  mul_221 = mul_223 = None
        mul_224: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_80, unsqueeze_124);  slice_80 = unsqueeze_124 = None
        add_143: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_222, mul_224);  mul_222 = mul_224 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_23: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_143, add_142], 3);  add_143 = add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_21: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32);  cat_22 = None
        convert_element_type_297: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
        mul_179: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_297, rsqrt_44);  convert_element_type_297 = None
        mul_226: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_179, convert_element_type_default_21)
        sum_16: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [3], True);  mul_226 = None
        div_5: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_179, 128);  mul_179 = None
        mul_227: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_5, sum_16);  div_5 = sum_16 = None
        sub_11: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_21, mul_227);  convert_element_type_default_21 = mul_227 = None
        mul_228: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_44);  sub_11 = rsqrt_44 = None
        convert_element_type_359: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16);  mul_228 = None
        convert_element_type_default_20: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32);  cat_23 = None
        convert_element_type_295: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32);  getitem_112 = None
        mul_178: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_295, rsqrt_43);  convert_element_type_295 = None
        mul_230: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_178, convert_element_type_default_20)
        sum_17: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [3], True);  mul_230 = None
        div_6: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_178, 128);  mul_178 = None
        mul_231: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_6, sum_17);  div_6 = sum_17 = None
        sub_12: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_20, mul_231);  convert_element_type_default_20 = mul_231 = None
        mul_232: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_43);  sub_12 = rsqrt_43 = None
        convert_element_type_362: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16);  mul_232 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_24: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_362, convert_element_type_359, mul_215], 2);  convert_element_type_362 = convert_element_type_359 = mul_215 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_201: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_24, [1, 6144, 2304]);  cat_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_202: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_201, [6144, 2304]);  view_201 = None
        permute_137: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_202, [1, 0])
        mm_66: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_137, view_168);  permute_137 = view_168 = None
        mm_67: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_202, permute_139);  view_202 = permute_139 = None
        view_203: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_67, [1, 6144, 768]);  mm_67 = None
        add_144: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter, view_203);  slice_scatter = view_203 = None
        convert_element_type_367: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        view_204: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_367, [3, 768, 768]);  convert_element_type_367 = None
        slice_scatter_1: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_204, 0, 0, 3);  view_204 = None
        add_145: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter, slice_scatter_1);  select_scatter = slice_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_368: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_144, torch.float32);  add_144 = None
        convert_element_type_290: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32);  add_125 = None
        mul_177: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_290, rsqrt_42);  convert_element_type_290 = None
        mul_234: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_177, convert_element_type_368)
        sum_18: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True);  mul_234 = None
        div_7: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_177, 768);  mul_177 = None
        mul_235: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_7, sum_18);  div_7 = sum_18 = None
        sub_13: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_368, mul_235);  convert_element_type_368 = mul_235 = None
        mul_236: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_42);  sub_13 = rsqrt_42 = None
        convert_element_type_370: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_236, torch.bfloat16);  mul_236 = None
        add_146: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_138, convert_element_type_370);  add_138 = convert_element_type_370 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:914 in forward, code: lambdas = self.scalars[1 * len(self.blocks) : 3 * len(self.blocks)].view(-1, 2)
        slice_10: "f32[24]" = torch.ops.aten.slice.Tensor(primals_7, 0, 12, 36)
        view_6: "f32[12, 2]" = torch.ops.aten.reshape.default(slice_10, [-1, 2]);  slice_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_76: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_79: "f32[]" = torch.ops.aten.select.int(select_76, 0, 1)
        mul_237: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_146, select_79);  select_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:909 in forward, code: x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        unsqueeze_42: "bf16[1, 6144, 768]" = torch.ops.aten.unsqueeze.default(embedding_3, 0);  embedding_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_10: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(unsqueeze_42, torch.float32);  unsqueeze_42 = None
        mul: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, rsqrt);  convert_element_type_10 = None
        convert_element_type_11: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_238: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_146, convert_element_type_11)
        sum_19: "bf16[]" = torch.ops.aten.sum.default(mul_238);  mul_238 = None
        convert_element_type_371: "f32[]" = torch.ops.prims.convert_element_type.default(sum_19, torch.float32);  sum_19 = None
        select_scatter_3: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_371, 0, 1);  convert_element_type_371 = None
        select_78: "f32[]" = torch.ops.aten.select.int(select_76, 0, 0);  select_76 = None
        mul_239: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_146, select_78);  select_78 = None
        mul_240: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_146, add_124);  add_146 = add_124 = None
        sum_20: "bf16[]" = torch.ops.aten.sum.default(mul_240);  mul_240 = None
        convert_element_type_372: "f32[]" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        select_scatter_4: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_372, 0, 0);  convert_element_type_372 = None
        add_147: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_3, select_scatter_4);  select_scatter_3 = select_scatter_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        full_default_25: "f32[12, 2]" = torch.ops.aten.full.default([12, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_5: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_139, 0, 11);  add_139 = None
        select_scatter_6: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_147, 0, 11);  add_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:913 in forward, code: skip_weights = self.scalars[: (len(self.blocks) // 2)]
        slice_9: "f32[6]" = torch.ops.aten.slice.Tensor(primals_7, 0, 0, 6);  primals_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_75: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 5)
        mul_241: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_239, select_75);  select_75 = None
        mul_242: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_239, add_12)
        sum_21: "bf16[]" = torch.ops.aten.sum.default(mul_242);  mul_242 = None
        convert_element_type_373: "f32[]" = torch.ops.prims.convert_element_type.default(sum_21, torch.float32);  sum_21 = None
        full_default_27: "f32[6]" = torch.ops.aten.full.default([6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_7: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_27, convert_element_type_373, 0, 5);  convert_element_type_373 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_205: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_239, [6144, 768])
        permute_141: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_205, [1, 0])
        mm_68: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_141, view_165);  permute_141 = view_165 = None
        mm_69: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_205, permute_143);  view_205 = permute_143 = None
        view_206: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_69, [1, 6144, 3072]);  mm_69 = None
        convert_element_type_378: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_164: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_50, [1, 6144, 3072]);  mm_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_10: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_164);  view_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_61: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_10, 1.0)
        mul_243: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_61, 2.0);  pow_61 = None
        mul_244: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_206, mul_243);  view_206 = mul_243 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_2: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_9: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_2, full_default_17, mul_244);  le_2 = mul_244 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_207: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_9, [6144, 3072]);  where_9 = None
        mm_70: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_145, view_207);  permute_145 = None
        mm_71: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_207, permute_146);  view_207 = permute_146 = None
        view_208: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_71, [1, 6144, 768]);  mm_71 = None
        permute_147: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
        convert_element_type_383: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_147, torch.float32);  permute_147 = None
        permute_148: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_383, [1, 0]);  convert_element_type_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_384: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        convert_element_type_282: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_121, torch.float32);  add_121 = None
        mul_173: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_282, rsqrt_41);  convert_element_type_282 = None
        mul_246: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_173, convert_element_type_384)
        sum_22: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_246, [2], True);  mul_246 = None
        div_8: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_173, 768);  mul_173 = None
        mul_247: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_8, sum_22);  div_8 = sum_22 = None
        sub_14: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_384, mul_247);  convert_element_type_384 = mul_247 = None
        mul_248: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_41);  sub_14 = rsqrt_41 = None
        convert_element_type_386: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        add_148: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_239, convert_element_type_386);  mul_239 = convert_element_type_386 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_209: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_148, [6144, 768])
        permute_149: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_209, [1, 0])
        mm_72: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_149, view_161);  permute_149 = view_161 = None
        mm_73: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_209, permute_151);  view_209 = permute_151 = None
        view_210: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_73, [1, 6144, 768]);  mm_73 = None
        convert_element_type_391: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        select_scatter_8: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_391, 0, 3);  convert_element_type_391 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_211: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_210, [1, 6144, 6, 128]);  view_210 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_101: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_109, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_249: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_211, permute_101);  permute_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_158: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_48, [1, 6144, 6]);  mm_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_9: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_158);  view_158 = None
        view_159: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_9, [1, 6144, 6, 1])
        mul_250: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_211, view_159);  view_211 = view_159 = None
        sum_23: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [3], True);  mul_249 = None
        view_212: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_23, [1, 6144, 6]);  sum_23 = None
        convert_element_type_392: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        convert_element_type_393: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_9, torch.float32);  sigmoid_9 = None
        sub_15: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_393)
        mul_251: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_393, sub_15);  convert_element_type_393 = sub_15 = None
        mul_252: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_392, mul_251);  convert_element_type_392 = mul_251 = None
        convert_element_type_394: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_213: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_394, [6144, 6]);  convert_element_type_394 = None
        permute_153: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_74: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_153, view_157);  permute_153 = view_157 = None
        mm_75: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_213, permute_155);  view_213 = permute_155 = None
        view_214: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_75, [1, 6144, 12]);  mm_75 = None
        convert_element_type_399: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_2: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_214, 2, 0, 12);  view_214 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_157: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_250, [0, 2, 1, 3]);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph1 = self.fw_graph1
        joint_graph1 = self.joint_graph1
        mask_graph1 = self.mask_graph1
        flex_attention_backward_1 = torch.ops.higher_order.flex_attention_backward(permute_98, permute_99, permute_100, getitem_109, getitem_110, permute_157, None, fw_graph1, joint_graph1, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph1), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_98 = permute_99 = permute_100 = getitem_109 = getitem_110 = permute_157 = fw_graph1 = joint_graph1 = mask_graph1 = None
        getitem_131: "bf16[1, 6, 6144, 128]" = flex_attention_backward_1[0]
        getitem_132: "bf16[1, 6, 6144, 128]" = flex_attention_backward_1[1]
        getitem_133: "bf16[1, 6, 6144, 128]" = flex_attention_backward_1[2];  flex_attention_backward_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_158: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_159: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_160: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3]);  getitem_131 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_69: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_73: "f32[]" = torch.ops.aten.select.int(select_69, 0, 1)
        mul_253: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_158, select_73);  select_73 = None
        view_28: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_1, [1, 6144, 6, 128]);  embedding_1 = None
        mul_254: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_158, view_28)
        sum_24: "bf16[]" = torch.ops.aten.sum.default(mul_254);  mul_254 = None
        convert_element_type_400: "f32[]" = torch.ops.prims.convert_element_type.default(sum_24, torch.float32);  sum_24 = None
        view_216: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_253, [6144, 768]);  mul_253 = None
        select_scatter_9: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_400, 0, 1);  convert_element_type_400 = None
        select_72: "f32[]" = torch.ops.aten.select.int(select_69, 0, 0);  select_69 = None
        mul_255: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_158, select_72);  select_72 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_153: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_47, [1, 6144, 2304]);  mm_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_154: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_153, [1, 6144, 18, 128]);  view_153 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_27 = torch.ops.aten.split.Tensor(view_154, 6, -2);  view_154 = None
        getitem_104: "bf16[1, 6144, 6, 128]" = split_27[2]
        getitem_103: "bf16[1, 6144, 6, 128]" = split_27[1]
        getitem_102: "bf16[1, 6144, 6, 128]" = split_27[0];  split_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        mul_256: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_158, getitem_104);  permute_158 = getitem_104 = None
        sum_25: "bf16[]" = torch.ops.aten.sum.default(mul_256);  mul_256 = None
        convert_element_type_401: "f32[]" = torch.ops.prims.convert_element_type.default(sum_25, torch.float32);  sum_25 = None
        select_scatter_10: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_401, 0, 0);  convert_element_type_401 = None
        add_149: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_9, select_scatter_10);  select_scatter_9 = select_scatter_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_402: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_159, torch.float32);  permute_159 = None
        slice_82: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_402, 3, 0, 64)
        slice_83: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_402, 3, 64, 128);  convert_element_type_402 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_115: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_65, 0);  primals_65 = None
        slice_67: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_115, 1, 0, 6144);  unsqueeze_115 = None
        unsqueeze_116: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_67, 2);  slice_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_257: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_83, unsqueeze_116)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_117: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_66, 0);  primals_66 = None
        slice_68: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_117, 1, 0, 6144);  unsqueeze_117 = None
        unsqueeze_118: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_68, 2);  slice_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_18: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_118)
        mul_258: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_83, neg_18);  slice_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_259: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_82, unsqueeze_118)
        add_150: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_257, mul_259);  mul_257 = mul_259 = None
        mul_260: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_82, unsqueeze_116);  slice_82 = None
        add_151: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_258, mul_260);  mul_258 = mul_260 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_25: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_151, add_150], 3);  add_151 = add_150 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_404: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_160, torch.float32);  permute_160 = None
        slice_84: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_404, 3, 0, 64)
        slice_85: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_404, 3, 64, 128);  convert_element_type_404 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_261: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_85, unsqueeze_116)
        mul_262: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_85, neg_18);  slice_85 = neg_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_263: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_84, unsqueeze_118);  unsqueeze_118 = None
        add_152: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_261, mul_263);  mul_261 = mul_263 = None
        mul_264: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_84, unsqueeze_116);  slice_84 = unsqueeze_116 = None
        add_153: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_262, mul_264);  mul_262 = mul_264 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_26: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_153, add_152], 3);  add_153 = add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_19: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_25, torch.float32);  cat_25 = None
        convert_element_type_270: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
        mul_161: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_270, rsqrt_40);  convert_element_type_270 = None
        mul_266: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_161, convert_element_type_default_19)
        sum_26: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [3], True);  mul_266 = None
        div_9: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_161, 128);  mul_161 = None
        mul_267: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_9, sum_26);  div_9 = sum_26 = None
        sub_16: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_19, mul_267);  convert_element_type_default_19 = mul_267 = None
        mul_268: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_40);  sub_16 = rsqrt_40 = None
        convert_element_type_408: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16);  mul_268 = None
        convert_element_type_default_18: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_26, torch.float32);  cat_26 = None
        convert_element_type_268: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_102, torch.float32);  getitem_102 = None
        mul_160: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_268, rsqrt_39);  convert_element_type_268 = None
        mul_270: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_160, convert_element_type_default_18)
        sum_27: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [3], True);  mul_270 = None
        div_10: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_160, 128);  mul_160 = None
        mul_271: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_10, sum_27);  div_10 = sum_27 = None
        sub_17: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_18, mul_271);  convert_element_type_default_18 = mul_271 = None
        mul_272: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_39);  sub_17 = rsqrt_39 = None
        convert_element_type_411: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_272, torch.bfloat16);  mul_272 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_27: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_411, convert_element_type_408, mul_255], 2);  convert_element_type_411 = convert_element_type_408 = mul_255 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_217: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_27, [1, 6144, 2304]);  cat_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_218: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_217, [6144, 2304]);  view_217 = None
        permute_161: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_76: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_161, view_152);  permute_161 = view_152 = None
        mm_77: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_218, permute_163);  view_218 = permute_163 = None
        view_219: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_77, [1, 6144, 768]);  mm_77 = None
        add_154: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_2, view_219);  slice_scatter_2 = view_219 = None
        convert_element_type_416: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        view_220: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_416, [3, 768, 768]);  convert_element_type_416 = None
        slice_scatter_3: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_220, 0, 0, 3);  view_220 = None
        add_155: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_8, slice_scatter_3);  select_scatter_8 = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_417: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_154, torch.float32);  add_154 = None
        convert_element_type_263: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_112, torch.float32);  add_112 = None
        mul_159: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_263, rsqrt_38);  convert_element_type_263 = None
        mul_274: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_159, convert_element_type_417)
        sum_28: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
        div_11: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_159, 768);  mul_159 = None
        mul_275: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_11, sum_28);  div_11 = sum_28 = None
        sub_18: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_417, mul_275);  convert_element_type_417 = mul_275 = None
        mul_276: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_38);  sub_18 = rsqrt_38 = None
        convert_element_type_419: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_276, torch.bfloat16);  mul_276 = None
        add_156: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_148, convert_element_type_419);  add_148 = convert_element_type_419 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_68: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_71: "f32[]" = torch.ops.aten.select.int(select_68, 0, 1)
        mul_277: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_156, select_71);  select_71 = None
        mul_278: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_156, convert_element_type_11)
        sum_29: "bf16[]" = torch.ops.aten.sum.default(mul_278);  mul_278 = None
        convert_element_type_420: "f32[]" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
        add_157: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_237, mul_277);  mul_237 = mul_277 = None
        select_scatter_11: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_420, 0, 1);  convert_element_type_420 = None
        select_70: "f32[]" = torch.ops.aten.select.int(select_68, 0, 0);  select_68 = None
        mul_279: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_156, select_70);  select_70 = None
        mul_280: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_156, add_111);  add_156 = add_111 = None
        sum_30: "bf16[]" = torch.ops.aten.sum.default(mul_280);  mul_280 = None
        convert_element_type_421: "f32[]" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
        select_scatter_12: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_421, 0, 0);  convert_element_type_421 = None
        add_158: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_11, select_scatter_12);  select_scatter_11 = select_scatter_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_13: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_149, 0, 10);  add_149 = None
        add_159: "f32[12, 2]" = torch.ops.aten.add.Tensor(select_scatter_5, select_scatter_13);  select_scatter_5 = select_scatter_13 = None
        select_scatter_14: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_158, 0, 10);  add_158 = None
        add_160: "f32[12, 2]" = torch.ops.aten.add.Tensor(select_scatter_6, select_scatter_14);  select_scatter_6 = select_scatter_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_67: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 4)
        mul_281: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_279, select_67);  select_67 = None
        mul_282: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_279, add_24)
        sum_31: "bf16[]" = torch.ops.aten.sum.default(mul_282);  mul_282 = None
        convert_element_type_422: "f32[]" = torch.ops.prims.convert_element_type.default(sum_31, torch.float32);  sum_31 = None
        select_scatter_15: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_27, convert_element_type_422, 0, 4);  convert_element_type_422 = None
        add_161: "f32[6]" = torch.ops.aten.add.Tensor(select_scatter_7, select_scatter_15);  select_scatter_7 = select_scatter_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_221: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_279, [6144, 768])
        permute_165: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_78: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_165, view_149);  permute_165 = view_149 = None
        mm_79: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_221, permute_167);  view_221 = permute_167 = None
        view_222: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_79, [1, 6144, 3072]);  mm_79 = None
        convert_element_type_427: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_148: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_45, [1, 6144, 3072]);  mm_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_9: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_148);  view_148 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_62: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_9, 1.0)
        mul_283: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
        mul_284: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_222, mul_283);  view_222 = mul_283 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_3: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_10: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_3, full_default_17, mul_284);  le_3 = mul_284 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_223: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_10, [6144, 3072]);  where_10 = None
        mm_80: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_169, view_223);  permute_169 = None
        mm_81: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_223, permute_170);  view_223 = permute_170 = None
        view_224: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_81, [1, 6144, 768]);  mm_81 = None
        permute_171: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
        convert_element_type_432: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_171, torch.float32);  permute_171 = None
        permute_172: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_432, [1, 0]);  convert_element_type_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_433: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_224, torch.float32);  view_224 = None
        convert_element_type_255: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_108, torch.float32);  add_108 = None
        mul_155: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_255, rsqrt_37);  convert_element_type_255 = None
        mul_286: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_155, convert_element_type_433)
        sum_32: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True);  mul_286 = None
        div_12: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_155, 768);  mul_155 = None
        mul_287: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_12, sum_32);  div_12 = sum_32 = None
        sub_19: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_433, mul_287);  convert_element_type_433 = mul_287 = None
        mul_288: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_37);  sub_19 = rsqrt_37 = None
        convert_element_type_435: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None
        add_162: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_279, convert_element_type_435);  mul_279 = convert_element_type_435 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_225: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_162, [6144, 768])
        permute_173: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_82: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_173, view_145);  permute_173 = view_145 = None
        mm_83: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_225, permute_175);  view_225 = permute_175 = None
        view_226: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_83, [1, 6144, 768]);  mm_83 = None
        convert_element_type_440: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        select_scatter_16: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_440, 0, 3);  convert_element_type_440 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_227: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_226, [1, 6144, 6, 128]);  view_226 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_91: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_289: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_227, permute_91);  permute_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_142: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_43, [1, 6144, 6]);  mm_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_8: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_142);  view_142 = None
        view_143: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_8, [1, 6144, 6, 1])
        mul_290: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_227, view_143);  view_227 = view_143 = None
        sum_33: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [3], True);  mul_289 = None
        view_228: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_33, [1, 6144, 6]);  sum_33 = None
        convert_element_type_441: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        convert_element_type_442: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_8, torch.float32);  sigmoid_8 = None
        sub_20: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_442)
        mul_291: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_442, sub_20);  convert_element_type_442 = sub_20 = None
        mul_292: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_441, mul_291);  convert_element_type_441 = mul_291 = None
        convert_element_type_443: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_292, torch.bfloat16);  mul_292 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_229: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_443, [6144, 6]);  convert_element_type_443 = None
        permute_177: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_229, [1, 0])
        mm_84: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_177, view_141);  permute_177 = view_141 = None
        mm_85: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_229, permute_179);  view_229 = permute_179 = None
        view_230: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_85, [1, 6144, 12]);  mm_85 = None
        convert_element_type_448: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_4: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_230, 2, 0, 12);  view_230 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_181: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_290, [0, 2, 1, 3]);  mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph2 = self.fw_graph2
        joint_graph2 = self.joint_graph2
        mask_graph2 = self.mask_graph2
        flex_attention_backward_2 = torch.ops.higher_order.flex_attention_backward(permute_88, permute_89, permute_90, getitem_99, getitem_100, permute_181, None, fw_graph2, joint_graph2, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph2), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_88 = permute_89 = permute_90 = getitem_99 = getitem_100 = permute_181 = fw_graph2 = joint_graph2 = mask_graph2 = None
        getitem_135: "bf16[1, 6, 6144, 128]" = flex_attention_backward_2[0]
        getitem_136: "bf16[1, 6, 6144, 128]" = flex_attention_backward_2[1]
        getitem_137: "bf16[1, 6, 6144, 128]" = flex_attention_backward_2[2];  flex_attention_backward_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_182: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_183: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3]);  getitem_136 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_184: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_61: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_65: "f32[]" = torch.ops.aten.select.int(select_61, 0, 1)
        mul_293: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_182, select_65);  select_65 = None
        view_12: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding, [1, 6144, 6, 128]);  embedding = None
        mul_294: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_182, view_12)
        sum_34: "bf16[]" = torch.ops.aten.sum.default(mul_294);  mul_294 = None
        convert_element_type_449: "f32[]" = torch.ops.prims.convert_element_type.default(sum_34, torch.float32);  sum_34 = None
        view_232: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_293, [6144, 768]);  mul_293 = None
        select_scatter_17: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_449, 0, 1);  convert_element_type_449 = None
        select_64: "f32[]" = torch.ops.aten.select.int(select_61, 0, 0);  select_61 = None
        mul_295: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_182, select_64);  select_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_137: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_42, [1, 6144, 2304]);  mm_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_138: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_137, [1, 6144, 18, 128]);  view_137 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_24 = torch.ops.aten.split.Tensor(view_138, 6, -2);  view_138 = None
        getitem_94: "bf16[1, 6144, 6, 128]" = split_24[2]
        getitem_93: "bf16[1, 6144, 6, 128]" = split_24[1]
        getitem_92: "bf16[1, 6144, 6, 128]" = split_24[0];  split_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        mul_296: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_182, getitem_94);  permute_182 = getitem_94 = None
        sum_35: "bf16[]" = torch.ops.aten.sum.default(mul_296);  mul_296 = None
        convert_element_type_450: "f32[]" = torch.ops.prims.convert_element_type.default(sum_35, torch.float32);  sum_35 = None
        select_scatter_18: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_450, 0, 0);  convert_element_type_450 = None
        add_163: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_17, select_scatter_18);  select_scatter_17 = select_scatter_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_451: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_183, torch.float32);  permute_183 = None
        slice_86: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_451, 3, 0, 64)
        slice_87: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_451, 3, 64, 128);  convert_element_type_451 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_107: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_59, 0);  primals_59 = None
        slice_61: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_107, 1, 0, 6144);  unsqueeze_107 = None
        unsqueeze_108: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_61, 2);  slice_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_297: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_87, unsqueeze_108)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_109: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_60, 0);  primals_60 = None
        slice_62: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_109, 1, 0, 6144);  unsqueeze_109 = None
        unsqueeze_110: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_62, 2);  slice_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_16: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_110)
        mul_298: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_87, neg_16);  slice_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_299: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_86, unsqueeze_110)
        add_164: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_297, mul_299);  mul_297 = mul_299 = None
        mul_300: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_86, unsqueeze_108);  slice_86 = None
        add_165: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_298, mul_300);  mul_298 = mul_300 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_28: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_165, add_164], 3);  add_165 = add_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_453: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_184, torch.float32);  permute_184 = None
        slice_88: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_453, 3, 0, 64)
        slice_89: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_453, 3, 64, 128);  convert_element_type_453 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_301: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_89, unsqueeze_108)
        mul_302: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_89, neg_16);  slice_89 = neg_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_303: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_88, unsqueeze_110);  unsqueeze_110 = None
        add_166: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_301, mul_303);  mul_301 = mul_303 = None
        mul_304: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_88, unsqueeze_108);  slice_88 = unsqueeze_108 = None
        add_167: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_302, mul_304);  mul_302 = mul_304 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_29: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_167, add_166], 3);  add_167 = add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_17: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_28, torch.float32);  cat_28 = None
        convert_element_type_243: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None
        mul_143: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_243, rsqrt_36);  convert_element_type_243 = None
        mul_306: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_143, convert_element_type_default_17)
        sum_36: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [3], True);  mul_306 = None
        div_13: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_143, 128);  mul_143 = None
        mul_307: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_13, sum_36);  div_13 = sum_36 = None
        sub_21: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_17, mul_307);  convert_element_type_default_17 = mul_307 = None
        mul_308: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_36);  sub_21 = rsqrt_36 = None
        convert_element_type_457: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_308, torch.bfloat16);  mul_308 = None
        convert_element_type_default_16: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_29, torch.float32);  cat_29 = None
        convert_element_type_241: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_92, torch.float32);  getitem_92 = None
        mul_142: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_241, rsqrt_35);  convert_element_type_241 = None
        mul_310: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_142, convert_element_type_default_16)
        sum_37: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [3], True);  mul_310 = None
        div_14: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_142, 128);  mul_142 = None
        mul_311: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_14, sum_37);  div_14 = sum_37 = None
        sub_22: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_16, mul_311);  convert_element_type_default_16 = mul_311 = None
        mul_312: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_35);  sub_22 = rsqrt_35 = None
        convert_element_type_460: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_30: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_460, convert_element_type_457, mul_295], 2);  convert_element_type_460 = convert_element_type_457 = mul_295 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_233: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_30, [1, 6144, 2304]);  cat_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_234: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_233, [6144, 2304]);  view_233 = None
        permute_185: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_86: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_185, view_136);  permute_185 = view_136 = None
        mm_87: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_234, permute_187);  view_234 = permute_187 = None
        view_235: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_87, [1, 6144, 768]);  mm_87 = None
        add_168: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_4, view_235);  slice_scatter_4 = view_235 = None
        convert_element_type_465: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        view_236: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_465, [3, 768, 768]);  convert_element_type_465 = None
        slice_scatter_5: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_236, 0, 0, 3);  view_236 = None
        add_169: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_16, slice_scatter_5);  select_scatter_16 = slice_scatter_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_466: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_168, torch.float32);  add_168 = None
        convert_element_type_236: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32);  add_99 = None
        mul_141: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_236, rsqrt_34);  convert_element_type_236 = None
        mul_314: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_141, convert_element_type_466)
        sum_38: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
        div_15: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_141, 768);  mul_141 = None
        mul_315: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_15, sum_38);  div_15 = sum_38 = None
        sub_23: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_466, mul_315);  convert_element_type_466 = mul_315 = None
        mul_316: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_34);  sub_23 = rsqrt_34 = None
        convert_element_type_468: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_316, torch.bfloat16);  mul_316 = None
        add_170: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_162, convert_element_type_468);  add_162 = convert_element_type_468 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_60: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_63: "f32[]" = torch.ops.aten.select.int(select_60, 0, 1)
        mul_317: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_170, select_63);  select_63 = None
        mul_318: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_170, convert_element_type_11)
        sum_39: "bf16[]" = torch.ops.aten.sum.default(mul_318);  mul_318 = None
        convert_element_type_469: "f32[]" = torch.ops.prims.convert_element_type.default(sum_39, torch.float32);  sum_39 = None
        add_171: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_157, mul_317);  add_157 = mul_317 = None
        select_scatter_19: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_469, 0, 1);  convert_element_type_469 = None
        select_62: "f32[]" = torch.ops.aten.select.int(select_60, 0, 0);  select_60 = None
        mul_319: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_170, select_62);  select_62 = None
        mul_320: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_170, add_98);  add_170 = add_98 = None
        sum_40: "bf16[]" = torch.ops.aten.sum.default(mul_320);  mul_320 = None
        convert_element_type_470: "f32[]" = torch.ops.prims.convert_element_type.default(sum_40, torch.float32);  sum_40 = None
        select_scatter_20: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_470, 0, 0);  convert_element_type_470 = None
        add_172: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_19, select_scatter_20);  select_scatter_19 = select_scatter_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_21: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_163, 0, 9);  add_163 = None
        add_173: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_159, select_scatter_21);  add_159 = select_scatter_21 = None
        select_scatter_22: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_172, 0, 9);  add_172 = None
        add_174: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_160, select_scatter_22);  add_160 = select_scatter_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_59: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 3)
        mul_321: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_319, select_59);  select_59 = None
        mul_322: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_319, add_36)
        sum_41: "bf16[]" = torch.ops.aten.sum.default(mul_322);  mul_322 = None
        convert_element_type_471: "f32[]" = torch.ops.prims.convert_element_type.default(sum_41, torch.float32);  sum_41 = None
        select_scatter_23: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_27, convert_element_type_471, 0, 3);  convert_element_type_471 = None
        add_175: "f32[6]" = torch.ops.aten.add.Tensor(add_161, select_scatter_23);  add_161 = select_scatter_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_237: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_319, [6144, 768])
        permute_189: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_88: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_189, view_133);  permute_189 = view_133 = None
        mm_89: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_237, permute_191);  view_237 = permute_191 = None
        view_238: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_89, [1, 6144, 3072]);  mm_89 = None
        convert_element_type_476: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_132: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_40, [1, 6144, 3072]);  mm_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_8: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_132);  view_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_63: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_8, 1.0)
        mul_323: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_63, 2.0);  pow_63 = None
        mul_324: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_238, mul_323);  view_238 = mul_323 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_4: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_11: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_4, full_default_17, mul_324);  le_4 = mul_324 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_239: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_11, [6144, 3072]);  where_11 = None
        mm_90: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_193, view_239);  permute_193 = None
        mm_91: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_239, permute_194);  view_239 = permute_194 = None
        view_240: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_91, [1, 6144, 768]);  mm_91 = None
        permute_195: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
        convert_element_type_481: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_195, torch.float32);  permute_195 = None
        permute_196: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_481, [1, 0]);  convert_element_type_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_482: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_240, torch.float32);  view_240 = None
        convert_element_type_228: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_95, torch.float32);  add_95 = None
        mul_137: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_228, rsqrt_33);  convert_element_type_228 = None
        mul_326: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_137, convert_element_type_482)
        sum_42: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
        div_16: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_137, 768);  mul_137 = None
        mul_327: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_16, sum_42);  div_16 = sum_42 = None
        sub_24: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_482, mul_327);  convert_element_type_482 = mul_327 = None
        mul_328: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_33);  sub_24 = rsqrt_33 = None
        convert_element_type_484: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None
        add_176: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_319, convert_element_type_484);  mul_319 = convert_element_type_484 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_241: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_176, [6144, 768])
        permute_197: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_241, [1, 0])
        mm_92: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_197, view_129);  permute_197 = view_129 = None
        mm_93: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_241, permute_199);  view_241 = permute_199 = None
        view_242: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_93, [1, 6144, 768]);  mm_93 = None
        convert_element_type_489: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        select_scatter_24: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_489, 0, 3);  convert_element_type_489 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_243: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_242, [1, 6144, 6, 128]);  view_242 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_81: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_329: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_243, permute_81);  permute_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_126: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_38, [1, 6144, 6]);  mm_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_7: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_126);  view_126 = None
        view_127: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_7, [1, 6144, 6, 1])
        mul_330: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_243, view_127);  view_243 = view_127 = None
        sum_43: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [3], True);  mul_329 = None
        view_244: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_43, [1, 6144, 6]);  sum_43 = None
        convert_element_type_490: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_244, torch.float32);  view_244 = None
        convert_element_type_491: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_7, torch.float32);  sigmoid_7 = None
        sub_25: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_491)
        mul_331: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_491, sub_25);  convert_element_type_491 = sub_25 = None
        mul_332: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_490, mul_331);  convert_element_type_490 = mul_331 = None
        convert_element_type_492: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16);  mul_332 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_245: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_492, [6144, 6]);  convert_element_type_492 = None
        permute_201: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_245, [1, 0])
        mm_94: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_201, view_125);  permute_201 = view_125 = None
        mm_95: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_245, permute_203);  view_245 = permute_203 = None
        view_246: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_95, [1, 6144, 12]);  mm_95 = None
        convert_element_type_497: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_6: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_246, 2, 0, 12);  view_246 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_205: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_330, [0, 2, 1, 3]);  mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph3 = self.fw_graph3
        joint_graph3 = self.joint_graph3
        mask_graph3 = self.mask_graph3
        flex_attention_backward_3 = torch.ops.higher_order.flex_attention_backward(permute_78, permute_79, permute_80, getitem_89, getitem_90, permute_205, None, fw_graph3, joint_graph3, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph3), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_78 = permute_79 = permute_80 = getitem_89 = getitem_90 = permute_205 = fw_graph3 = joint_graph3 = mask_graph3 = None
        getitem_139: "bf16[1, 6, 6144, 128]" = flex_attention_backward_3[0]
        getitem_140: "bf16[1, 6, 6144, 128]" = flex_attention_backward_3[1]
        getitem_141: "bf16[1, 6, 6144, 128]" = flex_attention_backward_3[2];  flex_attention_backward_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_206: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1, 3]);  getitem_141 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_207: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_208: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_54: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 8)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_57: "f32[]" = torch.ops.aten.select.int(select_54, 0, 0);  select_54 = None
        mul_333: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_206, select_57);  select_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_122: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_37, [1, 6144, 2304]);  mm_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_123: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_122, [1, 6144, 18, 128]);  view_122 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_21 = torch.ops.aten.split.Tensor(view_123, 6, -2);  view_123 = None
        getitem_84: "bf16[1, 6144, 6, 128]" = split_21[2]
        getitem_83: "bf16[1, 6144, 6, 128]" = split_21[1]
        getitem_82: "bf16[1, 6144, 6, 128]" = split_21[0];  split_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_334: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_206, getitem_84);  permute_206 = getitem_84 = None
        sum_44: "bf16[]" = torch.ops.aten.sum.default(mul_334);  mul_334 = None
        convert_element_type_498: "f32[]" = torch.ops.prims.convert_element_type.default(sum_44, torch.float32);  sum_44 = None
        select_scatter_25: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_498, 0, 0);  convert_element_type_498 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_499: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_207, torch.float32);  permute_207 = None
        slice_90: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_499, 3, 0, 64)
        slice_91: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_499, 3, 64, 128);  convert_element_type_499 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_99: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_53, 0);  primals_53 = None
        slice_55: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_99, 1, 0, 6144);  unsqueeze_99 = None
        unsqueeze_100: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_55, 2);  slice_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_335: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_91, unsqueeze_100)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_101: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_54, 0);  primals_54 = None
        slice_56: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_101, 1, 0, 6144);  unsqueeze_101 = None
        unsqueeze_102: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_56, 2);  slice_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_14: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_102)
        mul_336: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_91, neg_14);  slice_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_337: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_90, unsqueeze_102)
        add_177: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_335, mul_337);  mul_335 = mul_337 = None
        mul_338: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_90, unsqueeze_100);  slice_90 = None
        add_178: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_336, mul_338);  mul_336 = mul_338 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_31: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_178, add_177], 3);  add_178 = add_177 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_501: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_208, torch.float32);  permute_208 = None
        slice_92: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_501, 3, 0, 64)
        slice_93: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_501, 3, 64, 128);  convert_element_type_501 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_339: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_93, unsqueeze_100)
        mul_340: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_93, neg_14);  slice_93 = neg_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_341: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_92, unsqueeze_102);  unsqueeze_102 = None
        add_179: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_339, mul_341);  mul_339 = mul_341 = None
        mul_342: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_92, unsqueeze_100);  slice_92 = unsqueeze_100 = None
        add_180: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_340, mul_342);  mul_340 = mul_342 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_32: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_180, add_179], 3);  add_180 = add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_15: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_31, torch.float32);  cat_31 = None
        convert_element_type_216: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
        mul_126: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_216, rsqrt_32);  convert_element_type_216 = None
        mul_344: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_126, convert_element_type_default_15)
        sum_45: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [3], True);  mul_344 = None
        div_17: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_126, 128);  mul_126 = None
        mul_345: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_17, sum_45);  div_17 = sum_45 = None
        sub_26: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_15, mul_345);  convert_element_type_default_15 = mul_345 = None
        mul_346: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_32);  sub_26 = rsqrt_32 = None
        convert_element_type_505: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        convert_element_type_default_14: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_32, torch.float32);  cat_32 = None
        convert_element_type_214: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_82, torch.float32);  getitem_82 = None
        mul_125: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_214, rsqrt_31);  convert_element_type_214 = None
        mul_348: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_125, convert_element_type_default_14)
        sum_46: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [3], True);  mul_348 = None
        div_18: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_125, 128);  mul_125 = None
        mul_349: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_18, sum_46);  div_18 = sum_46 = None
        sub_27: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_14, mul_349);  convert_element_type_default_14 = mul_349 = None
        mul_350: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_31);  sub_27 = rsqrt_31 = None
        convert_element_type_508: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_350, torch.bfloat16);  mul_350 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_33: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_508, convert_element_type_505, mul_333], 2);  convert_element_type_508 = convert_element_type_505 = mul_333 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_248: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_33, [1, 6144, 2304]);  cat_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_249: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_248, [6144, 2304]);  view_248 = None
        permute_209: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_249, [1, 0])
        mm_96: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_209, view_121);  permute_209 = view_121 = None
        mm_97: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_249, permute_211);  view_249 = permute_211 = None
        view_250: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_97, [1, 6144, 768]);  mm_97 = None
        add_181: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_6, view_250);  slice_scatter_6 = view_250 = None
        convert_element_type_513: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        view_251: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_513, [3, 768, 768]);  convert_element_type_513 = None
        slice_scatter_7: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_251, 0, 0, 3);  view_251 = None
        add_182: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_24, slice_scatter_7);  select_scatter_24 = slice_scatter_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_514: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_181, torch.float32);  add_181 = None
        convert_element_type_209: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        mul_124: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_209, rsqrt_30);  convert_element_type_209 = None
        mul_352: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_124, convert_element_type_514)
        sum_47: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        div_19: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_124, 768);  mul_124 = None
        mul_353: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_19, sum_47);  div_19 = sum_47 = None
        sub_28: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_514, mul_353);  convert_element_type_514 = mul_353 = None
        mul_354: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_30);  sub_28 = rsqrt_30 = None
        convert_element_type_516: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        add_183: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_176, convert_element_type_516);  add_176 = convert_element_type_516 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_53: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 8)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_56: "f32[]" = torch.ops.aten.select.int(select_53, 0, 1)
        mul_355: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_183, select_56);  select_56 = None
        mul_356: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_183, convert_element_type_11)
        sum_48: "bf16[]" = torch.ops.aten.sum.default(mul_356);  mul_356 = None
        convert_element_type_517: "f32[]" = torch.ops.prims.convert_element_type.default(sum_48, torch.float32);  sum_48 = None
        add_184: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_171, mul_355);  add_171 = mul_355 = None
        select_scatter_26: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_517, 0, 1);  convert_element_type_517 = None
        select_55: "f32[]" = torch.ops.aten.select.int(select_53, 0, 0);  select_53 = None
        mul_357: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_183, select_55);  select_55 = None
        mul_358: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_183, add_86);  add_183 = add_86 = None
        sum_49: "bf16[]" = torch.ops.aten.sum.default(mul_358);  mul_358 = None
        convert_element_type_518: "f32[]" = torch.ops.prims.convert_element_type.default(sum_49, torch.float32);  sum_49 = None
        select_scatter_27: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_518, 0, 0);  convert_element_type_518 = None
        add_185: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_26, select_scatter_27);  select_scatter_26 = select_scatter_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_28: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, select_scatter_25, 0, 8);  select_scatter_25 = None
        add_186: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_173, select_scatter_28);  add_173 = select_scatter_28 = None
        select_scatter_29: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_185, 0, 8);  add_185 = None
        add_187: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_174, select_scatter_29);  add_174 = select_scatter_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_52: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 2)
        mul_359: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_357, select_52);  select_52 = None
        mul_360: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_357, add_47)
        sum_50: "bf16[]" = torch.ops.aten.sum.default(mul_360);  mul_360 = None
        convert_element_type_519: "f32[]" = torch.ops.prims.convert_element_type.default(sum_50, torch.float32);  sum_50 = None
        select_scatter_30: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_27, convert_element_type_519, 0, 2);  convert_element_type_519 = None
        add_188: "f32[6]" = torch.ops.aten.add.Tensor(add_175, select_scatter_30);  add_175 = select_scatter_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_252: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_357, [6144, 768])
        permute_213: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_252, [1, 0])
        mm_98: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_213, view_118);  permute_213 = view_118 = None
        mm_99: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_252, permute_215);  view_252 = permute_215 = None
        view_253: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_99, [1, 6144, 3072]);  mm_99 = None
        convert_element_type_524: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_117: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_35, [1, 6144, 3072]);  mm_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_7: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_117);  view_117 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_64: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_7, 1.0)
        mul_361: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_362: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_253, mul_361);  view_253 = mul_361 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_5: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_12: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_5, full_default_17, mul_362);  le_5 = mul_362 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_254: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_12, [6144, 3072]);  where_12 = None
        mm_100: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_217, view_254);  permute_217 = None
        mm_101: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_254, permute_218);  view_254 = permute_218 = None
        view_255: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_101, [1, 6144, 768]);  mm_101 = None
        permute_219: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
        convert_element_type_529: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_219, torch.float32);  permute_219 = None
        permute_220: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_529, [1, 0]);  convert_element_type_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_530: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_255, torch.float32);  view_255 = None
        convert_element_type_201: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        mul_120: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_201, rsqrt_29);  convert_element_type_201 = None
        mul_364: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_120, convert_element_type_530)
        sum_51: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True);  mul_364 = None
        div_20: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_120, 768);  mul_120 = None
        mul_365: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_20, sum_51);  div_20 = sum_51 = None
        sub_29: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_530, mul_365);  convert_element_type_530 = mul_365 = None
        mul_366: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        convert_element_type_532: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_366, torch.bfloat16);  mul_366 = None
        add_189: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_357, convert_element_type_532);  mul_357 = convert_element_type_532 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_49: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 7)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_51: "f32[]" = torch.ops.aten.select.int(select_49, 0, 1)
        mul_367: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_189, select_51);  select_51 = None
        mul_368: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_189, convert_element_type_11)
        sum_52: "bf16[]" = torch.ops.aten.sum.default(mul_368);  mul_368 = None
        convert_element_type_533: "f32[]" = torch.ops.prims.convert_element_type.default(sum_52, torch.float32);  sum_52 = None
        add_190: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_184, mul_367);  add_184 = mul_367 = None
        select_scatter_31: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_533, 0, 1);  convert_element_type_533 = None
        select_50: "f32[]" = torch.ops.aten.select.int(select_49, 0, 0);  select_49 = None
        mul_369: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_189, select_50);  select_50 = None
        mul_370: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_189, add_82);  add_189 = add_82 = None
        sum_53: "bf16[]" = torch.ops.aten.sum.default(mul_370);  mul_370 = None
        convert_element_type_534: "f32[]" = torch.ops.prims.convert_element_type.default(sum_53, torch.float32);  sum_53 = None
        select_scatter_32: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_534, 0, 0);  convert_element_type_534 = None
        add_191: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_31, select_scatter_32);  select_scatter_31 = select_scatter_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_33: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_191, 0, 7);  add_191 = None
        add_192: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_187, select_scatter_33);  add_187 = select_scatter_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_48: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 1)
        mul_371: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_369, select_48);  select_48 = None
        mul_372: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_369, add_58)
        sum_54: "bf16[]" = torch.ops.aten.sum.default(mul_372);  mul_372 = None
        convert_element_type_535: "f32[]" = torch.ops.prims.convert_element_type.default(sum_54, torch.float32);  sum_54 = None
        select_scatter_34: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_27, convert_element_type_535, 0, 1);  convert_element_type_535 = None
        add_193: "f32[6]" = torch.ops.aten.add.Tensor(add_188, select_scatter_34);  add_188 = select_scatter_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_256: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_369, [6144, 768])
        permute_221: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_102: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_221, view_114);  permute_221 = view_114 = None
        mm_103: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_256, permute_223);  view_256 = permute_223 = None
        view_257: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_103, [1, 6144, 3072]);  mm_103 = None
        convert_element_type_540: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_113: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_33, [1, 6144, 3072]);  mm_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_6: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_113);  view_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_65: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_6, 1.0)
        mul_373: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_65, 2.0);  pow_65 = None
        mul_374: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_257, mul_373);  view_257 = mul_373 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_6: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_13: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_6, full_default_17, mul_374);  le_6 = mul_374 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_258: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_13, [6144, 3072]);  where_13 = None
        mm_104: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_225, view_258);  permute_225 = None
        mm_105: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_258, permute_226);  view_258 = permute_226 = None
        view_259: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_105, [1, 6144, 768]);  mm_105 = None
        permute_227: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
        convert_element_type_545: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_227, torch.float32);  permute_227 = None
        permute_228: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_545, [1, 0]);  convert_element_type_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_546: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_259, torch.float32);  view_259 = None
        convert_element_type_193: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        mul_116: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_193, rsqrt_28);  convert_element_type_193 = None
        mul_376: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_116, convert_element_type_546)
        sum_55: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True);  mul_376 = None
        div_21: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_116, 768);  mul_116 = None
        mul_377: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_21, sum_55);  div_21 = sum_55 = None
        sub_30: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_546, mul_377);  convert_element_type_546 = mul_377 = None
        mul_378: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = rsqrt_28 = None
        convert_element_type_548: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None
        add_194: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_369, convert_element_type_548);  mul_369 = convert_element_type_548 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_260: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_194, [6144, 768])
        permute_229: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_260, [1, 0])
        mm_106: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_229, view_110);  permute_229 = view_110 = None
        mm_107: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_260, permute_231);  view_260 = permute_231 = None
        view_261: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_107, [1, 6144, 768]);  mm_107 = None
        convert_element_type_553: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        select_scatter_35: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_553, 0, 3);  convert_element_type_553 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_262: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_261, [1, 6144, 6, 128]);  view_261 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_68: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_379: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_262, permute_68);  permute_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_107: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_31, [1, 6144, 6]);  mm_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_6: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_107);  view_107 = None
        view_108: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_6, [1, 6144, 6, 1])
        mul_380: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_262, view_108);  view_262 = view_108 = None
        sum_56: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [3], True);  mul_379 = None
        view_263: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_56, [1, 6144, 6]);  sum_56 = None
        convert_element_type_554: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        convert_element_type_555: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_6, torch.float32);  sigmoid_6 = None
        sub_31: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_555)
        mul_381: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_555, sub_31);  convert_element_type_555 = sub_31 = None
        mul_382: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_554, mul_381);  convert_element_type_554 = mul_381 = None
        convert_element_type_556: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_382, torch.bfloat16);  mul_382 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_264: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_556, [6144, 6]);  convert_element_type_556 = None
        permute_233: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_264, [1, 0])
        mm_108: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_233, view_106);  permute_233 = view_106 = None
        mm_109: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_264, permute_235);  view_264 = permute_235 = None
        view_265: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_109, [1, 6144, 12]);  mm_109 = None
        convert_element_type_561: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_8: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_265, 2, 0, 12);  view_265 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_237: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_380, [0, 2, 1, 3]);  mul_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph4 = self.fw_graph4
        joint_graph4 = self.joint_graph4
        mask_graph4 = self.mask_graph4
        flex_attention_backward_4 = torch.ops.higher_order.flex_attention_backward(permute_65, permute_66, permute_67, getitem_79, getitem_80, permute_237, None, fw_graph4, joint_graph4, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph4), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_65 = permute_66 = permute_67 = getitem_79 = getitem_80 = permute_237 = fw_graph4 = joint_graph4 = mask_graph4 = None
        getitem_143: "bf16[1, 6, 6144, 128]" = flex_attention_backward_4[0]
        getitem_144: "bf16[1, 6, 6144, 128]" = flex_attention_backward_4[1]
        getitem_145: "bf16[1, 6, 6144, 128]" = flex_attention_backward_4[2];  flex_attention_backward_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_238: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_239: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_240: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_143, [0, 2, 1, 3]);  getitem_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_43: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 6)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_46: "f32[]" = torch.ops.aten.select.int(select_43, 0, 0);  select_43 = None
        mul_383: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_238, select_46);  select_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_103: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_30, [1, 6144, 2304]);  mm_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_104: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_103, [1, 6144, 18, 128]);  view_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_18 = torch.ops.aten.split.Tensor(view_104, 6, -2);  view_104 = None
        getitem_74: "bf16[1, 6144, 6, 128]" = split_18[2]
        getitem_73: "bf16[1, 6144, 6, 128]" = split_18[1]
        getitem_72: "bf16[1, 6144, 6, 128]" = split_18[0];  split_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_384: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_238, getitem_74);  permute_238 = getitem_74 = None
        sum_57: "bf16[]" = torch.ops.aten.sum.default(mul_384);  mul_384 = None
        convert_element_type_562: "f32[]" = torch.ops.prims.convert_element_type.default(sum_57, torch.float32);  sum_57 = None
        select_scatter_36: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_562, 0, 0);  convert_element_type_562 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_563: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_239, torch.float32);  permute_239 = None
        slice_94: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_563, 3, 0, 64)
        slice_95: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_563, 3, 64, 128);  convert_element_type_563 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_91: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_45, 0);  primals_45 = None
        slice_49: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_91, 1, 0, 6144);  unsqueeze_91 = None
        unsqueeze_92: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_49, 2);  slice_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_385: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_95, unsqueeze_92)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_93: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_46, 0);  primals_46 = None
        slice_50: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_93, 1, 0, 6144);  unsqueeze_93 = None
        unsqueeze_94: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_50, 2);  slice_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_12: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_94)
        mul_386: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_95, neg_12);  slice_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_387: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_94, unsqueeze_94)
        add_195: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_385, mul_387);  mul_385 = mul_387 = None
        mul_388: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_94, unsqueeze_92);  slice_94 = None
        add_196: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_386, mul_388);  mul_386 = mul_388 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_34: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_196, add_195], 3);  add_196 = add_195 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_565: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_240, torch.float32);  permute_240 = None
        slice_96: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_565, 3, 0, 64)
        slice_97: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_565, 3, 64, 128);  convert_element_type_565 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_389: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_97, unsqueeze_92)
        mul_390: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_97, neg_12);  slice_97 = neg_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_391: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_96, unsqueeze_94);  unsqueeze_94 = None
        add_197: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_389, mul_391);  mul_389 = mul_391 = None
        mul_392: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_96, unsqueeze_92);  slice_96 = unsqueeze_92 = None
        add_198: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_390, mul_392);  mul_390 = mul_392 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_35: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_198, add_197], 3);  add_198 = add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_13: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_34, torch.float32);  cat_34 = None
        convert_element_type_181: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
        mul_105: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_181, rsqrt_27);  convert_element_type_181 = None
        mul_394: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_105, convert_element_type_default_13)
        sum_58: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [3], True);  mul_394 = None
        div_22: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_105, 128);  mul_105 = None
        mul_395: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_22, sum_58);  div_22 = sum_58 = None
        sub_32: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_13, mul_395);  convert_element_type_default_13 = mul_395 = None
        mul_396: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_27);  sub_32 = rsqrt_27 = None
        convert_element_type_569: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_396, torch.bfloat16);  mul_396 = None
        convert_element_type_default_12: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_35, torch.float32);  cat_35 = None
        convert_element_type_179: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_72, torch.float32);  getitem_72 = None
        mul_104: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_179, rsqrt_26);  convert_element_type_179 = None
        mul_398: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_104, convert_element_type_default_12)
        sum_59: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [3], True);  mul_398 = None
        div_23: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_104, 128);  mul_104 = None
        mul_399: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_23, sum_59);  div_23 = sum_59 = None
        sub_33: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_12, mul_399);  convert_element_type_default_12 = mul_399 = None
        mul_400: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_26);  sub_33 = rsqrt_26 = None
        convert_element_type_572: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16);  mul_400 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_36: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_572, convert_element_type_569, mul_383], 2);  convert_element_type_572 = convert_element_type_569 = mul_383 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_267: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_36, [1, 6144, 2304]);  cat_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_268: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_267, [6144, 2304]);  view_267 = None
        permute_241: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_268, [1, 0])
        mm_110: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_241, view_102);  permute_241 = view_102 = None
        mm_111: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_268, permute_243);  view_268 = permute_243 = None
        view_269: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_111, [1, 6144, 768]);  mm_111 = None
        add_199: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_8, view_269);  slice_scatter_8 = view_269 = None
        convert_element_type_577: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        view_270: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_577, [3, 768, 768]);  convert_element_type_577 = None
        slice_scatter_9: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_270, 0, 0, 3);  view_270 = None
        add_200: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_35, slice_scatter_9);  select_scatter_35 = slice_scatter_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_578: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_199, torch.float32);  add_199 = None
        convert_element_type_174: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        mul_103: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_174, rsqrt_25);  convert_element_type_174 = None
        mul_402: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_103, convert_element_type_578)
        sum_60: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True);  mul_402 = None
        div_24: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_103, 768);  mul_103 = None
        mul_403: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_24, sum_60);  div_24 = sum_60 = None
        sub_34: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_578, mul_403);  convert_element_type_578 = mul_403 = None
        mul_404: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_25);  sub_34 = rsqrt_25 = None
        convert_element_type_580: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_404, torch.bfloat16);  mul_404 = None
        add_201: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_194, convert_element_type_580);  add_194 = convert_element_type_580 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_42: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 6)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_45: "f32[]" = torch.ops.aten.select.int(select_42, 0, 1)
        mul_405: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_201, select_45);  select_45 = None
        mul_406: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_201, convert_element_type_11)
        sum_61: "bf16[]" = torch.ops.aten.sum.default(mul_406);  mul_406 = None
        convert_element_type_581: "f32[]" = torch.ops.prims.convert_element_type.default(sum_61, torch.float32);  sum_61 = None
        add_202: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_190, mul_405);  add_190 = mul_405 = None
        select_scatter_37: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_581, 0, 1);  convert_element_type_581 = None
        select_44: "f32[]" = torch.ops.aten.select.int(select_42, 0, 0);  select_42 = None
        mul_407: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_201, select_44);  select_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_41: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 0);  slice_9 = None
        mul_100: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_41, add_69)
        add_70: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_69, mul_100);  mul_100 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_408: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_201, add_70);  add_201 = add_70 = None
        sum_62: "bf16[]" = torch.ops.aten.sum.default(mul_408);  mul_408 = None
        convert_element_type_582: "f32[]" = torch.ops.prims.convert_element_type.default(sum_62, torch.float32);  sum_62 = None
        select_scatter_38: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_582, 0, 0);  convert_element_type_582 = None
        add_203: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_37, select_scatter_38);  select_scatter_37 = select_scatter_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_39: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, select_scatter_36, 0, 6);  select_scatter_36 = None
        add_204: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_186, select_scatter_39);  add_186 = select_scatter_39 = None
        select_scatter_40: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_203, 0, 6);  add_203 = None
        add_205: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_192, select_scatter_40);  add_192 = select_scatter_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_409: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_407, select_41);  select_41 = None
        mul_410: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_407, add_69);  add_69 = None
        sum_63: "bf16[]" = torch.ops.aten.sum.default(mul_410);  mul_410 = None
        convert_element_type_583: "f32[]" = torch.ops.prims.convert_element_type.default(sum_63, torch.float32);  sum_63 = None
        add_206: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_407, mul_409);  mul_407 = mul_409 = None
        select_scatter_41: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_27, convert_element_type_583, 0, 0);  full_default_27 = convert_element_type_583 = None
        add_207: "f32[6]" = torch.ops.aten.add.Tensor(add_193, select_scatter_41);  add_193 = select_scatter_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_271: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_206, [6144, 768])
        permute_245: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_112: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_245, view_99);  permute_245 = view_99 = None
        mm_113: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_271, permute_247);  view_271 = permute_247 = None
        view_272: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_113, [1, 6144, 3072]);  mm_113 = None
        convert_element_type_588: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_98: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_28, [1, 6144, 3072]);  mm_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_5: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_98);  view_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_66: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_5, 1.0)
        mul_411: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
        mul_412: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_272, mul_411);  view_272 = mul_411 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_7: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_14: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_7, full_default_17, mul_412);  le_7 = mul_412 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_273: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_14, [6144, 3072]);  where_14 = None
        mm_114: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_249, view_273);  permute_249 = None
        mm_115: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_273, permute_250);  view_273 = permute_250 = None
        view_274: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_115, [1, 6144, 768]);  mm_115 = None
        permute_251: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
        convert_element_type_593: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_251, torch.float32);  permute_251 = None
        permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_593, [1, 0]);  convert_element_type_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_594: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        convert_element_type_166: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        mul_99: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_166, rsqrt_24);  convert_element_type_166 = None
        mul_414: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_99, convert_element_type_594)
        sum_64: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
        div_25: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_99, 768);  mul_99 = None
        mul_415: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_25, sum_64);  div_25 = sum_64 = None
        sub_35: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_594, mul_415);  convert_element_type_594 = mul_415 = None
        mul_416: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = rsqrt_24 = None
        convert_element_type_596: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_416, torch.bfloat16);  mul_416 = None
        add_208: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_206, convert_element_type_596);  add_206 = convert_element_type_596 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_275: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_208, [6144, 768])
        permute_253: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_275, [1, 0])
        mm_116: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_253, view_95);  permute_253 = view_95 = None
        mm_117: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_275, permute_255);  view_275 = permute_255 = None
        view_276: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_117, [1, 6144, 768]);  mm_117 = None
        convert_element_type_601: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        select_scatter_42: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_601, 0, 3);  convert_element_type_601 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_277: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_276, [1, 6144, 6, 128]);  view_276 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_58: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_417: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_277, permute_58);  permute_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_92: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_26, [1, 6144, 6]);  mm_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_5: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_92);  view_92 = None
        view_93: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_5, [1, 6144, 6, 1])
        mul_418: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_277, view_93);  view_277 = view_93 = None
        sum_65: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [3], True);  mul_417 = None
        view_278: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_65, [1, 6144, 6]);  sum_65 = None
        convert_element_type_602: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        convert_element_type_603: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_5, torch.float32);  sigmoid_5 = None
        sub_36: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_603)
        mul_419: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_603, sub_36);  convert_element_type_603 = sub_36 = None
        mul_420: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_602, mul_419);  convert_element_type_602 = mul_419 = None
        convert_element_type_604: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_420, torch.bfloat16);  mul_420 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_279: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_604, [6144, 6]);  convert_element_type_604 = None
        permute_257: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_118: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_257, view_91);  permute_257 = view_91 = None
        mm_119: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_279, permute_259);  view_279 = permute_259 = None
        view_280: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_119, [1, 6144, 12]);  mm_119 = None
        convert_element_type_609: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_10: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_280, 2, 0, 12);  view_280 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_261: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_418, [0, 2, 1, 3]);  mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph5 = self.fw_graph5
        joint_graph5 = self.joint_graph5
        mask_graph5 = self.mask_graph5
        flex_attention_backward_5 = torch.ops.higher_order.flex_attention_backward(permute_55, permute_56, permute_57, getitem_69, getitem_70, permute_261, None, fw_graph5, joint_graph5, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph5), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_55 = permute_56 = permute_57 = getitem_69 = getitem_70 = permute_261 = fw_graph5 = joint_graph5 = mask_graph5 = None
        getitem_147: "bf16[1, 6, 6144, 128]" = flex_attention_backward_5[0]
        getitem_148: "bf16[1, 6, 6144, 128]" = flex_attention_backward_5[1]
        getitem_149: "bf16[1, 6, 6144, 128]" = flex_attention_backward_5[2];  flex_attention_backward_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_262: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_263: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_264: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_36: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_39: "f32[]" = torch.ops.aten.select.int(select_36, 0, 0);  select_36 = None
        mul_421: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_262, select_39);  select_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_88: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_25, [1, 6144, 2304]);  mm_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_89: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_88, [1, 6144, 18, 128]);  view_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_15 = torch.ops.aten.split.Tensor(view_89, 6, -2);  view_89 = None
        getitem_64: "bf16[1, 6144, 6, 128]" = split_15[2]
        getitem_63: "bf16[1, 6144, 6, 128]" = split_15[1]
        getitem_62: "bf16[1, 6144, 6, 128]" = split_15[0];  split_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_422: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_262, getitem_64);  permute_262 = getitem_64 = None
        sum_66: "bf16[]" = torch.ops.aten.sum.default(mul_422);  mul_422 = None
        convert_element_type_610: "f32[]" = torch.ops.prims.convert_element_type.default(sum_66, torch.float32);  sum_66 = None
        select_scatter_43: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_610, 0, 0);  convert_element_type_610 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_611: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_263, torch.float32);  permute_263 = None
        slice_98: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_611, 3, 0, 64)
        slice_99: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_611, 3, 64, 128);  convert_element_type_611 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_83: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_39, 0);  primals_39 = None
        slice_43: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_83, 1, 0, 6144);  unsqueeze_83 = None
        unsqueeze_84: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_43, 2);  slice_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_423: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_99, unsqueeze_84)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_85: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_40, 0);  primals_40 = None
        slice_44: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_85, 1, 0, 6144);  unsqueeze_85 = None
        unsqueeze_86: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_44, 2);  slice_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_10: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_86)
        mul_424: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_99, neg_10);  slice_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_425: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_98, unsqueeze_86)
        add_209: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_423, mul_425);  mul_423 = mul_425 = None
        mul_426: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_98, unsqueeze_84);  slice_98 = None
        add_210: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_424, mul_426);  mul_424 = mul_426 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_37: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_210, add_209], 3);  add_210 = add_209 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_613: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_264, torch.float32);  permute_264 = None
        slice_100: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_613, 3, 0, 64)
        slice_101: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_613, 3, 64, 128);  convert_element_type_613 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_427: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_101, unsqueeze_84)
        mul_428: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_101, neg_10);  slice_101 = neg_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_429: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_100, unsqueeze_86);  unsqueeze_86 = None
        add_211: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_427, mul_429);  mul_427 = mul_429 = None
        mul_430: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_100, unsqueeze_84);  slice_100 = unsqueeze_84 = None
        add_212: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_428, mul_430);  mul_428 = mul_430 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_38: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_212, add_211], 3);  add_212 = add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_11: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_37, torch.float32);  cat_37 = None
        convert_element_type_154: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_63, torch.float32);  getitem_63 = None
        mul_88: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_154, rsqrt_23);  convert_element_type_154 = None
        mul_432: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_88, convert_element_type_default_11)
        sum_67: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [3], True);  mul_432 = None
        div_26: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_88, 128);  mul_88 = None
        mul_433: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_26, sum_67);  div_26 = sum_67 = None
        sub_37: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_11, mul_433);  convert_element_type_default_11 = mul_433 = None
        mul_434: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = rsqrt_23 = None
        convert_element_type_617: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None
        convert_element_type_default_10: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_38, torch.float32);  cat_38 = None
        convert_element_type_152: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_62, torch.float32);  getitem_62 = None
        mul_87: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_152, rsqrt_22);  convert_element_type_152 = None
        mul_436: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_87, convert_element_type_default_10)
        sum_68: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [3], True);  mul_436 = None
        div_27: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_87, 128);  mul_87 = None
        mul_437: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_27, sum_68);  div_27 = sum_68 = None
        sub_38: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_10, mul_437);  convert_element_type_default_10 = mul_437 = None
        mul_438: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_22);  sub_38 = rsqrt_22 = None
        convert_element_type_620: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_438, torch.bfloat16);  mul_438 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_39: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_620, convert_element_type_617, mul_421], 2);  convert_element_type_620 = convert_element_type_617 = mul_421 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_282: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_39, [1, 6144, 2304]);  cat_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_283: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_282, [6144, 2304]);  view_282 = None
        permute_265: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_120: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_265, view_87);  permute_265 = view_87 = None
        mm_121: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_283, permute_267);  view_283 = permute_267 = None
        view_284: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_121, [1, 6144, 768]);  mm_121 = None
        add_213: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_10, view_284);  slice_scatter_10 = view_284 = None
        convert_element_type_625: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        view_285: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_625, [3, 768, 768]);  convert_element_type_625 = None
        slice_scatter_11: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_285, 0, 0, 3);  view_285 = None
        add_214: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_42, slice_scatter_11);  select_scatter_42 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_626: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_213, torch.float32);  add_213 = None
        convert_element_type_147: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        mul_86: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_147, rsqrt_21);  convert_element_type_147 = None
        mul_440: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_86, convert_element_type_626)
        sum_69: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True);  mul_440 = None
        div_28: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_86, 768);  mul_86 = None
        mul_441: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_28, sum_69);  div_28 = sum_69 = None
        sub_39: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_626, mul_441);  convert_element_type_626 = mul_441 = None
        mul_442: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_21);  sub_39 = rsqrt_21 = None
        convert_element_type_628: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_442, torch.bfloat16);  mul_442 = None
        add_215: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_208, convert_element_type_628);  add_208 = convert_element_type_628 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_35: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_38: "f32[]" = torch.ops.aten.select.int(select_35, 0, 1)
        mul_443: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_215, select_38);  select_38 = None
        mul_444: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_215, convert_element_type_11)
        sum_70: "bf16[]" = torch.ops.aten.sum.default(mul_444);  mul_444 = None
        convert_element_type_629: "f32[]" = torch.ops.prims.convert_element_type.default(sum_70, torch.float32);  sum_70 = None
        add_216: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_202, mul_443);  add_202 = mul_443 = None
        select_scatter_44: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_629, 0, 1);  convert_element_type_629 = None
        select_37: "f32[]" = torch.ops.aten.select.int(select_35, 0, 0);  select_35 = None
        mul_445: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_215, select_37);  select_37 = None
        mul_446: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_215, add_58);  add_215 = add_58 = None
        sum_71: "bf16[]" = torch.ops.aten.sum.default(mul_446);  mul_446 = None
        convert_element_type_630: "f32[]" = torch.ops.prims.convert_element_type.default(sum_71, torch.float32);  sum_71 = None
        add_217: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_371, mul_445);  mul_371 = mul_445 = None
        select_scatter_45: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_630, 0, 0);  convert_element_type_630 = None
        add_218: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_44, select_scatter_45);  select_scatter_44 = select_scatter_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_46: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, select_scatter_43, 0, 5);  select_scatter_43 = None
        add_219: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_204, select_scatter_46);  add_204 = select_scatter_46 = None
        select_scatter_47: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_218, 0, 5);  add_218 = None
        add_220: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_205, select_scatter_47);  add_205 = select_scatter_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_286: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_217, [6144, 768])
        permute_269: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_122: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_269, view_84);  permute_269 = view_84 = None
        mm_123: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_286, permute_271);  view_286 = permute_271 = None
        view_287: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_123, [1, 6144, 3072]);  mm_123 = None
        convert_element_type_635: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_83: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_23, [1, 6144, 3072]);  mm_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_4: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_83);  view_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_67: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_4, 1.0)
        mul_447: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_67, 2.0);  pow_67 = None
        mul_448: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_287, mul_447);  view_287 = mul_447 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_8: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_15: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_8, full_default_17, mul_448);  le_8 = mul_448 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_288: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_15, [6144, 3072]);  where_15 = None
        mm_124: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_273, view_288);  permute_273 = None
        mm_125: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_288, permute_274);  view_288 = permute_274 = None
        view_289: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_125, [1, 6144, 768]);  mm_125 = None
        permute_275: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
        convert_element_type_640: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_275, torch.float32);  permute_275 = None
        permute_276: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_640, [1, 0]);  convert_element_type_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_641: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_289, torch.float32);  view_289 = None
        convert_element_type_139: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32);  add_56 = None
        mul_83: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_139, rsqrt_20);  convert_element_type_139 = None
        mul_450: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_83, convert_element_type_641)
        sum_72: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [2], True);  mul_450 = None
        div_29: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_83, 768);  mul_83 = None
        mul_451: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_29, sum_72);  div_29 = sum_72 = None
        sub_40: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_641, mul_451);  convert_element_type_641 = mul_451 = None
        mul_452: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_20);  sub_40 = rsqrt_20 = None
        convert_element_type_643: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_452, torch.bfloat16);  mul_452 = None
        add_221: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_217, convert_element_type_643);  add_217 = convert_element_type_643 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_290: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_221, [6144, 768])
        permute_277: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_126: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_277, view_80);  permute_277 = view_80 = None
        mm_127: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_290, permute_279);  view_290 = permute_279 = None
        view_291: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_127, [1, 6144, 768]);  mm_127 = None
        convert_element_type_648: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        select_scatter_48: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_648, 0, 3);  convert_element_type_648 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_292: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_291, [1, 6144, 6, 128]);  view_291 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_48: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_453: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_292, permute_48);  permute_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_77: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_21, [1, 6144, 6]);  mm_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_4: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_77);  view_77 = None
        view_78: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_4, [1, 6144, 6, 1])
        mul_454: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_292, view_78);  view_292 = view_78 = None
        sum_73: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [3], True);  mul_453 = None
        view_293: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_73, [1, 6144, 6]);  sum_73 = None
        convert_element_type_649: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_293, torch.float32);  view_293 = None
        convert_element_type_650: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_4, torch.float32);  sigmoid_4 = None
        sub_41: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_650)
        mul_455: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_650, sub_41);  convert_element_type_650 = sub_41 = None
        mul_456: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_649, mul_455);  convert_element_type_649 = mul_455 = None
        convert_element_type_651: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_456, torch.bfloat16);  mul_456 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_294: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_651, [6144, 6]);  convert_element_type_651 = None
        permute_281: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_294, [1, 0])
        mm_128: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_281, view_76);  permute_281 = view_76 = None
        mm_129: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_294, permute_283);  view_294 = permute_283 = None
        view_295: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_129, [1, 6144, 12]);  mm_129 = None
        convert_element_type_656: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_12: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_295, 2, 0, 12);  view_295 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_285: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_454, [0, 2, 1, 3]);  mul_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph6 = self.fw_graph6
        joint_graph6 = self.joint_graph6
        mask_graph6 = self.mask_graph6
        flex_attention_backward_6 = torch.ops.higher_order.flex_attention_backward(permute_45, permute_46, permute_47, getitem_59, getitem_60, permute_285, None, fw_graph6, joint_graph6, (6144, 6144, clamp_max, unsqueeze_9, clamp_max_1, unsqueeze_13, convert_element_type_2, clone_4, convert_element_type_4, clone_7, 128, 128, mask_graph6), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_45 = permute_46 = permute_47 = getitem_59 = getitem_60 = permute_285 = fw_graph6 = joint_graph6 = mask_graph6 = None
        getitem_151: "bf16[1, 6, 6144, 128]" = flex_attention_backward_6[0]
        getitem_152: "bf16[1, 6, 6144, 128]" = flex_attention_backward_6[1]
        getitem_153: "bf16[1, 6, 6144, 128]" = flex_attention_backward_6[2];  flex_attention_backward_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_286: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_287: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_288: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_30: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 4)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_33: "f32[]" = torch.ops.aten.select.int(select_30, 0, 0);  select_30 = None
        mul_457: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_286, select_33);  select_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_73: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_20, [1, 6144, 2304]);  mm_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_74: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_73, [1, 6144, 18, 128]);  view_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_12 = torch.ops.aten.split.Tensor(view_74, 6, -2);  view_74 = None
        getitem_54: "bf16[1, 6144, 6, 128]" = split_12[2]
        getitem_53: "bf16[1, 6144, 6, 128]" = split_12[1]
        getitem_52: "bf16[1, 6144, 6, 128]" = split_12[0];  split_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_458: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_286, getitem_54);  permute_286 = getitem_54 = None
        sum_74: "bf16[]" = torch.ops.aten.sum.default(mul_458);  mul_458 = None
        convert_element_type_657: "f32[]" = torch.ops.prims.convert_element_type.default(sum_74, torch.float32);  sum_74 = None
        select_scatter_49: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_657, 0, 0);  convert_element_type_657 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_658: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_287, torch.float32);  permute_287 = None
        slice_102: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_658, 3, 0, 64)
        slice_103: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_658, 3, 64, 128);  convert_element_type_658 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_75: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_33, 0);  primals_33 = None
        slice_37: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_75, 1, 0, 6144);  unsqueeze_75 = None
        unsqueeze_76: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_37, 2);  slice_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_459: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_103, unsqueeze_76)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_77: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_34, 0);  primals_34 = None
        slice_38: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_77, 1, 0, 6144);  unsqueeze_77 = None
        unsqueeze_78: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_38, 2);  slice_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_8: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_78)
        mul_460: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_103, neg_8);  slice_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_461: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_102, unsqueeze_78)
        add_222: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_459, mul_461);  mul_459 = mul_461 = None
        mul_462: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_102, unsqueeze_76);  slice_102 = None
        add_223: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_460, mul_462);  mul_460 = mul_462 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_40: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_223, add_222], 3);  add_223 = add_222 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_660: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_288, torch.float32);  permute_288 = None
        slice_104: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_660, 3, 0, 64)
        slice_105: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_660, 3, 64, 128);  convert_element_type_660 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_463: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_105, unsqueeze_76)
        mul_464: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_105, neg_8);  slice_105 = neg_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_465: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_104, unsqueeze_78);  unsqueeze_78 = None
        add_224: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_463, mul_465);  mul_463 = mul_465 = None
        mul_466: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_104, unsqueeze_76);  slice_104 = unsqueeze_76 = None
        add_225: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_464, mul_466);  mul_464 = mul_466 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_41: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_225, add_224], 3);  add_225 = add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_9: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_40, torch.float32);  cat_40 = None
        convert_element_type_127: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
        mul_72: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, rsqrt_19);  convert_element_type_127 = None
        mul_468: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_72, convert_element_type_default_9)
        sum_75: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_468, [3], True);  mul_468 = None
        div_30: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_72, 128);  mul_72 = None
        mul_469: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_30, sum_75);  div_30 = sum_75 = None
        sub_42: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_9, mul_469);  convert_element_type_default_9 = mul_469 = None
        mul_470: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_19);  sub_42 = rsqrt_19 = None
        convert_element_type_664: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_470, torch.bfloat16);  mul_470 = None
        convert_element_type_default_8: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_41, torch.float32);  cat_41 = None
        convert_element_type_125: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_52, torch.float32);  getitem_52 = None
        mul_71: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_125, rsqrt_18);  convert_element_type_125 = None
        mul_472: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_71, convert_element_type_default_8)
        sum_76: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [3], True);  mul_472 = None
        div_31: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_71, 128);  mul_71 = None
        mul_473: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_31, sum_76);  div_31 = sum_76 = None
        sub_43: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_8, mul_473);  convert_element_type_default_8 = mul_473 = None
        mul_474: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_18);  sub_43 = rsqrt_18 = None
        convert_element_type_667: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_474, torch.bfloat16);  mul_474 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_42: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_667, convert_element_type_664, mul_457], 2);  convert_element_type_667 = convert_element_type_664 = mul_457 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_297: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_42, [1, 6144, 2304]);  cat_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_298: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_297, [6144, 2304]);  view_297 = None
        permute_289: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_130: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_289, view_72);  permute_289 = view_72 = None
        mm_131: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_298, permute_291);  view_298 = permute_291 = None
        view_299: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_131, [1, 6144, 768]);  mm_131 = None
        add_226: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_12, view_299);  slice_scatter_12 = view_299 = None
        convert_element_type_672: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        view_300: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_672, [3, 768, 768]);  convert_element_type_672 = None
        slice_scatter_13: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_300, 0, 0, 3);  view_300 = None
        add_227: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_48, slice_scatter_13);  select_scatter_48 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_673: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_226, torch.float32);  add_226 = None
        convert_element_type_120: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        mul_70: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_120, rsqrt_17);  convert_element_type_120 = None
        mul_476: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_70, convert_element_type_673)
        sum_77: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True);  mul_476 = None
        div_32: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_70, 768);  mul_70 = None
        mul_477: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_32, sum_77);  div_32 = sum_77 = None
        sub_44: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_673, mul_477);  convert_element_type_673 = mul_477 = None
        mul_478: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_17);  sub_44 = rsqrt_17 = None
        convert_element_type_675: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_478, torch.bfloat16);  mul_478 = None
        add_228: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_221, convert_element_type_675);  add_221 = convert_element_type_675 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_29: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 4)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_32: "f32[]" = torch.ops.aten.select.int(select_29, 0, 1)
        mul_479: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_228, select_32);  select_32 = None
        mul_480: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_228, convert_element_type_11)
        sum_78: "bf16[]" = torch.ops.aten.sum.default(mul_480);  mul_480 = None
        convert_element_type_676: "f32[]" = torch.ops.prims.convert_element_type.default(sum_78, torch.float32);  sum_78 = None
        add_229: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_216, mul_479);  add_216 = mul_479 = None
        select_scatter_50: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_676, 0, 1);  convert_element_type_676 = None
        select_31: "f32[]" = torch.ops.aten.select.int(select_29, 0, 0);  select_29 = None
        mul_481: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_228, select_31);  select_31 = None
        mul_482: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_228, add_47);  add_228 = add_47 = None
        sum_79: "bf16[]" = torch.ops.aten.sum.default(mul_482);  mul_482 = None
        convert_element_type_677: "f32[]" = torch.ops.prims.convert_element_type.default(sum_79, torch.float32);  sum_79 = None
        add_230: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_359, mul_481);  mul_359 = mul_481 = None
        select_scatter_51: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_677, 0, 0);  convert_element_type_677 = None
        add_231: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_50, select_scatter_51);  select_scatter_50 = select_scatter_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_52: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, select_scatter_49, 0, 4);  select_scatter_49 = None
        add_232: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_219, select_scatter_52);  add_219 = select_scatter_52 = None
        select_scatter_53: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_231, 0, 4);  add_231 = None
        add_233: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_220, select_scatter_53);  add_220 = select_scatter_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_301: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_230, [6144, 768])
        permute_293: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_301, [1, 0])
        mm_132: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_293, view_69);  permute_293 = view_69 = None
        mm_133: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_301, permute_295);  view_301 = permute_295 = None
        view_302: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_133, [1, 6144, 3072]);  mm_133 = None
        convert_element_type_682: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_68: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_18, [1, 6144, 3072]);  mm_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_3: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_68);  view_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_68: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_3, 1.0)
        mul_483: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
        mul_484: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_302, mul_483);  view_302 = mul_483 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_9: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_16: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_9, full_default_17, mul_484);  le_9 = mul_484 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_303: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_16, [6144, 3072]);  where_16 = None
        mm_134: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_297, view_303);  permute_297 = None
        mm_135: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_303, permute_298);  view_303 = permute_298 = None
        view_304: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_135, [1, 6144, 768]);  mm_135 = None
        permute_299: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
        convert_element_type_687: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_299, torch.float32);  permute_299 = None
        permute_300: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_687, [1, 0]);  convert_element_type_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_688: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        convert_element_type_112: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        mul_67: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_112, rsqrt_16);  convert_element_type_112 = None
        mul_486: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_67, convert_element_type_688)
        sum_80: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_486, [2], True);  mul_486 = None
        div_33: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_67, 768);  mul_67 = None
        mul_487: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_33, sum_80);  div_33 = sum_80 = None
        sub_45: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_688, mul_487);  convert_element_type_688 = mul_487 = None
        mul_488: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_16);  sub_45 = rsqrt_16 = None
        convert_element_type_690: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_488, torch.bfloat16);  mul_488 = None
        add_234: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_230, convert_element_type_690);  add_230 = convert_element_type_690 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_305: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_234, [6144, 768])
        permute_301: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_136: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_301, view_65);  permute_301 = view_65 = None
        mm_137: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_305, permute_303);  view_305 = permute_303 = None
        view_306: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_137, [1, 6144, 768]);  mm_137 = None
        convert_element_type_695: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        select_scatter_54: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_695, 0, 3);  convert_element_type_695 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_307: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_306, [1, 6144, 6, 128]);  view_306 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_38: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_489: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_307, permute_38);  permute_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_62: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_16, [1, 6144, 6]);  mm_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_3: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_62);  view_62 = None
        view_63: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_3, [1, 6144, 6, 1])
        mul_490: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_307, view_63);  view_307 = view_63 = None
        sum_81: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_489, [3], True);  mul_489 = None
        view_308: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_81, [1, 6144, 6]);  sum_81 = None
        convert_element_type_696: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_308, torch.float32);  view_308 = None
        convert_element_type_697: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_3, torch.float32);  sigmoid_3 = None
        sub_46: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_697)
        mul_491: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_697, sub_46);  convert_element_type_697 = sub_46 = None
        mul_492: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_696, mul_491);  convert_element_type_696 = mul_491 = None
        convert_element_type_698: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_492, torch.bfloat16);  mul_492 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_309: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_698, [6144, 6]);  convert_element_type_698 = None
        permute_305: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_309, [1, 0])
        mm_138: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_305, view_61);  permute_305 = view_61 = None
        mm_139: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_309, permute_307);  view_309 = permute_307 = None
        view_310: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_139, [1, 6144, 12]);  mm_139 = None
        convert_element_type_703: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_14: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_310, 2, 0, 12);  view_310 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_309: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_490, [0, 2, 1, 3]);  mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph7 = self.fw_graph7
        joint_graph7 = self.joint_graph7
        mask_graph7 = self.mask_graph7
        flex_attention_backward_7 = torch.ops.higher_order.flex_attention_backward(permute_35, permute_36, permute_37, getitem_49, getitem_50, permute_309, None, fw_graph7, joint_graph7, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph7), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_35 = permute_36 = permute_37 = getitem_49 = getitem_50 = permute_309 = fw_graph7 = joint_graph7 = mask_graph7 = None
        getitem_155: "bf16[1, 6, 6144, 128]" = flex_attention_backward_7[0]
        getitem_156: "bf16[1, 6, 6144, 128]" = flex_attention_backward_7[1]
        getitem_157: "bf16[1, 6, 6144, 128]" = flex_attention_backward_7[2];  flex_attention_backward_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_310: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_157, [0, 2, 1, 3]);  getitem_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_311: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_312: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_24: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_27: "f32[]" = torch.ops.aten.select.int(select_24, 0, 0);  select_24 = None
        mul_493: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_310, select_27);  select_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_58: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_15, [1, 6144, 2304]);  mm_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_59: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_58, [1, 6144, 18, 128]);  view_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_9 = torch.ops.aten.split.Tensor(view_59, 6, -2);  view_59 = None
        getitem_44: "bf16[1, 6144, 6, 128]" = split_9[2]
        getitem_43: "bf16[1, 6144, 6, 128]" = split_9[1]
        getitem_42: "bf16[1, 6144, 6, 128]" = split_9[0];  split_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_494: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_310, getitem_44);  permute_310 = getitem_44 = None
        sum_82: "bf16[]" = torch.ops.aten.sum.default(mul_494);  mul_494 = None
        convert_element_type_704: "f32[]" = torch.ops.prims.convert_element_type.default(sum_82, torch.float32);  sum_82 = None
        select_scatter_55: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_704, 0, 0);  convert_element_type_704 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_705: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_311, torch.float32);  permute_311 = None
        slice_106: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_705, 3, 0, 64)
        slice_107: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_705, 3, 64, 128);  convert_element_type_705 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_67: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_27, 0);  primals_27 = None
        slice_31: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_67, 1, 0, 6144);  unsqueeze_67 = None
        unsqueeze_68: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_31, 2);  slice_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_495: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_107, unsqueeze_68)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_69: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_28, 0);  primals_28 = None
        slice_32: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_69, 1, 0, 6144);  unsqueeze_69 = None
        unsqueeze_70: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_32, 2);  slice_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_6: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_70)
        mul_496: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_107, neg_6);  slice_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_497: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_106, unsqueeze_70)
        add_235: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_495, mul_497);  mul_495 = mul_497 = None
        mul_498: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_106, unsqueeze_68);  slice_106 = None
        add_236: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_496, mul_498);  mul_496 = mul_498 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_43: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_236, add_235], 3);  add_236 = add_235 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_707: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_312, torch.float32);  permute_312 = None
        slice_108: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_707, 3, 0, 64)
        slice_109: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_707, 3, 64, 128);  convert_element_type_707 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_499: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_109, unsqueeze_68)
        mul_500: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_109, neg_6);  slice_109 = neg_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_501: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_108, unsqueeze_70);  unsqueeze_70 = None
        add_237: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_499, mul_501);  mul_499 = mul_501 = None
        mul_502: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_108, unsqueeze_68);  slice_108 = unsqueeze_68 = None
        add_238: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_500, mul_502);  mul_500 = mul_502 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_44: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_238, add_237], 3);  add_238 = add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_7: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_43, torch.float32);  cat_43 = None
        convert_element_type_100: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
        mul_56: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_100, rsqrt_15);  convert_element_type_100 = None
        mul_504: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_56, convert_element_type_default_7)
        sum_83: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_504, [3], True);  mul_504 = None
        div_34: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_56, 128);  mul_56 = None
        mul_505: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_34, sum_83);  div_34 = sum_83 = None
        sub_47: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_7, mul_505);  convert_element_type_default_7 = mul_505 = None
        mul_506: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_15);  sub_47 = rsqrt_15 = None
        convert_element_type_711: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_506, torch.bfloat16);  mul_506 = None
        convert_element_type_default_6: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_44, torch.float32);  cat_44 = None
        convert_element_type_98: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_42, torch.float32);  getitem_42 = None
        mul_55: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_98, rsqrt_14);  convert_element_type_98 = None
        mul_508: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_55, convert_element_type_default_6)
        sum_84: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [3], True);  mul_508 = None
        div_35: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_55, 128);  mul_55 = None
        mul_509: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_35, sum_84);  div_35 = sum_84 = None
        sub_48: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_6, mul_509);  convert_element_type_default_6 = mul_509 = None
        mul_510: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_14);  sub_48 = rsqrt_14 = None
        convert_element_type_714: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_510, torch.bfloat16);  mul_510 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_45: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_714, convert_element_type_711, mul_493], 2);  convert_element_type_714 = convert_element_type_711 = mul_493 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_312: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_45, [1, 6144, 2304]);  cat_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_313: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_312, [6144, 2304]);  view_312 = None
        permute_313: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_140: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_313, view_57);  permute_313 = view_57 = None
        mm_141: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_313, permute_315);  view_313 = permute_315 = None
        view_314: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_141, [1, 6144, 768]);  mm_141 = None
        add_239: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_14, view_314);  slice_scatter_14 = view_314 = None
        convert_element_type_719: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        view_315: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_719, [3, 768, 768]);  convert_element_type_719 = None
        slice_scatter_15: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_315, 0, 0, 3);  view_315 = None
        add_240: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_54, slice_scatter_15);  select_scatter_54 = slice_scatter_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_720: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_239, torch.float32);  add_239 = None
        convert_element_type_93: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        mul_54: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_13);  convert_element_type_93 = None
        mul_512: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_54, convert_element_type_720)
        sum_85: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True);  mul_512 = None
        div_36: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_54, 768);  mul_54 = None
        mul_513: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_36, sum_85);  div_36 = sum_85 = None
        sub_49: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_720, mul_513);  convert_element_type_720 = mul_513 = None
        mul_514: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_13);  sub_49 = rsqrt_13 = None
        convert_element_type_722: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_514, torch.bfloat16);  mul_514 = None
        add_241: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_234, convert_element_type_722);  add_234 = convert_element_type_722 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_23: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_26: "f32[]" = torch.ops.aten.select.int(select_23, 0, 1)
        mul_515: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_241, select_26);  select_26 = None
        mul_516: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_241, convert_element_type_11)
        sum_86: "bf16[]" = torch.ops.aten.sum.default(mul_516);  mul_516 = None
        convert_element_type_723: "f32[]" = torch.ops.prims.convert_element_type.default(sum_86, torch.float32);  sum_86 = None
        add_242: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_229, mul_515);  add_229 = mul_515 = None
        select_scatter_56: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_723, 0, 1);  convert_element_type_723 = None
        select_25: "f32[]" = torch.ops.aten.select.int(select_23, 0, 0);  select_23 = None
        mul_517: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_241, select_25);  select_25 = None
        mul_518: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_241, add_36);  add_241 = add_36 = None
        sum_87: "bf16[]" = torch.ops.aten.sum.default(mul_518);  mul_518 = None
        convert_element_type_724: "f32[]" = torch.ops.prims.convert_element_type.default(sum_87, torch.float32);  sum_87 = None
        add_243: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_321, mul_517);  mul_321 = mul_517 = None
        select_scatter_57: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_724, 0, 0);  convert_element_type_724 = None
        add_244: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_56, select_scatter_57);  select_scatter_56 = select_scatter_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_58: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, select_scatter_55, 0, 3);  select_scatter_55 = None
        add_245: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_232, select_scatter_58);  add_232 = select_scatter_58 = None
        select_scatter_59: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_244, 0, 3);  add_244 = None
        add_246: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_233, select_scatter_59);  add_233 = select_scatter_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_316: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_243, [6144, 768])
        permute_317: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_316, [1, 0])
        mm_142: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_317, view_54);  permute_317 = view_54 = None
        mm_143: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_316, permute_319);  view_316 = permute_319 = None
        view_317: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_143, [1, 6144, 3072]);  mm_143 = None
        convert_element_type_729: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_53: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_13, [1, 6144, 3072]);  mm_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_2: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_53);  view_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_69: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_2, 1.0)
        mul_519: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_69, 2.0);  pow_69 = None
        mul_520: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_317, mul_519);  view_317 = mul_519 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_10: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_17: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_10, full_default_17, mul_520);  le_10 = mul_520 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_318: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_17, [6144, 3072]);  where_17 = None
        mm_144: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_321, view_318);  permute_321 = None
        mm_145: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_318, permute_322);  view_318 = permute_322 = None
        view_319: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_145, [1, 6144, 768]);  mm_145 = None
        permute_323: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
        convert_element_type_734: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_323, torch.float32);  permute_323 = None
        permute_324: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_734, [1, 0]);  convert_element_type_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_735: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        convert_element_type_85: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32);  add_34 = None
        mul_51: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_85, rsqrt_12);  convert_element_type_85 = None
        mul_522: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_51, convert_element_type_735)
        sum_88: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True);  mul_522 = None
        div_37: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_51, 768);  mul_51 = None
        mul_523: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_37, sum_88);  div_37 = sum_88 = None
        sub_50: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_735, mul_523);  convert_element_type_735 = mul_523 = None
        mul_524: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_12);  sub_50 = rsqrt_12 = None
        convert_element_type_737: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_524, torch.bfloat16);  mul_524 = None
        add_247: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_243, convert_element_type_737);  add_243 = convert_element_type_737 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_320: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_247, [6144, 768])
        permute_325: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_320, [1, 0])
        mm_146: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_325, view_50);  permute_325 = view_50 = None
        mm_147: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_320, permute_327);  view_320 = permute_327 = None
        view_321: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_147, [1, 6144, 768]);  mm_147 = None
        convert_element_type_742: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        select_scatter_60: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_742, 0, 3);  convert_element_type_742 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_322: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_321, [1, 6144, 6, 128]);  view_321 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_28: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_525: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_322, permute_28);  permute_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_47: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_11, [1, 6144, 6]);  mm_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_2: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_47);  view_47 = None
        view_48: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_2, [1, 6144, 6, 1])
        mul_526: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_322, view_48);  view_322 = view_48 = None
        sum_89: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [3], True);  mul_525 = None
        view_323: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_89, [1, 6144, 6]);  sum_89 = None
        convert_element_type_743: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        convert_element_type_744: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_2, torch.float32);  sigmoid_2 = None
        sub_51: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_744)
        mul_527: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_744, sub_51);  convert_element_type_744 = sub_51 = None
        mul_528: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_743, mul_527);  convert_element_type_743 = mul_527 = None
        convert_element_type_745: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_528, torch.bfloat16);  mul_528 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_324: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_745, [6144, 6]);  convert_element_type_745 = None
        permute_329: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_148: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_329, view_46);  permute_329 = view_46 = None
        mm_149: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_324, permute_331);  view_324 = permute_331 = None
        view_325: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_149, [1, 6144, 12]);  mm_149 = None
        convert_element_type_750: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_16: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_325, 2, 0, 12);  view_325 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_333: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_526, [0, 2, 1, 3]);  mul_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph8 = self.fw_graph8
        joint_graph8 = self.joint_graph8
        mask_graph8 = self.mask_graph8
        flex_attention_backward_8 = torch.ops.higher_order.flex_attention_backward(permute_25, permute_26, permute_27, getitem_39, getitem_40, permute_333, None, fw_graph8, joint_graph8, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph8), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_25 = permute_26 = permute_27 = getitem_39 = getitem_40 = permute_333 = fw_graph8 = joint_graph8 = mask_graph8 = None
        getitem_159: "bf16[1, 6, 6144, 128]" = flex_attention_backward_8[0]
        getitem_160: "bf16[1, 6, 6144, 128]" = flex_attention_backward_8[1]
        getitem_161: "bf16[1, 6, 6144, 128]" = flex_attention_backward_8[2];  flex_attention_backward_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_334: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_335: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_336: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_17: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_21: "f32[]" = torch.ops.aten.select.int(select_17, 0, 1)
        mul_529: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_334, select_21);  select_21 = None
        mul_530: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_334, view_44);  view_44 = None
        sum_90: "bf16[]" = torch.ops.aten.sum.default(mul_530);  mul_530 = None
        convert_element_type_751: "f32[]" = torch.ops.prims.convert_element_type.default(sum_90, torch.float32);  sum_90 = None
        view_327: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_529, [6144, 768]);  mul_529 = None
        add_248: "bf16[6144, 768]" = torch.ops.aten.add.Tensor(view_200, view_327);  view_200 = view_327 = None
        select_scatter_61: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_751, 0, 1);  convert_element_type_751 = None
        select_20: "f32[]" = torch.ops.aten.select.int(select_17, 0, 0);  select_17 = None
        mul_531: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_334, select_20);  select_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_42: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_10, [1, 6144, 2304]);  mm_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_43: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_42, [1, 6144, 18, 128]);  view_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_6 = torch.ops.aten.split.Tensor(view_43, 6, -2);  view_43 = None
        getitem_34: "bf16[1, 6144, 6, 128]" = split_6[2]
        getitem_33: "bf16[1, 6144, 6, 128]" = split_6[1]
        getitem_32: "bf16[1, 6144, 6, 128]" = split_6[0];  split_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        mul_532: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_334, getitem_34);  permute_334 = getitem_34 = None
        sum_91: "bf16[]" = torch.ops.aten.sum.default(mul_532);  mul_532 = None
        convert_element_type_752: "f32[]" = torch.ops.prims.convert_element_type.default(sum_91, torch.float32);  sum_91 = None
        select_scatter_62: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_752, 0, 0);  convert_element_type_752 = None
        add_249: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_61, select_scatter_62);  select_scatter_61 = select_scatter_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_753: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_335, torch.float32);  permute_335 = None
        slice_110: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_753, 3, 0, 64)
        slice_111: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_753, 3, 64, 128);  convert_element_type_753 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_59: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_21, 0);  primals_21 = None
        slice_25: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_59, 1, 0, 6144);  unsqueeze_59 = None
        unsqueeze_60: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_25, 2);  slice_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_533: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_111, unsqueeze_60)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_61: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_22, 0);  primals_22 = None
        slice_26: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_61, 1, 0, 6144);  unsqueeze_61 = None
        unsqueeze_62: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_26, 2);  slice_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_4: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_62)
        mul_534: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_111, neg_4);  slice_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_535: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_110, unsqueeze_62)
        add_250: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_533, mul_535);  mul_533 = mul_535 = None
        mul_536: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_110, unsqueeze_60);  slice_110 = None
        add_251: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_534, mul_536);  mul_534 = mul_536 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_46: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_251, add_250], 3);  add_251 = add_250 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_755: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_336, torch.float32);  permute_336 = None
        slice_112: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_755, 3, 0, 64)
        slice_113: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_755, 3, 64, 128);  convert_element_type_755 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_537: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_113, unsqueeze_60)
        mul_538: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_113, neg_4);  slice_113 = neg_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_539: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_112, unsqueeze_62);  unsqueeze_62 = None
        add_252: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_537, mul_539);  mul_537 = mul_539 = None
        mul_540: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_112, unsqueeze_60);  slice_112 = unsqueeze_60 = None
        add_253: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_538, mul_540);  mul_538 = mul_540 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_47: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_253, add_252], 3);  add_253 = add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_5: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_46, torch.float32);  cat_46 = None
        convert_element_type_73: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
        mul_39: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_11);  convert_element_type_73 = None
        mul_542: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_39, convert_element_type_default_5)
        sum_92: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [3], True);  mul_542 = None
        div_38: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_39, 128);  mul_39 = None
        mul_543: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_38, sum_92);  div_38 = sum_92 = None
        sub_52: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_5, mul_543);  convert_element_type_default_5 = mul_543 = None
        mul_544: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_11);  sub_52 = rsqrt_11 = None
        convert_element_type_759: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_544, torch.bfloat16);  mul_544 = None
        convert_element_type_default_4: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_47, torch.float32);  cat_47 = None
        convert_element_type_71: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_32, torch.float32);  getitem_32 = None
        mul_38: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_71, rsqrt_10);  convert_element_type_71 = None
        mul_546: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_38, convert_element_type_default_4)
        sum_93: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_546, [3], True);  mul_546 = None
        div_39: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_38, 128);  mul_38 = None
        mul_547: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_39, sum_93);  div_39 = sum_93 = None
        sub_53: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_4, mul_547);  convert_element_type_default_4 = mul_547 = None
        mul_548: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_10);  sub_53 = rsqrt_10 = None
        convert_element_type_762: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_548, torch.bfloat16);  mul_548 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_48: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_762, convert_element_type_759, mul_531], 2);  convert_element_type_762 = convert_element_type_759 = mul_531 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_328: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_48, [1, 6144, 2304]);  cat_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_329: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_328, [6144, 2304]);  view_328 = None
        permute_337: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_329, [1, 0])
        mm_150: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_337, view_41);  permute_337 = view_41 = None
        mm_151: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_329, permute_339);  view_329 = permute_339 = None
        view_330: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_151, [1, 6144, 768]);  mm_151 = None
        add_254: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_16, view_330);  slice_scatter_16 = view_330 = None
        convert_element_type_767: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        view_331: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_767, [3, 768, 768]);  convert_element_type_767 = None
        slice_scatter_17: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_331, 0, 0, 3);  view_331 = None
        add_255: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_60, slice_scatter_17);  select_scatter_60 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_768: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_254, torch.float32);  add_254 = None
        convert_element_type_66: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        mul_37: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_66, rsqrt_9);  convert_element_type_66 = None
        mul_550: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_37, convert_element_type_768)
        sum_94: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_550, [2], True);  mul_550 = None
        div_40: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_37, 768);  mul_37 = None
        mul_551: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_40, sum_94);  div_40 = sum_94 = None
        sub_54: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_768, mul_551);  convert_element_type_768 = mul_551 = None
        mul_552: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_9);  sub_54 = rsqrt_9 = None
        convert_element_type_770: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_552, torch.bfloat16);  mul_552 = None
        add_256: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_247, convert_element_type_770);  add_247 = convert_element_type_770 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_16: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_19: "f32[]" = torch.ops.aten.select.int(select_16, 0, 1)
        mul_553: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_256, select_19);  select_19 = None
        mul_554: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_256, convert_element_type_11)
        sum_95: "bf16[]" = torch.ops.aten.sum.default(mul_554);  mul_554 = None
        convert_element_type_771: "f32[]" = torch.ops.prims.convert_element_type.default(sum_95, torch.float32);  sum_95 = None
        add_257: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_242, mul_553);  add_242 = mul_553 = None
        select_scatter_63: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_771, 0, 1);  convert_element_type_771 = None
        select_18: "f32[]" = torch.ops.aten.select.int(select_16, 0, 0);  select_16 = None
        mul_555: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_256, select_18);  select_18 = None
        mul_556: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_256, add_24);  add_256 = add_24 = None
        sum_96: "bf16[]" = torch.ops.aten.sum.default(mul_556);  mul_556 = None
        convert_element_type_772: "f32[]" = torch.ops.prims.convert_element_type.default(sum_96, torch.float32);  sum_96 = None
        add_258: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_281, mul_555);  mul_281 = mul_555 = None
        select_scatter_64: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_772, 0, 0);  convert_element_type_772 = None
        add_259: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_63, select_scatter_64);  select_scatter_63 = select_scatter_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_65: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_249, 0, 2);  add_249 = None
        add_260: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_245, select_scatter_65);  add_245 = select_scatter_65 = None
        select_scatter_66: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_259, 0, 2);  add_259 = None
        add_261: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_246, select_scatter_66);  add_246 = select_scatter_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_332: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_258, [6144, 768])
        permute_341: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_152: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_341, view_38);  permute_341 = view_38 = None
        mm_153: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_332, permute_343);  view_332 = permute_343 = None
        view_333: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_153, [1, 6144, 3072]);  mm_153 = None
        convert_element_type_777: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_37: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_8, [1, 6144, 3072]);  mm_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_1: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_37);  view_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_70: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_1, 1.0)
        mul_557: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_558: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_333, mul_557);  view_333 = mul_557 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_11: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_18: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_11, full_default_17, mul_558);  le_11 = mul_558 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_334: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_18, [6144, 3072]);  where_18 = None
        mm_154: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_345, view_334);  permute_345 = None
        mm_155: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_334, permute_346);  view_334 = permute_346 = None
        view_335: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_155, [1, 6144, 768]);  mm_155 = None
        permute_347: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
        convert_element_type_782: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_347, torch.float32);  permute_347 = None
        permute_348: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_782, [1, 0]);  convert_element_type_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_783: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        convert_element_type_58: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32);  add_22 = None
        mul_34: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_58, rsqrt_8);  convert_element_type_58 = None
        mul_560: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_34, convert_element_type_783)
        sum_97: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True);  mul_560 = None
        div_41: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_34, 768);  mul_34 = None
        mul_561: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_41, sum_97);  div_41 = sum_97 = None
        sub_55: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_783, mul_561);  convert_element_type_783 = mul_561 = None
        mul_562: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_8);  sub_55 = rsqrt_8 = None
        convert_element_type_785: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_562, torch.bfloat16);  mul_562 = None
        add_262: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_258, convert_element_type_785);  add_258 = convert_element_type_785 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_336: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_262, [6144, 768])
        permute_349: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_156: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_349, view_34);  permute_349 = view_34 = None
        mm_157: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_336, permute_351);  view_336 = permute_351 = None
        view_337: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_157, [1, 6144, 768]);  mm_157 = None
        convert_element_type_790: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        select_scatter_67: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_790, 0, 3);  convert_element_type_790 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_338: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_337, [1, 6144, 6, 128]);  view_337 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_18: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_29, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_563: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_338, permute_18);  permute_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_31: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_6, [1, 6144, 6]);  mm_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_1: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_31);  view_31 = None
        view_32: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_1, [1, 6144, 6, 1])
        mul_564: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_338, view_32);  view_338 = view_32 = None
        sum_98: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_563, [3], True);  mul_563 = None
        view_339: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_98, [1, 6144, 6]);  sum_98 = None
        convert_element_type_791: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        convert_element_type_792: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_1, torch.float32);  sigmoid_1 = None
        sub_56: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_792)
        mul_565: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_792, sub_56);  convert_element_type_792 = sub_56 = None
        mul_566: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_791, mul_565);  convert_element_type_791 = mul_565 = None
        convert_element_type_793: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_566, torch.bfloat16);  mul_566 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_340: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_793, [6144, 6]);  convert_element_type_793 = None
        permute_353: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_158: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_353, view_30);  permute_353 = view_30 = None
        mm_159: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_340, permute_355);  view_340 = permute_355 = None
        view_341: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_159, [1, 6144, 12]);  mm_159 = None
        convert_element_type_798: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_18: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_341, 2, 0, 12);  view_341 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_357: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_564, [0, 2, 1, 3]);  mul_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph9 = self.fw_graph9
        joint_graph9 = self.joint_graph9
        mask_graph9 = self.mask_graph9
        flex_attention_backward_9 = torch.ops.higher_order.flex_attention_backward(permute_15, permute_16, permute_17, getitem_29, getitem_30, permute_357, None, fw_graph9, joint_graph9, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, mask_graph9), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_15 = permute_16 = permute_17 = getitem_29 = getitem_30 = permute_357 = fw_graph9 = joint_graph9 = clamp_max_2 = clamp_max_3 = convert_element_type_6 = clone_10 = convert_element_type_8 = clone_13 = mask_graph9 = None
        getitem_163: "bf16[1, 6, 6144, 128]" = flex_attention_backward_9[0]
        getitem_164: "bf16[1, 6, 6144, 128]" = flex_attention_backward_9[1]
        getitem_165: "bf16[1, 6, 6144, 128]" = flex_attention_backward_9[2];  flex_attention_backward_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_358: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_359: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_164, [0, 2, 1, 3]);  getitem_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_360: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3]);  getitem_163 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_10: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_14: "f32[]" = torch.ops.aten.select.int(select_10, 0, 1)
        mul_567: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_358, select_14);  select_14 = None
        mul_568: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_358, view_28);  view_28 = None
        sum_99: "bf16[]" = torch.ops.aten.sum.default(mul_568);  mul_568 = None
        convert_element_type_799: "f32[]" = torch.ops.prims.convert_element_type.default(sum_99, torch.float32);  sum_99 = None
        view_343: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_567, [6144, 768]);  mul_567 = None
        add_263: "bf16[6144, 768]" = torch.ops.aten.add.Tensor(view_216, view_343);  view_216 = view_343 = None
        select_scatter_68: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_799, 0, 1);  convert_element_type_799 = None
        select_13: "f32[]" = torch.ops.aten.select.int(select_10, 0, 0);  select_10 = None
        mul_569: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_358, select_13);  select_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_26: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_5, [1, 6144, 2304]);  mm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_27: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_26, [1, 6144, 18, 128]);  view_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_3 = torch.ops.aten.split.Tensor(view_27, 6, -2);  view_27 = None
        getitem_24: "bf16[1, 6144, 6, 128]" = split_3[2]
        getitem_23: "bf16[1, 6144, 6, 128]" = split_3[1]
        getitem_22: "bf16[1, 6144, 6, 128]" = split_3[0];  split_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        mul_570: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_358, getitem_24);  permute_358 = getitem_24 = None
        sum_100: "bf16[]" = torch.ops.aten.sum.default(mul_570);  mul_570 = None
        convert_element_type_800: "f32[]" = torch.ops.prims.convert_element_type.default(sum_100, torch.float32);  sum_100 = None
        select_scatter_69: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_800, 0, 0);  convert_element_type_800 = None
        add_264: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_68, select_scatter_69);  select_scatter_68 = select_scatter_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_801: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_359, torch.float32);  permute_359 = None
        slice_114: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_801, 3, 0, 64)
        slice_115: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_801, 3, 64, 128);  convert_element_type_801 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_51: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_15, 0);  primals_15 = None
        slice_19: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_51, 1, 0, 6144);  unsqueeze_51 = None
        unsqueeze_52: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_19, 2);  slice_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_571: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_115, unsqueeze_52)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_53: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_16, 0);  primals_16 = None
        slice_20: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_53, 1, 0, 6144);  unsqueeze_53 = None
        unsqueeze_54: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_20, 2);  slice_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_2: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_54)
        mul_572: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_115, neg_2);  slice_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_573: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_114, unsqueeze_54)
        add_265: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_571, mul_573);  mul_571 = mul_573 = None
        mul_574: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_114, unsqueeze_52);  slice_114 = None
        add_266: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_572, mul_574);  mul_572 = mul_574 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_49: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_266, add_265], 3);  add_266 = add_265 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_803: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_360, torch.float32);  permute_360 = None
        slice_116: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_803, 3, 0, 64)
        slice_117: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_803, 3, 64, 128);  convert_element_type_803 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_575: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_117, unsqueeze_52)
        mul_576: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_117, neg_2);  slice_117 = neg_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_577: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_116, unsqueeze_54);  unsqueeze_54 = None
        add_267: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_575, mul_577);  mul_575 = mul_577 = None
        mul_578: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_116, unsqueeze_52);  slice_116 = unsqueeze_52 = None
        add_268: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_576, mul_578);  mul_576 = mul_578 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_50: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_268, add_267], 3);  add_268 = add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_3: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_49, torch.float32);  cat_49 = None
        convert_element_type_46: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
        mul_22: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_46, rsqrt_7);  convert_element_type_46 = None
        mul_580: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_22, convert_element_type_default_3)
        sum_101: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_580, [3], True);  mul_580 = None
        div_42: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_22, 128);  mul_22 = None
        mul_581: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_42, sum_101);  div_42 = sum_101 = None
        sub_57: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_3, mul_581);  convert_element_type_default_3 = mul_581 = None
        mul_582: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_7);  sub_57 = rsqrt_7 = None
        convert_element_type_807: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_582, torch.bfloat16);  mul_582 = None
        convert_element_type_default_2: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_50, torch.float32);  cat_50 = None
        convert_element_type_44: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        mul_21: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_44, rsqrt_6);  convert_element_type_44 = None
        mul_584: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_21, convert_element_type_default_2)
        sum_102: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [3], True);  mul_584 = None
        div_43: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_21, 128);  mul_21 = None
        mul_585: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_43, sum_102);  div_43 = sum_102 = None
        sub_58: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, mul_585);  convert_element_type_default_2 = mul_585 = None
        mul_586: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_6);  sub_58 = rsqrt_6 = None
        convert_element_type_810: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_586, torch.bfloat16);  mul_586 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_51: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_810, convert_element_type_807, mul_569], 2);  convert_element_type_810 = convert_element_type_807 = mul_569 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_344: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_51, [1, 6144, 2304]);  cat_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_345: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_344, [6144, 2304]);  view_344 = None
        permute_361: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_345, [1, 0])
        mm_160: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_361, view_25);  permute_361 = view_25 = None
        mm_161: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_345, permute_363);  view_345 = permute_363 = None
        view_346: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_161, [1, 6144, 768]);  mm_161 = None
        add_269: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_18, view_346);  slice_scatter_18 = view_346 = None
        convert_element_type_815: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        view_347: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_815, [3, 768, 768]);  convert_element_type_815 = None
        slice_scatter_19: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_347, 0, 0, 3);  view_347 = None
        add_270: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_67, slice_scatter_19);  select_scatter_67 = slice_scatter_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_816: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_269, torch.float32);  add_269 = None
        convert_element_type_39: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        mul_20: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_39, rsqrt_5);  convert_element_type_39 = None
        mul_588: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_20, convert_element_type_816)
        sum_103: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True);  mul_588 = None
        div_44: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_20, 768);  mul_20 = None
        mul_589: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_44, sum_103);  div_44 = sum_103 = None
        sub_59: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_816, mul_589);  convert_element_type_816 = mul_589 = None
        mul_590: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_5);  sub_59 = rsqrt_5 = None
        convert_element_type_818: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_590, torch.bfloat16);  mul_590 = None
        add_271: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_262, convert_element_type_818);  add_262 = convert_element_type_818 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_9: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_12: "f32[]" = torch.ops.aten.select.int(select_9, 0, 1)
        mul_591: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_271, select_12);  select_12 = None
        mul_592: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_271, convert_element_type_11)
        sum_104: "bf16[]" = torch.ops.aten.sum.default(mul_592);  mul_592 = None
        convert_element_type_819: "f32[]" = torch.ops.prims.convert_element_type.default(sum_104, torch.float32);  sum_104 = None
        add_272: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_257, mul_591);  add_257 = mul_591 = None
        select_scatter_70: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_819, 0, 1);  convert_element_type_819 = None
        select_11: "f32[]" = torch.ops.aten.select.int(select_9, 0, 0);  select_9 = None
        mul_593: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_271, select_11);  select_11 = None
        mul_594: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_271, add_12);  add_271 = add_12 = None
        sum_105: "bf16[]" = torch.ops.aten.sum.default(mul_594);  mul_594 = None
        convert_element_type_820: "f32[]" = torch.ops.prims.convert_element_type.default(sum_105, torch.float32);  sum_105 = None
        add_273: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_241, mul_593);  mul_241 = mul_593 = None
        select_scatter_71: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_820, 0, 0);  convert_element_type_820 = None
        add_274: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_70, select_scatter_71);  select_scatter_70 = select_scatter_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_72: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_264, 0, 1);  add_264 = None
        add_275: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_260, select_scatter_72);  add_260 = select_scatter_72 = None
        select_scatter_73: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_274, 0, 1);  add_274 = None
        add_276: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_261, select_scatter_73);  add_261 = select_scatter_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_348: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_273, [6144, 768])
        permute_365: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_348, [1, 0])
        mm_162: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_365, view_22);  permute_365 = view_22 = None
        mm_163: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_348, permute_367);  view_348 = permute_367 = None
        view_349: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_163, [1, 6144, 3072]);  mm_163 = None
        convert_element_type_825: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_21: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_3, [1, 6144, 3072]);  mm_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_21);  view_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_71: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu, 1.0)
        mul_595: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_71, 2.0);  pow_71 = None
        mul_596: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(view_349, mul_595);  view_349 = mul_595 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_12: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_19: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_12, full_default_17, mul_596);  le_12 = full_default_17 = mul_596 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_350: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_19, [6144, 3072]);  where_19 = None
        mm_164: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_369, view_350);  permute_369 = None
        mm_165: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_350, permute_370);  view_350 = permute_370 = None
        view_351: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_165, [1, 6144, 768]);  mm_165 = None
        permute_371: "bf16[3072, 768]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
        convert_element_type_830: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_371, torch.float32);  permute_371 = None
        permute_372: "f32[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_830, [1, 0]);  convert_element_type_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_831: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(view_351, torch.float32);  view_351 = None
        convert_element_type_31: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        mul_17: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, rsqrt_4);  convert_element_type_31 = None
        mul_598: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_17, convert_element_type_831)
        sum_106: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True);  mul_598 = None
        div_45: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_17, 768);  mul_17 = None
        mul_599: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_45, sum_106);  div_45 = sum_106 = None
        sub_60: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_831, mul_599);  convert_element_type_831 = mul_599 = None
        mul_600: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_4);  sub_60 = rsqrt_4 = None
        convert_element_type_833: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_600, torch.bfloat16);  mul_600 = None
        add_277: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_273, convert_element_type_833);  add_273 = convert_element_type_833 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_352: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_277, [6144, 768])
        permute_373: "bf16[768, 6144]" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_166: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_373, view_18);  permute_373 = view_18 = None
        mm_167: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_352, permute_375);  view_352 = permute_375 = None
        view_353: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_167, [1, 6144, 768]);  mm_167 = None
        convert_element_type_838: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        select_scatter_74: "f32[4, 768, 768]" = torch.ops.aten.select_scatter.default(full_default_18, convert_element_type_838, 0, 3);  convert_element_type_838 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_354: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(view_353, [1, 6144, 6, 128]);  view_353 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_8: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_601: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_354, permute_8);  permute_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_15: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(mm_1, [1, 6144, 6]);  mm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(view_15);  view_15 = None
        view_16: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid, [1, 6144, 6, 1])
        mul_602: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_354, view_16);  view_354 = view_16 = None
        sum_107: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [3], True);  mul_601 = None
        view_355: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_107, [1, 6144, 6]);  sum_107 = None
        convert_element_type_839: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        convert_element_type_840: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32);  sigmoid = None
        sub_61: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_840)
        mul_603: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_840, sub_61);  convert_element_type_840 = sub_61 = None
        mul_604: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_839, mul_603);  convert_element_type_839 = mul_603 = None
        convert_element_type_841: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_604, torch.bfloat16);  mul_604 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_356: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_841, [6144, 6]);  convert_element_type_841 = None
        permute_377: "bf16[6, 6144]" = torch.ops.aten.permute.default(view_356, [1, 0])
        mm_168: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_377, view_14);  permute_377 = view_14 = None
        mm_169: "bf16[6144, 12]" = torch.ops.aten.mm.default(view_356, permute_379);  view_356 = permute_379 = None
        view_357: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_169, [1, 6144, 12]);  mm_169 = None
        convert_element_type_846: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_20: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, view_357, 2, 0, 12);  full_default_19 = view_357 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_381: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_602, [0, 2, 1, 3]);  mul_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        fw_graph10 = self.fw_graph10
        joint_graph10 = self.joint_graph10
        mask_graph10 = self.mask_graph10
        flex_attention_backward_10 = torch.ops.higher_order.flex_attention_backward(permute_5, permute_6, permute_7, getitem_19, getitem_20, permute_381, None, fw_graph10, joint_graph10, (6144, 6144, clamp_max, unsqueeze_9, clamp_max_1, unsqueeze_13, convert_element_type_2, clone_4, convert_element_type_4, clone_7, 128, 128, mask_graph10), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_5 = permute_6 = permute_7 = getitem_19 = getitem_20 = permute_381 = fw_graph10 = joint_graph10 = clamp_max = unsqueeze_9 = clamp_max_1 = unsqueeze_13 = convert_element_type_2 = clone_4 = convert_element_type_4 = clone_7 = mask_graph10 = cumsum = None
        getitem_167: "bf16[1, 6, 6144, 128]" = flex_attention_backward_10[0]
        getitem_168: "bf16[1, 6, 6144, 128]" = flex_attention_backward_10[1]
        getitem_169: "bf16[1, 6, 6144, 128]" = flex_attention_backward_10[2];  flex_attention_backward_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_382: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_169, [0, 2, 1, 3]);  getitem_169 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_383: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_384: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_3: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 0);  view_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_7: "f32[]" = torch.ops.aten.select.int(select_3, 0, 1)
        mul_605: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_382, select_7);  select_7 = None
        mul_606: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_382, view_12);  view_12 = None
        sum_108: "bf16[]" = torch.ops.aten.sum.default(mul_606);  mul_606 = None
        convert_element_type_847: "f32[]" = torch.ops.prims.convert_element_type.default(sum_108, torch.float32);  sum_108 = None
        view_359: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_605, [6144, 768]);  mul_605 = None
        add_278: "bf16[6144, 768]" = torch.ops.aten.add.Tensor(view_232, view_359);  view_232 = view_359 = None
        select_scatter_75: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_847, 0, 1);  convert_element_type_847 = None
        select_6: "f32[]" = torch.ops.aten.select.int(select_3, 0, 0);  select_3 = None
        mul_607: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_382, select_6);  select_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_10: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm, [1, 6144, 2304]);  mm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_11: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_10, [1, 6144, 18, 128]);  view_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split = torch.ops.aten.split.Tensor(view_11, 6, -2);  view_11 = None
        getitem_14: "bf16[1, 6144, 6, 128]" = split[2]
        getitem_13: "bf16[1, 6144, 6, 128]" = split[1]
        getitem_12: "bf16[1, 6144, 6, 128]" = split[0];  split = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        mul_608: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_382, getitem_14);  permute_382 = getitem_14 = None
        sum_109: "bf16[]" = torch.ops.aten.sum.default(mul_608);  mul_608 = None
        convert_element_type_848: "f32[]" = torch.ops.prims.convert_element_type.default(sum_109, torch.float32);  sum_109 = None
        select_scatter_76: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_848, 0, 0);  convert_element_type_848 = None
        add_279: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_75, select_scatter_76);  select_scatter_75 = select_scatter_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_849: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_383, torch.float32);  permute_383 = None
        slice_118: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_849, 3, 0, 64)
        slice_119: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_849, 3, 64, 128);  convert_element_type_849 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_43: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_9, 0);  primals_9 = None
        slice_13: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_43, 1, 0, 6144);  unsqueeze_43 = None
        unsqueeze_44: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_13, 2);  slice_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_609: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_119, unsqueeze_44)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_45: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_10, 0);  primals_10 = None
        slice_14: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_45, 1, 0, 6144);  unsqueeze_45 = None
        unsqueeze_46: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_14, 2);  slice_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_46)
        mul_610: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_119, neg);  slice_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_611: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_118, unsqueeze_46)
        add_280: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_609, mul_611);  mul_609 = mul_611 = None
        mul_612: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_118, unsqueeze_44);  slice_118 = None
        add_281: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_610, mul_612);  mul_610 = mul_612 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_52: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_281, add_280], 3);  add_281 = add_280 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_851: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_384, torch.float32);  permute_384 = None
        slice_120: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_851, 3, 0, 64)
        slice_121: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_851, 3, 64, 128);  convert_element_type_851 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_613: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_121, unsqueeze_44)
        mul_614: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_121, neg);  slice_121 = neg = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_615: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_120, unsqueeze_46);  unsqueeze_46 = None
        add_282: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_613, mul_615);  mul_613 = mul_615 = None
        mul_616: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_120, unsqueeze_44);  slice_120 = unsqueeze_44 = None
        add_283: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_614, mul_616);  mul_614 = mul_616 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_53: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_283, add_282], 3);  add_283 = add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_1: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_52, torch.float32);  cat_52 = None
        convert_element_type_19: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        mul_5: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_19, rsqrt_3);  convert_element_type_19 = None
        mul_618: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_5, convert_element_type_default_1)
        sum_110: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [3], True);  mul_618 = None
        div_46: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_5, 128);  mul_5 = None
        mul_619: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_46, sum_110);  div_46 = sum_110 = None
        sub_62: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_619);  convert_element_type_default_1 = mul_619 = None
        mul_620: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_3);  sub_62 = rsqrt_3 = None
        convert_element_type_855: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_620, torch.bfloat16);  mul_620 = None
        convert_element_type_default: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_53, torch.float32);  cat_53 = None
        convert_element_type_17: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_12, torch.float32);  getitem_12 = None
        mul_4: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_2);  convert_element_type_17 = None
        mul_622: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_4, convert_element_type_default)
        sum_111: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [3], True);  mul_622 = None
        div_47: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_4, 128);  mul_4 = None
        mul_623: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_47, sum_111);  div_47 = sum_111 = None
        sub_63: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_623);  convert_element_type_default = mul_623 = None
        mul_624: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_2);  sub_63 = rsqrt_2 = None
        convert_element_type_858: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_624, torch.bfloat16);  mul_624 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_54: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_858, convert_element_type_855, mul_607], 2);  convert_element_type_858 = convert_element_type_855 = mul_607 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_360: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_54, [1, 6144, 2304]);  cat_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_361: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(view_360, [6144, 2304]);  view_360 = None
        permute_385: "bf16[2304, 6144]" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_170: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_385, view_9);  permute_385 = view_9 = None
        mm_171: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_361, permute_387);  view_361 = permute_387 = None
        view_362: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_171, [1, 6144, 768]);  mm_171 = None
        add_284: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_20, view_362);  slice_scatter_20 = view_362 = None
        convert_element_type_863: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        view_363: "f32[3, 768, 768]" = torch.ops.aten.reshape.default(convert_element_type_863, [3, 768, 768]);  convert_element_type_863 = None
        slice_scatter_21: "f32[4, 768, 768]" = torch.ops.aten.slice_scatter.default(full_default_18, view_363, 0, 0, 3);  full_default_18 = view_363 = None
        add_285: "f32[4, 768, 768]" = torch.ops.aten.add.Tensor(select_scatter_74, slice_scatter_21);  select_scatter_74 = slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_864: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_284, torch.float32);  add_284 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_2: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 0);  view_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_4: "f32[]" = torch.ops.aten.select.int(select_2, 0, 0)
        mul_1: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_4, convert_element_type_11)
        select_5: "f32[]" = torch.ops.aten.select.int(select_2, 0, 1);  select_2 = None
        mul_2: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_5, convert_element_type_11)
        add_1: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_12: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        mul_3: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, rsqrt_1);  convert_element_type_12 = None
        mul_626: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_3, convert_element_type_864)
        sum_112: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [2], True);  mul_626 = None
        div_48: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_3, 768);  mul_3 = None
        mul_627: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_48, sum_112);  div_48 = sum_112 = None
        sub_64: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_864, mul_627);  convert_element_type_864 = mul_627 = None
        mul_628: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_1);  sub_64 = rsqrt_1 = None
        convert_element_type_866: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_628, torch.bfloat16);  mul_628 = None
        add_286: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_277, convert_element_type_866);  add_277 = convert_element_type_866 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_629: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_286, select_5);  select_5 = None
        mul_630: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_286, convert_element_type_11);  convert_element_type_11 = None
        sum_113: "bf16[]" = torch.ops.aten.sum.default(mul_630);  mul_630 = None
        convert_element_type_867: "f32[]" = torch.ops.prims.convert_element_type.default(sum_113, torch.float32);  sum_113 = None
        add_287: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_272, mul_629);  add_272 = mul_629 = None
        select_scatter_77: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_867, 0, 1)
        mul_631: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_286, select_4);  add_286 = select_4 = None
        add_288: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_287, mul_631);  add_287 = mul_631 = None
        select_scatter_78: "f32[2]" = torch.ops.aten.select_scatter.default(full_default_20, convert_element_type_867, 0, 0);  full_default_20 = convert_element_type_867 = None
        add_289: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_77, select_scatter_78);  select_scatter_77 = select_scatter_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_79: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_279, 0, 0);  add_279 = None
        add_290: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_275, select_scatter_79);  add_275 = select_scatter_79 = None
        select_scatter_80: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_25, add_289, 0, 0);  full_default_25 = add_289 = None
        add_291: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_276, select_scatter_80);  add_276 = select_scatter_80 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:915 in forward, code: sa_lambdas = self.scalars[3 * len(self.blocks) : 5 * len(self.blocks)].view(
        view_364: "f32[24]" = torch.ops.aten.reshape.default(add_290, [24]);  add_290 = None
        full_default_132: "f32[64]" = torch.ops.aten.full.default([64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_22: "f32[64]" = torch.ops.aten.slice_scatter.default(full_default_132, view_364, 0, 36, 60);  view_364 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:914 in forward, code: lambdas = self.scalars[1 * len(self.blocks) : 3 * len(self.blocks)].view(-1, 2)
        view_365: "f32[24]" = torch.ops.aten.reshape.default(add_291, [24]);  add_291 = None
        slice_scatter_23: "f32[64]" = torch.ops.aten.slice_scatter.default(full_default_132, view_365, 0, 12, 36);  view_365 = None
        add_292: "f32[64]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:913 in forward, code: skip_weights = self.scalars[: (len(self.blocks) // 2)]
        slice_scatter_24: "f32[64]" = torch.ops.aten.slice_scatter.default(full_default_132, add_207, 0, 0, 6);  full_default_132 = add_207 = None
        add_293: "f32[64]" = torch.ops.aten.add.Tensor(add_292, slice_scatter_24);  add_292 = slice_scatter_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_869: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_288, torch.float32);  add_288 = None
        mul_634: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul, convert_element_type_869)
        sum_115: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [2], True);  mul_634 = None
        div_49: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul, 768);  mul = None
        mul_635: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_49, sum_115);  div_49 = sum_115 = None
        sub_65: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_869, mul_635);  convert_element_type_869 = mul_635 = None
        mul_636: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt);  sub_65 = rsqrt = None
        convert_element_type_871: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_636, torch.bfloat16);  mul_636 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:909 in forward, code: x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        squeeze_1: "bf16[6144, 768]" = torch.ops.aten.squeeze.dim(convert_element_type_871, 0);  convert_element_type_871 = None
        convert_element_type_872: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        convert_element_type_873: "i64[6144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.int64);  primals_1 = None
        eq_3: "b8[6144]" = torch.ops.aten.eq.Scalar(convert_element_type_873, -1)
        unsqueeze_133: "b8[6144, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
        where_20: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_133, full_default_13, convert_element_type_872);  convert_element_type_872 = None
        full_default_136: "f32[50304, 768]" = torch.ops.aten.full.default([50304, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_4: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_136, [convert_element_type_873], where_20, True);  where_20 = None
        convert_element_type_874: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_4, torch.bfloat16);  index_put_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:883 in forward, code: ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        convert_element_type_875: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(add_248, torch.float32);  add_248 = None
        where_21: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_133, full_default_13, convert_element_type_875);  convert_element_type_875 = None
        index_put_5: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_136, [convert_element_type_873], where_21, True);  where_21 = None
        convert_element_type_877: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_5, torch.bfloat16);  index_put_5 = None
        convert_element_type_878: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(add_263, torch.float32);  add_263 = None
        where_22: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_133, full_default_13, convert_element_type_878);  convert_element_type_878 = None
        index_put_6: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_136, [convert_element_type_873], where_22, True);  where_22 = None
        convert_element_type_880: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_6, torch.bfloat16);  index_put_6 = None
        convert_element_type_881: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(add_278, torch.float32);  add_278 = None
        where_23: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_133, full_default_13, convert_element_type_881);  unsqueeze_133 = full_default_13 = convert_element_type_881 = None
        index_put_7: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_136, [convert_element_type_873], where_23, True);  full_default_136 = convert_element_type_873 = where_23 = None
        convert_element_type_883: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_7, torch.bfloat16);  index_put_7 = None
        return (None, convert_element_type_883, convert_element_type_880, convert_element_type_877, None, convert_element_type_874, add_293, add_285, None, None, convert_element_type_846, permute_372, convert_element_type_825, add_270, None, None, convert_element_type_798, permute_348, convert_element_type_777, add_255, None, None, convert_element_type_750, permute_324, convert_element_type_729, add_240, None, None, convert_element_type_703, permute_300, convert_element_type_682, add_227, None, None, convert_element_type_656, permute_276, convert_element_type_635, add_214, None, None, convert_element_type_609, permute_252, convert_element_type_588, add_200, None, None, convert_element_type_561, permute_228, convert_element_type_540, permute_220, convert_element_type_524, add_182, None, None, convert_element_type_497, permute_196, convert_element_type_476, add_169, None, None, convert_element_type_448, permute_172, convert_element_type_427, add_155, None, None, convert_element_type_399, permute_148, convert_element_type_378, add_145, None, None, convert_element_type_350, permute_124, convert_element_type_329, getitem_126, None)

    class fw_graph0(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph0(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph0(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph1(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph1(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph1(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph2(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph2(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph2(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph3(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph3(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph3(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph4(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph4(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph4(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph5(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph5(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph5(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph6(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph6(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph6(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph7(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph7(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph7(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph8(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph8(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph8(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph9(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph9(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph9(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class fw_graph10(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class joint_graph10(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "bf16[]"):
            return [arg5_1, None, None, None, None]

    class mask_graph10(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and
