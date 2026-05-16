"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: 16ef771ba559
Shape hash: 9472c803
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 1000]", _shape_param_0, select_scatter: "f32[4, 197, 768]", mul_86: "f32[4, 197, 768]", view_182: "f32[788, 768]", _shape_param_1, view_185: "f32[788, 3072]", _shape_param_2, view_187: "f32[4, 197, 768]", mul_100: "f32[4, 197, 768]", view_188: "f32[788, 768]", _shape_param_3, view_197: "f32[197, 4, 2304]", _shape_param_4, view_199: "f32[788, 2304]", permute_160: "f32[4, 197, 768]", mul_107: "f32[4, 197, 768]", view_201: "f32[788, 768]", _shape_param_5, view_204: "f32[788, 3072]", _shape_param_6, view_206: "f32[4, 197, 768]", mul_121: "f32[4, 197, 768]", view_207: "f32[788, 768]", _shape_param_7, view_216: "f32[197, 4, 2304]", _shape_param_8, view_218: "f32[788, 2304]", permute_183: "f32[4, 197, 768]", mul_128: "f32[4, 197, 768]", view_220: "f32[788, 768]", _shape_param_9, view_223: "f32[788, 3072]", _shape_param_10, view_225: "f32[4, 197, 768]", mul_142: "f32[4, 197, 768]", view_226: "f32[788, 768]", _shape_param_11, view_235: "f32[197, 4, 2304]", _shape_param_12, view_237: "f32[788, 2304]", permute_206: "f32[4, 197, 768]", mul_149: "f32[4, 197, 768]", view_239: "f32[788, 768]", _shape_param_13, view_242: "f32[788, 3072]", _shape_param_14, view_244: "f32[4, 197, 768]", mul_163: "f32[4, 197, 768]", view_245: "f32[788, 768]", _shape_param_15, view_254: "f32[197, 4, 2304]", _shape_param_16, view_256: "f32[788, 2304]", permute_229: "f32[4, 197, 768]", mul_170: "f32[4, 197, 768]", view_258: "f32[788, 768]", _shape_param_17, view_261: "f32[788, 3072]", _shape_param_18, view_263: "f32[4, 197, 768]", mul_184: "f32[4, 197, 768]", view_264: "f32[788, 768]", _shape_param_19, view_273: "f32[197, 4, 2304]", _shape_param_20, view_275: "f32[788, 2304]", permute_252: "f32[4, 197, 768]", mul_191: "f32[4, 197, 768]", view_277: "f32[788, 768]", _shape_param_21, view_280: "f32[788, 3072]", _shape_param_22, view_282: "f32[4, 197, 768]", mul_205: "f32[4, 197, 768]", view_283: "f32[788, 768]", _shape_param_23, view_292: "f32[197, 4, 2304]", _shape_param_24, view_294: "f32[788, 2304]", permute_275: "f32[4, 197, 768]", mul_212: "f32[4, 197, 768]", view_296: "f32[788, 768]", _shape_param_25, view_299: "f32[788, 3072]", _shape_param_26, view_301: "f32[4, 197, 768]", mul_226: "f32[4, 197, 768]", view_302: "f32[788, 768]", _shape_param_27, view_311: "f32[197, 4, 2304]", _shape_param_28, view_313: "f32[788, 2304]", permute_298: "f32[4, 197, 768]", mul_233: "f32[4, 197, 768]", view_315: "f32[788, 768]", _shape_param_29, view_318: "f32[788, 3072]", _shape_param_30, view_320: "f32[4, 197, 768]", mul_247: "f32[4, 197, 768]", view_321: "f32[788, 768]", _shape_param_31, view_330: "f32[197, 4, 2304]", _shape_param_32, view_332: "f32[788, 2304]", permute_321: "f32[4, 197, 768]", mul_254: "f32[4, 197, 768]", view_334: "f32[788, 768]", _shape_param_33, view_337: "f32[788, 3072]", _shape_param_34, view_339: "f32[4, 197, 768]", mul_268: "f32[4, 197, 768]", view_340: "f32[788, 768]", _shape_param_35, view_349: "f32[197, 4, 2304]", _shape_param_36, view_351: "f32[788, 2304]", permute_344: "f32[4, 197, 768]", mul_275: "f32[4, 197, 768]", view_353: "f32[788, 768]", _shape_param_37, view_356: "f32[788, 3072]", _shape_param_38, view_358: "f32[4, 197, 768]", mul_289: "f32[4, 197, 768]", view_359: "f32[788, 768]", _shape_param_39, view_368: "f32[197, 4, 2304]", _shape_param_40, view_370: "f32[788, 2304]", permute_367: "f32[4, 197, 768]", mul_296: "f32[4, 197, 768]", view_372: "f32[788, 768]", _shape_param_41, view_375: "f32[788, 3072]", _shape_param_42, view_377: "f32[4, 197, 768]", mul_310: "f32[4, 197, 768]", view_378: "f32[788, 768]", _shape_param_43, view_387: "f32[197, 4, 2304]", _shape_param_44, view_389: "f32[788, 2304]", permute_390: "f32[4, 197, 768]", mul_317: "f32[4, 197, 768]", view_391: "f32[788, 768]", _shape_param_45, view_394: "f32[788, 3072]", _shape_param_46, view_396: "f32[4, 197, 768]", mul_331: "f32[4, 197, 768]", view_397: "f32[788, 768]", _shape_param_47, view_406: "f32[197, 4, 2304]", _shape_param_48, view_408: "f32[788, 2304]", mm_109: "f32[788, 768]", _shape_param_49, primals_6: "f32[768]", cat: "f32[4, 197, 768]", primals_5: "f32[1, 197, 768]", getitem_1: "f32[4, 197, 1]", rsqrt: "f32[4, 197, 1]", add_167: "f32[4, 197, 768]", _shape_param_50):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:303 in forward, code: x = self.heads(x)
        permute_default: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:157 in forward, code: return self.ln(self.layers(self.dropout(input)))
        mul_tensor: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(select_scatter, mul_86);  mul_86 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_1: "f32[768, 788]" = torch.ops.aten.permute.default(view_182, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None
        permute_default_2: "f32[3072, 788]" = torch.ops.aten.permute.default(view_185, [1, 0])
        sum_dim_int_list_4: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
        reshape_default_2: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_187, mul_100);  mul_100 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_187, [0, 1]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_3: "f32[768, 788]" = torch.ops.aten.permute.default(view_188, [1, 0])
        sum_dim_int_list_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None
        sum_dim_int_list_8: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_197, [0, 1], True);  view_197 = None
        reshape_default_4: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_4);  sum_dim_int_list_8 = _shape_param_4 = None
        permute_default_4: "f32[2304, 788]" = torch.ops.aten.permute.default(view_199, [1, 0]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_2: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_160, mul_107);  mul_107 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_160, [0, 1]);  permute_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_5: "f32[768, 788]" = torch.ops.aten.permute.default(view_201, [1, 0])
        sum_dim_int_list_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None
        permute_default_6: "f32[3072, 788]" = torch.ops.aten.permute.default(view_204, [1, 0])
        sum_dim_int_list_12: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
        reshape_default_6: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_3: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_206, mul_121);  mul_121 = None
        sum_dim_int_list_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_206, [0, 1]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_7: "f32[768, 788]" = torch.ops.aten.permute.default(view_207, [1, 0])
        sum_dim_int_list_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_7);  sum_dim_int_list_15 = _shape_param_7 = None
        sum_dim_int_list_16: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 1], True);  view_216 = None
        reshape_default_8: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None
        permute_default_8: "f32[2304, 788]" = torch.ops.aten.permute.default(view_218, [1, 0]);  view_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_4: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_183, mul_128);  mul_128 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_183, [0, 1]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_9: "f32[768, 788]" = torch.ops.aten.permute.default(view_220, [1, 0])
        sum_dim_int_list_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_220, [0], True);  view_220 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_9);  sum_dim_int_list_19 = _shape_param_9 = None
        permute_default_10: "f32[3072, 788]" = torch.ops.aten.permute.default(view_223, [1, 0])
        sum_dim_int_list_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_223, [0], True);  view_223 = None
        reshape_default_10: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_5: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_225, mul_142);  mul_142 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 1]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_11: "f32[768, 788]" = torch.ops.aten.permute.default(view_226, [1, 0])
        sum_dim_int_list_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_11);  sum_dim_int_list_23 = _shape_param_11 = None
        sum_dim_int_list_24: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1], True);  view_235 = None
        reshape_default_12: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_12);  sum_dim_int_list_24 = _shape_param_12 = None
        permute_default_12: "f32[2304, 788]" = torch.ops.aten.permute.default(view_237, [1, 0]);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_6: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_206, mul_149);  mul_149 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_206, [0, 1]);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_13: "f32[768, 788]" = torch.ops.aten.permute.default(view_239, [1, 0])
        sum_dim_int_list_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_13);  sum_dim_int_list_27 = _shape_param_13 = None
        permute_default_14: "f32[3072, 788]" = torch.ops.aten.permute.default(view_242, [1, 0])
        sum_dim_int_list_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
        reshape_default_14: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_14);  sum_dim_int_list_28 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_7: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_244, mul_163);  mul_163 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_244, [0, 1]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_15: "f32[768, 788]" = torch.ops.aten.permute.default(view_245, [1, 0])
        sum_dim_int_list_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_15);  sum_dim_int_list_31 = _shape_param_15 = None
        sum_dim_int_list_32: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_254, [0, 1], True);  view_254 = None
        reshape_default_16: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_16);  sum_dim_int_list_32 = _shape_param_16 = None
        permute_default_16: "f32[2304, 788]" = torch.ops.aten.permute.default(view_256, [1, 0]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_8: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_229, mul_170);  mul_170 = None
        sum_dim_int_list_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_229, [0, 1]);  permute_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_17: "f32[768, 788]" = torch.ops.aten.permute.default(view_258, [1, 0])
        sum_dim_int_list_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        reshape_default_17: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_17);  sum_dim_int_list_35 = _shape_param_17 = None
        permute_default_18: "f32[3072, 788]" = torch.ops.aten.permute.default(view_261, [1, 0])
        sum_dim_int_list_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
        reshape_default_18: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_18);  sum_dim_int_list_36 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_9: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_263, mul_184);  mul_184 = None
        sum_dim_int_list_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_19: "f32[768, 788]" = torch.ops.aten.permute.default(view_264, [1, 0])
        sum_dim_int_list_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
        reshape_default_19: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_19);  sum_dim_int_list_39 = _shape_param_19 = None
        sum_dim_int_list_40: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1], True);  view_273 = None
        reshape_default_20: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_20);  sum_dim_int_list_40 = _shape_param_20 = None
        permute_default_20: "f32[2304, 788]" = torch.ops.aten.permute.default(view_275, [1, 0]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_10: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_252, mul_191);  mul_191 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_252, [0, 1]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_21: "f32[768, 788]" = torch.ops.aten.permute.default(view_277, [1, 0])
        sum_dim_int_list_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_21);  sum_dim_int_list_43 = _shape_param_21 = None
        permute_default_22: "f32[3072, 788]" = torch.ops.aten.permute.default(view_280, [1, 0])
        sum_dim_int_list_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        reshape_default_22: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_22);  sum_dim_int_list_44 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_11: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_282, mul_205);  mul_205 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 1]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_23: "f32[768, 788]" = torch.ops.aten.permute.default(view_283, [1, 0])
        sum_dim_int_list_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
        reshape_default_23: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_23);  sum_dim_int_list_47 = _shape_param_23 = None
        sum_dim_int_list_48: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_292, [0, 1], True);  view_292 = None
        reshape_default_24: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_24);  sum_dim_int_list_48 = _shape_param_24 = None
        permute_default_24: "f32[2304, 788]" = torch.ops.aten.permute.default(view_294, [1, 0]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_12: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_275, mul_212);  mul_212 = None
        sum_dim_int_list_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_275, [0, 1]);  permute_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_25: "f32[768, 788]" = torch.ops.aten.permute.default(view_296, [1, 0])
        sum_dim_int_list_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        reshape_default_25: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_25);  sum_dim_int_list_51 = _shape_param_25 = None
        permute_default_26: "f32[3072, 788]" = torch.ops.aten.permute.default(view_299, [1, 0])
        sum_dim_int_list_52: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        reshape_default_26: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_26);  sum_dim_int_list_52 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_13: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_301, mul_226);  mul_226 = None
        sum_dim_int_list_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1]);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_27: "f32[768, 788]" = torch.ops.aten.permute.default(view_302, [1, 0])
        sum_dim_int_list_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        reshape_default_27: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_27);  sum_dim_int_list_55 = _shape_param_27 = None
        sum_dim_int_list_56: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_311, [0, 1], True);  view_311 = None
        reshape_default_28: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_28);  sum_dim_int_list_56 = _shape_param_28 = None
        permute_default_28: "f32[2304, 788]" = torch.ops.aten.permute.default(view_313, [1, 0]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_14: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_298, mul_233);  mul_233 = None
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_298, [0, 1]);  permute_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_29: "f32[768, 788]" = torch.ops.aten.permute.default(view_315, [1, 0])
        sum_dim_int_list_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_29);  sum_dim_int_list_59 = _shape_param_29 = None
        permute_default_30: "f32[3072, 788]" = torch.ops.aten.permute.default(view_318, [1, 0])
        sum_dim_int_list_60: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        reshape_default_30: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_30);  sum_dim_int_list_60 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_15: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_320, mul_247);  mul_247 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_320, [0, 1]);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_31: "f32[768, 788]" = torch.ops.aten.permute.default(view_321, [1, 0])
        sum_dim_int_list_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_31);  sum_dim_int_list_63 = _shape_param_31 = None
        sum_dim_int_list_64: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 1], True);  view_330 = None
        reshape_default_32: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_32);  sum_dim_int_list_64 = _shape_param_32 = None
        permute_default_32: "f32[2304, 788]" = torch.ops.aten.permute.default(view_332, [1, 0]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_16: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_321, mul_254);  mul_254 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_321, [0, 1]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_33: "f32[768, 788]" = torch.ops.aten.permute.default(view_334, [1, 0])
        sum_dim_int_list_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        reshape_default_33: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_33);  sum_dim_int_list_67 = _shape_param_33 = None
        permute_default_34: "f32[3072, 788]" = torch.ops.aten.permute.default(view_337, [1, 0])
        sum_dim_int_list_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        reshape_default_34: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_34);  sum_dim_int_list_68 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_17: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_268);  mul_268 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_35: "f32[768, 788]" = torch.ops.aten.permute.default(view_340, [1, 0])
        sum_dim_int_list_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_35);  sum_dim_int_list_71 = _shape_param_35 = None
        sum_dim_int_list_72: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_349, [0, 1], True);  view_349 = None
        reshape_default_36: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_36);  sum_dim_int_list_72 = _shape_param_36 = None
        permute_default_36: "f32[2304, 788]" = torch.ops.aten.permute.default(view_351, [1, 0]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_18: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_344, mul_275);  mul_275 = None
        sum_dim_int_list_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_344, [0, 1]);  permute_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_37: "f32[768, 788]" = torch.ops.aten.permute.default(view_353, [1, 0])
        sum_dim_int_list_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
        reshape_default_37: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_37);  sum_dim_int_list_75 = _shape_param_37 = None
        permute_default_38: "f32[3072, 788]" = torch.ops.aten.permute.default(view_356, [1, 0])
        sum_dim_int_list_76: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
        reshape_default_38: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_38);  sum_dim_int_list_76 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_19: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_358, mul_289);  mul_289 = None
        sum_dim_int_list_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_358, [0, 1]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_39: "f32[768, 788]" = torch.ops.aten.permute.default(view_359, [1, 0])
        sum_dim_int_list_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
        reshape_default_39: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_39);  sum_dim_int_list_79 = _shape_param_39 = None
        sum_dim_int_list_80: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_368, [0, 1], True);  view_368 = None
        reshape_default_40: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_40);  sum_dim_int_list_80 = _shape_param_40 = None
        permute_default_40: "f32[2304, 788]" = torch.ops.aten.permute.default(view_370, [1, 0]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_20: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_367, mul_296);  mul_296 = None
        sum_dim_int_list_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_367, [0, 1]);  permute_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_41: "f32[768, 788]" = torch.ops.aten.permute.default(view_372, [1, 0])
        sum_dim_int_list_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_372, [0], True);  view_372 = None
        reshape_default_41: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_41);  sum_dim_int_list_83 = _shape_param_41 = None
        permute_default_42: "f32[3072, 788]" = torch.ops.aten.permute.default(view_375, [1, 0])
        sum_dim_int_list_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
        reshape_default_42: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_42);  sum_dim_int_list_84 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_21: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_377, mul_310);  mul_310 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_377, [0, 1]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_43: "f32[768, 788]" = torch.ops.aten.permute.default(view_378, [1, 0])
        sum_dim_int_list_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
        reshape_default_43: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_43);  sum_dim_int_list_87 = _shape_param_43 = None
        sum_dim_int_list_88: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_387, [0, 1], True);  view_387 = None
        reshape_default_44: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_44);  sum_dim_int_list_88 = _shape_param_44 = None
        permute_default_44: "f32[2304, 788]" = torch.ops.aten.permute.default(view_389, [1, 0]);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_22: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_390, mul_317);  mul_317 = None
        sum_dim_int_list_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_390, [0, 1]);  permute_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        permute_default_45: "f32[768, 788]" = torch.ops.aten.permute.default(view_391, [1, 0])
        sum_dim_int_list_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        reshape_default_45: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_45);  sum_dim_int_list_91 = _shape_param_45 = None
        permute_default_46: "f32[3072, 788]" = torch.ops.aten.permute.default(view_394, [1, 0])
        sum_dim_int_list_92: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        reshape_default_46: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_46);  sum_dim_int_list_92 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        mul_tensor_23: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(view_396, mul_331);  mul_331 = None
        sum_dim_int_list_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_396, [0, 1]);  view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default_47: "f32[768, 788]" = torch.ops.aten.permute.default(view_397, [1, 0])
        sum_dim_int_list_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
        reshape_default_47: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_47);  sum_dim_int_list_95 = _shape_param_47 = None
        sum_dim_int_list_96: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1], True);  view_406 = None
        reshape_default_48: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_48);  sum_dim_int_list_96 = _shape_param_48 = None
        permute_default_48: "f32[2304, 788]" = torch.ops.aten.permute.default(view_408, [1, 0]);  view_408 = None
        reshape_default_49: "f32[197, 4, 768]" = torch.ops.aten.reshape.default(mm_109, _shape_param_49);  mm_109 = _shape_param_49 = None
        permute_default_49: "f32[4, 197, 768]" = torch.ops.aten.permute.default(reshape_default_49, [1, 0, 2]);  reshape_default_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor_24: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_default_49, primals_6);  primals_6 = None
        mul_tensor_25: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_24, 768)
        sum_dim_int_list_97: "f32[4, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:156 in forward, code: input = input + self.pos_embedding
        add_tensor: "f32[4, 197, 768]" = torch.ops.aten.add.Tensor(cat, primals_5);  cat = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_26: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_27: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_24, mul_tensor_26);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[4, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True);  mul_tensor_27 = None
        mul_tensor_28: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, sum_dim_int_list_98);  sum_dim_int_list_98 = None
        sub_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_25, sum_dim_int_list_97);  mul_tensor_25 = sum_dim_int_list_97 = None
        sub_tensor_2: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_28);  sub_tensor_1 = mul_tensor_28 = None
        div_tensor: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_29: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_30: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_default_49, mul_tensor_26);  mul_tensor_26 = None
        sum_dim_int_list_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_default_49, [0, 1]);  permute_default_49 = None
        add_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.add.Tensor(add_167, mul_tensor_29);  add_167 = mul_tensor_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:156 in forward, code: input = input + self.pos_embedding
        sum_dim_int_list_101: "f32[1, 197, 768]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:296 in forward, code: x = torch.cat([batch_class_token, x], dim=1)
        slice_tensor: "f32[4, 1, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 0, 1)
        slice_tensor_1: "f32[4, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 1, 197);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:295 in forward, code: batch_class_token = self.class_token.expand(n, -1, -1)
        sum_dim_int_list_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:285 in _process_input, code: x = x.permute(0, 2, 1)
        permute_default_50: "f32[4, 768, 196]" = torch.ops.aten.permute.default(slice_tensor_1, [0, 2, 1]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:279 in _process_input, code: x = x.reshape(n, self.hidden_dim, n_h * n_w)
        reshape_default_50: "f32[4, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_default_50, _shape_param_50);  permute_default_50 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:277 in _process_input, code: x = self.conv_proj(x)
        sum_dim_int_list_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_50, [0, 2, 3]);  reshape_default_50 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, sum_dim_int_list_5, sum_dim_int_list_6, permute_default_3, reshape_default_3, reshape_default_4, permute_default_4, sum_dim_int_list_9, sum_dim_int_list_10, permute_default_5, reshape_default_5, permute_default_6, reshape_default_6, sum_dim_int_list_13, sum_dim_int_list_14, permute_default_7, reshape_default_7, reshape_default_8, permute_default_8, sum_dim_int_list_17, sum_dim_int_list_18, permute_default_9, reshape_default_9, permute_default_10, reshape_default_10, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_11, reshape_default_11, reshape_default_12, permute_default_12, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_13, reshape_default_13, permute_default_14, reshape_default_14, sum_dim_int_list_29, sum_dim_int_list_30, permute_default_15, reshape_default_15, reshape_default_16, permute_default_16, sum_dim_int_list_33, sum_dim_int_list_34, permute_default_17, reshape_default_17, permute_default_18, reshape_default_18, sum_dim_int_list_37, sum_dim_int_list_38, permute_default_19, reshape_default_19, reshape_default_20, permute_default_20, sum_dim_int_list_41, sum_dim_int_list_42, permute_default_21, reshape_default_21, permute_default_22, reshape_default_22, sum_dim_int_list_45, sum_dim_int_list_46, permute_default_23, reshape_default_23, reshape_default_24, permute_default_24, sum_dim_int_list_49, sum_dim_int_list_50, permute_default_25, reshape_default_25, permute_default_26, reshape_default_26, sum_dim_int_list_53, sum_dim_int_list_54, permute_default_27, reshape_default_27, reshape_default_28, permute_default_28, sum_dim_int_list_57, sum_dim_int_list_58, permute_default_29, reshape_default_29, permute_default_30, reshape_default_30, sum_dim_int_list_61, sum_dim_int_list_62, permute_default_31, reshape_default_31, reshape_default_32, permute_default_32, sum_dim_int_list_65, sum_dim_int_list_66, permute_default_33, reshape_default_33, permute_default_34, reshape_default_34, sum_dim_int_list_69, sum_dim_int_list_70, permute_default_35, reshape_default_35, reshape_default_36, permute_default_36, sum_dim_int_list_73, sum_dim_int_list_74, permute_default_37, reshape_default_37, permute_default_38, reshape_default_38, sum_dim_int_list_77, sum_dim_int_list_78, permute_default_39, reshape_default_39, reshape_default_40, permute_default_40, sum_dim_int_list_81, sum_dim_int_list_82, permute_default_41, reshape_default_41, permute_default_42, reshape_default_42, sum_dim_int_list_85, sum_dim_int_list_86, permute_default_43, reshape_default_43, reshape_default_44, permute_default_44, sum_dim_int_list_89, sum_dim_int_list_90, permute_default_45, reshape_default_45, permute_default_46, reshape_default_46, sum_dim_int_list_93, sum_dim_int_list_94, permute_default_47, reshape_default_47, reshape_default_48, permute_default_48, sum_dim_int_list_99, sum_dim_int_list_100, sum_dim_int_list_101, sum_dim_int_list_102, sum_dim_int_list_103)



def make_inputs():
    return [
    torch.randn([4, 1000], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_86
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_1
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_2
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_100
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_3
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_4
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_160
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_107
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_5
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_6
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_121
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_7
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_8
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_183
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_128
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_9
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_10
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_142
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_11
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_12
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_206
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_149
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_13
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_14
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_163
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_15
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_16
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_229
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_170
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_17
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_18
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_184
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_19
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_20
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_252
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_191
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_21
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_22
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_205
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_23
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_24
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_275
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_212
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_25
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_26
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_226
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_27
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_28
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_298
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_233
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_29
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_30
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_247
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_31
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_32
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_321
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_254
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_33
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_34
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_268
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_35
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_36
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_344
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_275
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_37
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_38
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_289
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_39
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_40
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_367
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_296
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_41
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_42
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_310
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_43
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_44
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # permute_390
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_317
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_45
    torch.randn([788, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_46
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_331
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_47
    torch.randn([197, 4, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_48
    torch.randn([788, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [197, 4, 768],  # _shape_param_49
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    [4, 768, 14, 14],  # _shape_param_50
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
