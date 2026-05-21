"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: 98b8d5a6936c
Shape hash: 102dc53d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[64, 1000]", sum_3: "f32[1024]", squeeze_361: "f32[1024]", sum_5: "f32[128]", squeeze_358: "f32[128]", sum_7: "f32[992]", squeeze_355: "f32[992]", sum_9: "f32[128]", squeeze_352: "f32[128]", sum_11: "f32[960]", squeeze_349: "f32[960]", sum_13: "f32[128]", squeeze_346: "f32[128]", sum_15: "f32[928]", squeeze_343: "f32[928]", sum_17: "f32[128]", squeeze_340: "f32[128]", sum_19: "f32[896]", squeeze_337: "f32[896]", sum_21: "f32[128]", squeeze_334: "f32[128]", sum_23: "f32[864]", squeeze_331: "f32[864]", sum_25: "f32[128]", squeeze_328: "f32[128]", sum_27: "f32[832]", squeeze_325: "f32[832]", sum_29: "f32[128]", squeeze_322: "f32[128]", sum_31: "f32[800]", squeeze_319: "f32[800]", sum_33: "f32[128]", squeeze_316: "f32[128]", sum_35: "f32[768]", squeeze_313: "f32[768]", sum_37: "f32[128]", squeeze_310: "f32[128]", sum_39: "f32[736]", squeeze_307: "f32[736]", sum_41: "f32[128]", squeeze_304: "f32[128]", sum_43: "f32[704]", squeeze_301: "f32[704]", sum_45: "f32[128]", squeeze_298: "f32[128]", sum_47: "f32[672]", squeeze_295: "f32[672]", sum_49: "f32[128]", squeeze_292: "f32[128]", sum_51: "f32[640]", squeeze_289: "f32[640]", sum_53: "f32[128]", squeeze_286: "f32[128]", sum_55: "f32[608]", squeeze_283: "f32[608]", sum_57: "f32[128]", squeeze_280: "f32[128]", sum_59: "f32[576]", squeeze_277: "f32[576]", sum_61: "f32[128]", squeeze_274: "f32[128]", sum_63: "f32[544]", squeeze_271: "f32[544]", sum_65: "f32[128]", squeeze_268: "f32[128]", sum_67: "f32[512]", squeeze_265: "f32[512]", sum_69: "f32[1024]", squeeze_262: "f32[1024]", sum_71: "f32[128]", squeeze_259: "f32[128]", sum_73: "f32[992]", squeeze_256: "f32[992]", sum_75: "f32[128]", squeeze_253: "f32[128]", sum_77: "f32[960]", squeeze_250: "f32[960]", sum_79: "f32[128]", squeeze_247: "f32[128]", sum_81: "f32[928]", squeeze_244: "f32[928]", sum_83: "f32[128]", squeeze_241: "f32[128]", sum_85: "f32[896]", squeeze_238: "f32[896]", sum_87: "f32[128]", squeeze_235: "f32[128]", sum_89: "f32[864]", squeeze_232: "f32[864]", sum_91: "f32[128]", squeeze_229: "f32[128]", sum_93: "f32[832]", squeeze_226: "f32[832]", sum_95: "f32[128]", squeeze_223: "f32[128]", sum_97: "f32[800]", squeeze_220: "f32[800]", sum_99: "f32[128]", squeeze_217: "f32[128]", sum_101: "f32[768]", squeeze_214: "f32[768]", sum_103: "f32[128]", squeeze_211: "f32[128]", sum_105: "f32[736]", squeeze_208: "f32[736]", sum_107: "f32[128]", squeeze_205: "f32[128]", sum_109: "f32[704]", squeeze_202: "f32[704]", sum_111: "f32[128]", squeeze_199: "f32[128]", sum_113: "f32[672]", squeeze_196: "f32[672]", sum_115: "f32[128]", squeeze_193: "f32[128]", sum_117: "f32[640]", squeeze_190: "f32[640]", sum_119: "f32[128]", squeeze_187: "f32[128]", sum_121: "f32[608]", squeeze_184: "f32[608]", sum_123: "f32[128]", squeeze_181: "f32[128]", sum_125: "f32[576]", squeeze_178: "f32[576]", sum_127: "f32[128]", squeeze_175: "f32[128]", sum_129: "f32[544]", squeeze_172: "f32[544]", sum_131: "f32[128]", squeeze_169: "f32[128]", sum_133: "f32[512]", squeeze_166: "f32[512]", sum_135: "f32[128]", squeeze_163: "f32[128]", sum_137: "f32[480]", squeeze_160: "f32[480]", sum_139: "f32[128]", squeeze_157: "f32[128]", sum_141: "f32[448]", squeeze_154: "f32[448]", sum_143: "f32[128]", squeeze_151: "f32[128]", sum_145: "f32[416]", squeeze_148: "f32[416]", sum_147: "f32[128]", squeeze_145: "f32[128]", sum_149: "f32[384]", squeeze_142: "f32[384]", sum_151: "f32[128]", squeeze_139: "f32[128]", sum_153: "f32[352]", squeeze_136: "f32[352]", sum_155: "f32[128]", squeeze_133: "f32[128]", sum_157: "f32[320]", squeeze_130: "f32[320]", sum_159: "f32[128]", squeeze_127: "f32[128]", sum_161: "f32[288]", squeeze_124: "f32[288]", sum_163: "f32[128]", squeeze_121: "f32[128]", sum_165: "f32[256]", squeeze_118: "f32[256]", sum_167: "f32[512]", squeeze_115: "f32[512]", sum_169: "f32[128]", squeeze_112: "f32[128]", sum_171: "f32[480]", squeeze_109: "f32[480]", sum_173: "f32[128]", squeeze_106: "f32[128]", sum_175: "f32[448]", squeeze_103: "f32[448]", sum_177: "f32[128]", squeeze_100: "f32[128]", sum_179: "f32[416]", squeeze_97: "f32[416]", sum_181: "f32[128]", squeeze_94: "f32[128]", sum_183: "f32[384]", squeeze_91: "f32[384]", sum_185: "f32[128]", squeeze_88: "f32[128]", sum_187: "f32[352]", squeeze_85: "f32[352]", sum_189: "f32[128]", squeeze_82: "f32[128]", sum_191: "f32[320]", squeeze_79: "f32[320]", sum_193: "f32[128]", squeeze_76: "f32[128]", sum_195: "f32[288]", squeeze_73: "f32[288]", sum_197: "f32[128]", squeeze_70: "f32[128]", sum_199: "f32[256]", squeeze_67: "f32[256]", sum_201: "f32[128]", squeeze_64: "f32[128]", sum_203: "f32[224]", squeeze_61: "f32[224]", sum_205: "f32[128]", squeeze_58: "f32[128]", sum_207: "f32[192]", squeeze_55: "f32[192]", sum_209: "f32[128]", squeeze_52: "f32[128]", sum_211: "f32[160]", squeeze_49: "f32[160]", sum_213: "f32[128]", squeeze_46: "f32[128]", sum_215: "f32[128]", squeeze_43: "f32[128]", sum_217: "f32[256]", squeeze_40: "f32[256]", sum_219: "f32[128]", squeeze_37: "f32[128]", sum_221: "f32[224]", squeeze_34: "f32[224]", sum_223: "f32[128]", squeeze_31: "f32[128]", sum_225: "f32[192]", squeeze_28: "f32[192]", sum_227: "f32[128]", squeeze_25: "f32[128]", sum_229: "f32[160]", squeeze_22: "f32[160]", sum_231: "f32[128]", squeeze_19: "f32[128]", sum_233: "f32[128]", squeeze_16: "f32[128]", sum_235: "f32[128]", squeeze_13: "f32[128]", sum_237: "f32[96]", squeeze_10: "f32[96]", sum_239: "f32[128]", squeeze_7: "f32[128]", sum_241: "f32[64]", squeeze_4: "f32[64]", sum_243: "f32[64]", squeeze_1: "f32[64]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:217 in forward, code: out = self.classifier(out)
        permute_default: "f32[1000, 64]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        mul_tensor: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_361);  sum_3 = squeeze_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_358);  sum_5 = squeeze_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_2: "f32[992]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_355);  sum_7 = squeeze_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_352);  sum_9 = squeeze_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_4: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_349);  sum_11 = squeeze_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_346);  sum_13 = squeeze_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_6: "f32[928]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_343);  sum_15 = squeeze_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_7: "f32[128]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_340);  sum_17 = squeeze_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_8: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_337);  sum_19 = squeeze_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_9: "f32[128]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_334);  sum_21 = squeeze_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_10: "f32[864]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_331);  sum_23 = squeeze_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_11: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_328);  sum_25 = squeeze_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_12: "f32[832]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_325);  sum_27 = squeeze_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_13: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_322);  sum_29 = squeeze_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_14: "f32[800]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_319);  sum_31 = squeeze_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_15: "f32[128]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_316);  sum_33 = squeeze_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_16: "f32[768]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_313);  sum_35 = squeeze_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_17: "f32[128]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_310);  sum_37 = squeeze_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_18: "f32[736]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_307);  sum_39 = squeeze_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_19: "f32[128]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_304);  sum_41 = squeeze_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_20: "f32[704]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_301);  sum_43 = squeeze_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_21: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_298);  sum_45 = squeeze_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_22: "f32[672]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_295);  sum_47 = squeeze_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_23: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_292);  sum_49 = squeeze_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_24: "f32[640]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_289);  sum_51 = squeeze_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_25: "f32[128]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_286);  sum_53 = squeeze_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_26: "f32[608]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_283);  sum_55 = squeeze_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_27: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_280);  sum_57 = squeeze_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_28: "f32[576]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_277);  sum_59 = squeeze_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_29: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_274);  sum_61 = squeeze_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_30: "f32[544]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_271);  sum_63 = squeeze_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_31: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_268);  sum_65 = squeeze_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_32: "f32[512]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_265);  sum_67 = squeeze_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        mul_tensor_33: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_262);  sum_69 = squeeze_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_34: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_259);  sum_71 = squeeze_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_35: "f32[992]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_256);  sum_73 = squeeze_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_36: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_253);  sum_75 = squeeze_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_37: "f32[960]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_250);  sum_77 = squeeze_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_38: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_247);  sum_79 = squeeze_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_39: "f32[928]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_244);  sum_81 = squeeze_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_40: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_241);  sum_83 = squeeze_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_41: "f32[896]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_238);  sum_85 = squeeze_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_42: "f32[128]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_235);  sum_87 = squeeze_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_43: "f32[864]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_232);  sum_89 = squeeze_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_44: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_229);  sum_91 = squeeze_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_45: "f32[832]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_226);  sum_93 = squeeze_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_46: "f32[128]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_223);  sum_95 = squeeze_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_47: "f32[800]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_220);  sum_97 = squeeze_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_48: "f32[128]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_217);  sum_99 = squeeze_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_49: "f32[768]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_214);  sum_101 = squeeze_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_50: "f32[128]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_211);  sum_103 = squeeze_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_51: "f32[736]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_208);  sum_105 = squeeze_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_52: "f32[128]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_205);  sum_107 = squeeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_53: "f32[704]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_202);  sum_109 = squeeze_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_54: "f32[128]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_199);  sum_111 = squeeze_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_55: "f32[672]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_196);  sum_113 = squeeze_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_56: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_193);  sum_115 = squeeze_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_57: "f32[640]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_190);  sum_117 = squeeze_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_58: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_187);  sum_119 = squeeze_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_59: "f32[608]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_184);  sum_121 = squeeze_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_60: "f32[128]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_181);  sum_123 = squeeze_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_61: "f32[576]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_178);  sum_125 = squeeze_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_62: "f32[128]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_175);  sum_127 = squeeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_63: "f32[544]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_172);  sum_129 = squeeze_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_64: "f32[128]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_169);  sum_131 = squeeze_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_65: "f32[512]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_166);  sum_133 = squeeze_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_66: "f32[128]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_163);  sum_135 = squeeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_67: "f32[480]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_160);  sum_137 = squeeze_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_68: "f32[128]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_157);  sum_139 = squeeze_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_69: "f32[448]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_154);  sum_141 = squeeze_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_70: "f32[128]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_151);  sum_143 = squeeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_71: "f32[416]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_148);  sum_145 = squeeze_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_72: "f32[128]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_145);  sum_147 = squeeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_73: "f32[384]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_142);  sum_149 = squeeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_74: "f32[128]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_139);  sum_151 = squeeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_75: "f32[352]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_136);  sum_153 = squeeze_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_76: "f32[128]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_133);  sum_155 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_77: "f32[320]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_130);  sum_157 = squeeze_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_78: "f32[128]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_127);  sum_159 = squeeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_79: "f32[288]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_124);  sum_161 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_80: "f32[128]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_121);  sum_163 = squeeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_81: "f32[256]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_118);  sum_165 = squeeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        mul_tensor_82: "f32[512]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_115);  sum_167 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_83: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_112);  sum_169 = squeeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_84: "f32[480]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_109);  sum_171 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_85: "f32[128]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_106);  sum_173 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_86: "f32[448]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_103);  sum_175 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_87: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_100);  sum_177 = squeeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_88: "f32[416]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_97);  sum_179 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_89: "f32[128]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_94);  sum_181 = squeeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_90: "f32[384]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_91);  sum_183 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_91: "f32[128]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_88);  sum_185 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_92: "f32[352]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_85);  sum_187 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_93: "f32[128]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_82);  sum_189 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_94: "f32[320]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_79);  sum_191 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_95: "f32[128]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_76);  sum_193 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_96: "f32[288]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_73);  sum_195 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_97: "f32[128]" = torch.ops.aten.mul.Tensor(sum_197, squeeze_70);  sum_197 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_98: "f32[256]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_67);  sum_199 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_99: "f32[128]" = torch.ops.aten.mul.Tensor(sum_201, squeeze_64);  sum_201 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_100: "f32[224]" = torch.ops.aten.mul.Tensor(sum_203, squeeze_61);  sum_203 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_101: "f32[128]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_58);  sum_205 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_102: "f32[192]" = torch.ops.aten.mul.Tensor(sum_207, squeeze_55);  sum_207 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_103: "f32[128]" = torch.ops.aten.mul.Tensor(sum_209, squeeze_52);  sum_209 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_104: "f32[160]" = torch.ops.aten.mul.Tensor(sum_211, squeeze_49);  sum_211 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_105: "f32[128]" = torch.ops.aten.mul.Tensor(sum_213, squeeze_46);  sum_213 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_106: "f32[128]" = torch.ops.aten.mul.Tensor(sum_215, squeeze_43);  sum_215 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        mul_tensor_107: "f32[256]" = torch.ops.aten.mul.Tensor(sum_217, squeeze_40);  sum_217 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_108: "f32[128]" = torch.ops.aten.mul.Tensor(sum_219, squeeze_37);  sum_219 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_109: "f32[224]" = torch.ops.aten.mul.Tensor(sum_221, squeeze_34);  sum_221 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_110: "f32[128]" = torch.ops.aten.mul.Tensor(sum_223, squeeze_31);  sum_223 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_111: "f32[192]" = torch.ops.aten.mul.Tensor(sum_225, squeeze_28);  sum_225 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_112: "f32[128]" = torch.ops.aten.mul.Tensor(sum_227, squeeze_25);  sum_227 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_113: "f32[160]" = torch.ops.aten.mul.Tensor(sum_229, squeeze_22);  sum_229 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_114: "f32[128]" = torch.ops.aten.mul.Tensor(sum_231, squeeze_19);  sum_231 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_115: "f32[128]" = torch.ops.aten.mul.Tensor(sum_233, squeeze_16);  sum_233 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_116: "f32[128]" = torch.ops.aten.mul.Tensor(sum_235, squeeze_13);  sum_235 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_117: "f32[96]" = torch.ops.aten.mul.Tensor(sum_237, squeeze_10);  sum_237 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        mul_tensor_118: "f32[128]" = torch.ops.aten.mul.Tensor(sum_239, squeeze_7);  sum_239 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        mul_tensor_119: "f32[64]" = torch.ops.aten.mul.Tensor(sum_241, squeeze_4);  sum_241 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        mul_tensor_120: "f32[64]" = torch.ops.aten.mul.Tensor(sum_243, squeeze_1);  sum_243 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, mul_tensor_31, mul_tensor_32, mul_tensor_33, mul_tensor_34, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_52, mul_tensor_53, mul_tensor_54, mul_tensor_55, mul_tensor_56, mul_tensor_57, mul_tensor_58, mul_tensor_59, mul_tensor_60, mul_tensor_61, mul_tensor_62, mul_tensor_63, mul_tensor_64, mul_tensor_65, mul_tensor_66, mul_tensor_67, mul_tensor_68, mul_tensor_69, mul_tensor_70, mul_tensor_71, mul_tensor_72, mul_tensor_73, mul_tensor_74, mul_tensor_75, mul_tensor_76, mul_tensor_77, mul_tensor_78, mul_tensor_79, mul_tensor_80, mul_tensor_81, mul_tensor_82, mul_tensor_83, mul_tensor_84, mul_tensor_85, mul_tensor_86, mul_tensor_87, mul_tensor_88, mul_tensor_89, mul_tensor_90, mul_tensor_91, mul_tensor_92, mul_tensor_93, mul_tensor_94, mul_tensor_95, mul_tensor_96, mul_tensor_97, mul_tensor_98, mul_tensor_99, mul_tensor_100, mul_tensor_101, mul_tensor_102, mul_tensor_103, mul_tensor_104, mul_tensor_105, mul_tensor_106, mul_tensor_107, mul_tensor_108, mul_tensor_109, mul_tensor_110, mul_tensor_111, mul_tensor_112, mul_tensor_113, mul_tensor_114, mul_tensor_115, mul_tensor_116, mul_tensor_117, mul_tensor_118, mul_tensor_119, mul_tensor_120)


def _default_make_inputs():
    return [
    torch.randn([64, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
