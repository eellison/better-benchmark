"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '2048'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([2048, 8192], bf16), T([2048, 8192], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([2048, 512], bf16), T([2048, 2048], bf16), T([2048, 512], bf16), T([2048, 2048], bf16), T([2048, 2048], bf16), T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([4, 512], i64, gen=Index(100)), T([], f32), T([128256, 2048], bf16))"

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_323: "f32[4, 512, 2048]", rsqrt_32: "f32[4, 512, 1]", view_330: "bf16[4, 512, 2048]", view_332: "bf16[2048, 2048]", view_334: "bf16[2048, 8192]", view_336: "bf16[2048, 8192]", convert_element_type_313: "f32[4, 512, 2048]", rsqrt_31: "f32[4, 512, 1]", add_120: "bf16[4, 512, 2048]", view_339: "bf16[2048, 2048]", getitem_135: "bf16[4, 32, 512, 64]", view_345: "bf16[2048, 512]", view_348: "bf16[2048, 512]", view_351: "bf16[2048, 2048]", convert_element_type_303: "f32[4, 512, 2048]", rsqrt_30: "f32[4, 512, 1]", add_128: "bf16[4, 512, 2048]", view_354: "bf16[2048, 2048]", view_356: "bf16[2048, 8192]", view_358: "bf16[2048, 8192]", convert_element_type_293: "f32[4, 512, 2048]", rsqrt_29: "f32[4, 512, 1]", add_133: "bf16[4, 512, 2048]", view_361: "bf16[2048, 2048]", getitem_126: "bf16[4, 32, 512, 64]", view_367: "bf16[2048, 512]", view_370: "bf16[2048, 512]", view_373: "bf16[2048, 2048]", convert_element_type_283: "f32[4, 512, 2048]", rsqrt_28: "f32[4, 512, 1]", add_141: "bf16[4, 512, 2048]", view_376: "bf16[2048, 2048]", view_378: "bf16[2048, 8192]", view_380: "bf16[2048, 8192]", convert_element_type_273: "f32[4, 512, 2048]", rsqrt_27: "f32[4, 512, 1]", add_146: "bf16[4, 512, 2048]", view_383: "bf16[2048, 2048]", getitem_117: "bf16[4, 32, 512, 64]", view_389: "bf16[2048, 512]", view_392: "bf16[2048, 512]", view_395: "bf16[2048, 2048]", convert_element_type_263: "f32[4, 512, 2048]", rsqrt_26: "f32[4, 512, 1]", add_154: "bf16[4, 512, 2048]", view_398: "bf16[2048, 2048]", view_400: "bf16[2048, 8192]", view_402: "bf16[2048, 8192]", convert_element_type_253: "f32[4, 512, 2048]", rsqrt_25: "f32[4, 512, 1]", add_159: "bf16[4, 512, 2048]", view_405: "bf16[2048, 2048]", getitem_108: "bf16[4, 32, 512, 64]", view_411: "bf16[2048, 512]", view_414: "bf16[2048, 512]", view_417: "bf16[2048, 2048]", convert_element_type_243: "f32[4, 512, 2048]", rsqrt_24: "f32[4, 512, 1]", add_167: "bf16[4, 512, 2048]", view_420: "bf16[2048, 2048]", view_422: "bf16[2048, 8192]", view_424: "bf16[2048, 8192]", convert_element_type_233: "f32[4, 512, 2048]", rsqrt_23: "f32[4, 512, 1]", add_172: "bf16[4, 512, 2048]", view_427: "bf16[2048, 2048]", getitem_99: "bf16[4, 32, 512, 64]", view_433: "bf16[2048, 512]", view_436: "bf16[2048, 512]", view_439: "bf16[2048, 2048]", convert_element_type_223: "f32[4, 512, 2048]", rsqrt_22: "f32[4, 512, 1]", add_180: "bf16[4, 512, 2048]", view_442: "bf16[2048, 2048]", view_444: "bf16[2048, 8192]", view_446: "bf16[2048, 8192]", convert_element_type_213: "f32[4, 512, 2048]", rsqrt_21: "f32[4, 512, 1]", add_185: "bf16[4, 512, 2048]", view_449: "bf16[2048, 2048]", getitem_90: "bf16[4, 32, 512, 64]", view_455: "bf16[2048, 512]", view_458: "bf16[2048, 512]", view_461: "bf16[2048, 2048]", convert_element_type_203: "f32[4, 512, 2048]", rsqrt_20: "f32[4, 512, 1]", add_193: "bf16[4, 512, 2048]", view_464: "bf16[2048, 2048]", view_466: "bf16[2048, 8192]", view_468: "bf16[2048, 8192]", convert_element_type_193: "f32[4, 512, 2048]", rsqrt_19: "f32[4, 512, 1]", add_198: "bf16[4, 512, 2048]", view_471: "bf16[2048, 2048]", getitem_81: "bf16[4, 32, 512, 64]", view_477: "bf16[2048, 512]", view_480: "bf16[2048, 512]", view_483: "bf16[2048, 2048]", convert_element_type_183: "f32[4, 512, 2048]", rsqrt_18: "f32[4, 512, 1]", add_206: "bf16[4, 512, 2048]", view_486: "bf16[2048, 2048]", view_488: "bf16[2048, 8192]", view_490: "bf16[2048, 8192]", convert_element_type_173: "f32[4, 512, 2048]", rsqrt_17: "f32[4, 512, 1]", add_211: "bf16[4, 512, 2048]", view_493: "bf16[2048, 2048]", getitem_72: "bf16[4, 32, 512, 64]", view_499: "bf16[2048, 512]", view_502: "bf16[2048, 512]", view_505: "bf16[2048, 2048]", convert_element_type_163: "f32[4, 512, 2048]", rsqrt_16: "f32[4, 512, 1]", add_219: "bf16[4, 512, 2048]", view_508: "bf16[2048, 2048]", view_510: "bf16[2048, 8192]", view_512: "bf16[2048, 8192]", convert_element_type_153: "f32[4, 512, 2048]", rsqrt_15: "f32[4, 512, 1]", add_224: "bf16[4, 512, 2048]", view_515: "bf16[2048, 2048]", getitem_63: "bf16[4, 32, 512, 64]", view_521: "bf16[2048, 512]", view_524: "bf16[2048, 512]", view_527: "bf16[2048, 2048]", convert_element_type_143: "f32[4, 512, 2048]", rsqrt_14: "f32[4, 512, 1]", add_232: "bf16[4, 512, 2048]", view_530: "bf16[2048, 2048]", view_532: "bf16[2048, 8192]", view_534: "bf16[2048, 8192]", convert_element_type_133: "f32[4, 512, 2048]", rsqrt_13: "f32[4, 512, 1]", add_237: "bf16[4, 512, 2048]", view_537: "bf16[2048, 2048]", getitem_54: "bf16[4, 32, 512, 64]", view_543: "bf16[2048, 512]", view_546: "bf16[2048, 512]", view_549: "bf16[2048, 2048]", convert_element_type_123: "f32[4, 512, 2048]", rsqrt_12: "f32[4, 512, 1]", add_245: "bf16[4, 512, 2048]", view_552: "bf16[2048, 2048]", view_554: "bf16[2048, 8192]", view_556: "bf16[2048, 8192]", convert_element_type_113: "f32[4, 512, 2048]", rsqrt_11: "f32[4, 512, 1]", add_250: "bf16[4, 512, 2048]", view_559: "bf16[2048, 2048]", getitem_45: "bf16[4, 32, 512, 64]", view_565: "bf16[2048, 512]", view_568: "bf16[2048, 512]", view_571: "bf16[2048, 2048]", convert_element_type_103: "f32[4, 512, 2048]", rsqrt_10: "f32[4, 512, 1]", add_258: "bf16[4, 512, 2048]", view_574: "bf16[2048, 2048]", view_576: "bf16[2048, 8192]", view_578: "bf16[2048, 8192]", convert_element_type_93: "f32[4, 512, 2048]", rsqrt_9: "f32[4, 512, 1]", add_263: "bf16[4, 512, 2048]", view_581: "bf16[2048, 2048]", getitem_36: "bf16[4, 32, 512, 64]", view_587: "bf16[2048, 512]", view_590: "bf16[2048, 512]", view_593: "bf16[2048, 2048]", convert_element_type_83: "f32[4, 512, 2048]", rsqrt_8: "f32[4, 512, 1]", add_271: "bf16[4, 512, 2048]", view_596: "bf16[2048, 2048]", view_598: "bf16[2048, 8192]", view_600: "bf16[2048, 8192]", convert_element_type_73: "f32[4, 512, 2048]", rsqrt_7: "f32[4, 512, 1]", add_276: "bf16[4, 512, 2048]", view_603: "bf16[2048, 2048]", getitem_27: "bf16[4, 32, 512, 64]", view_609: "bf16[2048, 512]", view_612: "bf16[2048, 512]", view_615: "bf16[2048, 2048]", convert_element_type_63: "f32[4, 512, 2048]", rsqrt_6: "f32[4, 512, 1]", add_284: "bf16[4, 512, 2048]", view_618: "bf16[2048, 2048]", view_620: "bf16[2048, 8192]", view_622: "bf16[2048, 8192]", convert_element_type_53: "f32[4, 512, 2048]", rsqrt_5: "f32[4, 512, 1]", add_289: "bf16[4, 512, 2048]", view_625: "bf16[2048, 2048]", getitem_18: "bf16[4, 32, 512, 64]", view_631: "bf16[2048, 512]", view_634: "bf16[2048, 512]", view_637: "bf16[2048, 2048]", convert_element_type_43: "f32[4, 512, 2048]", rsqrt_4: "f32[4, 512, 1]", add_297: "bf16[4, 512, 2048]", view_640: "bf16[2048, 2048]", view_642: "bf16[2048, 8192]", view_644: "bf16[2048, 8192]", convert_element_type_33: "f32[4, 512, 2048]", rsqrt_3: "f32[4, 512, 1]", add_302: "bf16[4, 512, 2048]", view_647: "bf16[2048, 2048]", getitem_9: "bf16[4, 32, 512, 64]", view_653: "bf16[2048, 512]", view_656: "bf16[2048, 512]", view_659: "bf16[2048, 2048]", convert_element_type_23: "f32[4, 512, 2048]", rsqrt_2: "f32[4, 512, 1]", add_310: "bf16[4, 512, 2048]", view_662: "bf16[2048, 2048]", view_664: "bf16[2048, 8192]", view_666: "bf16[2048, 8192]", convert_element_type_13: "f32[4, 512, 2048]", rsqrt_1: "f32[4, 512, 1]", add_315: "bf16[4, 512, 2048]", view_669: "bf16[2048, 2048]", getitem: "bf16[4, 32, 512, 64]", view_675: "bf16[2048, 512]", mm_334: "bf16[2048, 2048]", view_678: "bf16[2048, 512]", mm_336: "bf16[2048, 2048]", view_681: "bf16[2048, 2048]", mm_338: "bf16[2048, 2048]", primals_4: "bf16[2048]", embedding: "bf16[4, 512, 2048]", rsqrt: "f32[4, 512, 1]", add_317: "bf16[4, 512, 2048]", primals_1: "i64[4, 512]", full_default_34: "f32[]", mm_113: "bf16[128256, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_323, rsqrt_32);  convert_element_type_323 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_330, convert_element_type_default);  view_330 = convert_element_type_default = None
        sum_dim_int_list: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list, [2048]);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_332, [1, 0]);  view_332 = None
        permute_default_1: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_334, [1, 0]);  view_334 = None
        permute_default_2: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_336, [1, 0]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_313, rsqrt_31);  convert_element_type_313 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_120, convert_element_type_default_1);  add_120 = convert_element_type_default_1 = None
        sum_dim_int_list_1: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [2048]);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_3: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_339, [1, 0]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_4: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_4, [4, 512, -1]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_3: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_2, [2048, 2048]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_345, [1, 0]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_348, [1, 0]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_351, [1, 0]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_303, rsqrt_30);  convert_element_type_303 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_128, convert_element_type_default_2);  add_128 = convert_element_type_default_2 = None
        sum_dim_int_list_2: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_4: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [2048]);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_8: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_354, [1, 0]);  view_354 = None
        permute_default_9: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_356, [1, 0]);  view_356 = None
        permute_default_10: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_358, [1, 0]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_293, rsqrt_29);  convert_element_type_293 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_3: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        mul_tensor_7: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_133, convert_element_type_default_3);  add_133 = convert_element_type_default_3 = None
        sum_dim_int_list_3: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_5: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [2048]);  sum_dim_int_list_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_11: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_361, [1, 0]);  view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_12: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_6: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_12, [4, 512, -1]);  permute_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_7: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_6, [2048, 2048]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_13: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_367, [1, 0]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_14: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_370, [1, 0]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_15: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_373, [1, 0]);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_283, rsqrt_28);  convert_element_type_283 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_4: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        mul_tensor_9: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_141, convert_element_type_default_4);  add_141 = convert_element_type_default_4 = None
        sum_dim_int_list_4: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_8: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [2048]);  sum_dim_int_list_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_16: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_376, [1, 0]);  view_376 = None
        permute_default_17: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_378, [1, 0]);  view_378 = None
        permute_default_18: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_380, [1, 0]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_273, rsqrt_27);  convert_element_type_273 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_5: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        mul_tensor_11: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_146, convert_element_type_default_5);  add_146 = convert_element_type_default_5 = None
        sum_dim_int_list_5: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_9: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [2048]);  sum_dim_int_list_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_19: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_383, [1, 0]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_20: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_10: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_20, [4, 512, -1]);  permute_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_11: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_10, [2048, 2048]);  reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_21: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_389, [1, 0]);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_22: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_392, [1, 0]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_23: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_395, [1, 0]);  view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_263, rsqrt_26);  convert_element_type_263 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_6: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_154, convert_element_type_default_6);  add_154 = convert_element_type_default_6 = None
        sum_dim_int_list_6: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_12: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [2048]);  sum_dim_int_list_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_24: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_398, [1, 0]);  view_398 = None
        permute_default_25: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_400, [1, 0]);  view_400 = None
        permute_default_26: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_402, [1, 0]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_253, rsqrt_25);  convert_element_type_253 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_7: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.bfloat16);  mul_tensor_14 = None
        mul_tensor_15: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_159, convert_element_type_default_7);  add_159 = convert_element_type_default_7 = None
        sum_dim_int_list_7: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_13: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [2048]);  sum_dim_int_list_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_27: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_405, [1, 0]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_28: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_14: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_28, [4, 512, -1]);  permute_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_15: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_14, [2048, 2048]);  reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_29: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_411, [1, 0]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_30: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_414, [1, 0]);  view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_31: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_417, [1, 0]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_243, rsqrt_24);  convert_element_type_243 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_8: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        mul_tensor_17: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_167, convert_element_type_default_8);  add_167 = convert_element_type_default_8 = None
        sum_dim_int_list_8: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_16: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [2048]);  sum_dim_int_list_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_32: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_420, [1, 0]);  view_420 = None
        permute_default_33: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_422, [1, 0]);  view_422 = None
        permute_default_34: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_424, [1, 0]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_23);  convert_element_type_233 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_9: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.bfloat16);  mul_tensor_18 = None
        mul_tensor_19: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_172, convert_element_type_default_9);  add_172 = convert_element_type_default_9 = None
        sum_dim_int_list_9: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_17: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [2048]);  sum_dim_int_list_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_35: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_427, [1, 0]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_36: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_18: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_36, [4, 512, -1]);  permute_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_19: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_18, [2048, 2048]);  reshape_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_37: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_433, [1, 0]);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_38: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_436, [1, 0]);  view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_39: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_439, [1, 0]);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_223, rsqrt_22);  convert_element_type_223 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_10: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.bfloat16);  mul_tensor_20 = None
        mul_tensor_21: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_180, convert_element_type_default_10);  add_180 = convert_element_type_default_10 = None
        sum_dim_int_list_10: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_20: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [2048]);  sum_dim_int_list_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_40: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_442, [1, 0]);  view_442 = None
        permute_default_41: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_444, [1, 0]);  view_444 = None
        permute_default_42: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_446, [1, 0]);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_213, rsqrt_21);  convert_element_type_213 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_11: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_22, torch.bfloat16);  mul_tensor_22 = None
        mul_tensor_23: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_185, convert_element_type_default_11);  add_185 = convert_element_type_default_11 = None
        sum_dim_int_list_11: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_21: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [2048]);  sum_dim_int_list_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_43: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_449, [1, 0]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_44: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_22: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_44, [4, 512, -1]);  permute_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_23: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_22, [2048, 2048]);  reshape_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_45: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_455, [1, 0]);  view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_46: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_458, [1, 0]);  view_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_47: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_461, [1, 0]);  view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_203, rsqrt_20);  convert_element_type_203 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_12: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.bfloat16);  mul_tensor_24 = None
        mul_tensor_25: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_193, convert_element_type_default_12);  add_193 = convert_element_type_default_12 = None
        sum_dim_int_list_12: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_24: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [2048]);  sum_dim_int_list_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_48: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_464, [1, 0]);  view_464 = None
        permute_default_49: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_466, [1, 0]);  view_466 = None
        permute_default_50: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_468, [1, 0]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_193, rsqrt_19);  convert_element_type_193 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_13: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_26, torch.bfloat16);  mul_tensor_26 = None
        mul_tensor_27: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_198, convert_element_type_default_13);  add_198 = convert_element_type_default_13 = None
        sum_dim_int_list_13: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_25: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, [2048]);  sum_dim_int_list_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_51: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_471, [1, 0]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_52: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_26: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_52, [4, 512, -1]);  permute_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_27: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_26, [2048, 2048]);  reshape_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_53: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_477, [1, 0]);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_54: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_480, [1, 0]);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_55: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_483, [1, 0]);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_183, rsqrt_18);  convert_element_type_183 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_14: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.bfloat16);  mul_tensor_28 = None
        mul_tensor_29: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_206, convert_element_type_default_14);  add_206 = convert_element_type_default_14 = None
        sum_dim_int_list_14: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_28: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, [2048]);  sum_dim_int_list_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_56: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_486, [1, 0]);  view_486 = None
        permute_default_57: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_488, [1, 0]);  view_488 = None
        permute_default_58: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_490, [1, 0]);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_173, rsqrt_17);  convert_element_type_173 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_15: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_30, torch.bfloat16);  mul_tensor_30 = None
        mul_tensor_31: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_211, convert_element_type_default_15);  add_211 = convert_element_type_default_15 = None
        sum_dim_int_list_15: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_29: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, [2048]);  sum_dim_int_list_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_59: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_493, [1, 0]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_60: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_30: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_60, [4, 512, -1]);  permute_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_31: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_30, [2048, 2048]);  reshape_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_61: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_499, [1, 0]);  view_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_62: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_502, [1, 0]);  view_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_63: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_505, [1, 0]);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_32: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_16);  convert_element_type_163 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_16: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.bfloat16);  mul_tensor_32 = None
        mul_tensor_33: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_219, convert_element_type_default_16);  add_219 = convert_element_type_default_16 = None
        sum_dim_int_list_16: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_32: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [2048]);  sum_dim_int_list_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_64: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_508, [1, 0]);  view_508 = None
        permute_default_65: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_510, [1, 0]);  view_510 = None
        permute_default_66: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_512, [1, 0]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_34: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_15);  convert_element_type_153 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_17: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_34, torch.bfloat16);  mul_tensor_34 = None
        mul_tensor_35: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_224, convert_element_type_default_17);  add_224 = convert_element_type_default_17 = None
        sum_dim_int_list_17: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_33: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, [2048]);  sum_dim_int_list_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_67: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_515, [1, 0]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_68: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_34: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_68, [4, 512, -1]);  permute_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_35: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_34, [2048, 2048]);  reshape_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_69: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_521, [1, 0]);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_70: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_524, [1, 0]);  view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_71: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_527, [1, 0]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_36: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_143, rsqrt_14);  convert_element_type_143 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_18: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_36, torch.bfloat16);  mul_tensor_36 = None
        mul_tensor_37: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_232, convert_element_type_default_18);  add_232 = convert_element_type_default_18 = None
        sum_dim_int_list_18: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1], True);  mul_tensor_37 = None
        reshape_default_36: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, [2048]);  sum_dim_int_list_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_72: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_530, [1, 0]);  view_530 = None
        permute_default_73: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_532, [1, 0]);  view_532 = None
        permute_default_74: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_534, [1, 0]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_38: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_13);  convert_element_type_133 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_19: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_38, torch.bfloat16);  mul_tensor_38 = None
        mul_tensor_39: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_237, convert_element_type_default_19);  add_237 = convert_element_type_default_19 = None
        sum_dim_int_list_19: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1], True);  mul_tensor_39 = None
        reshape_default_37: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, [2048]);  sum_dim_int_list_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_75: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_537, [1, 0]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_76: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_38: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_76, [4, 512, -1]);  permute_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_39: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_38, [2048, 2048]);  reshape_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_77: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_543, [1, 0]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_78: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_546, [1, 0]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_79: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_549, [1, 0]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_40: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_12);  convert_element_type_123 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_20: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_40, torch.bfloat16);  mul_tensor_40 = None
        mul_tensor_41: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_245, convert_element_type_default_20);  add_245 = convert_element_type_default_20 = None
        sum_dim_int_list_20: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_40: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, [2048]);  sum_dim_int_list_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_80: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_552, [1, 0]);  view_552 = None
        permute_default_81: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_554, [1, 0]);  view_554 = None
        permute_default_82: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_556, [1, 0]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_42: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_11);  convert_element_type_113 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_21: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_42, torch.bfloat16);  mul_tensor_42 = None
        mul_tensor_43: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_250, convert_element_type_default_21);  add_250 = convert_element_type_default_21 = None
        sum_dim_int_list_21: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_41: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, [2048]);  sum_dim_int_list_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_83: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_559, [1, 0]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_84: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_42: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_84, [4, 512, -1]);  permute_default_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_43: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_42, [2048, 2048]);  reshape_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_85: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_565, [1, 0]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_86: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_568, [1, 0]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_87: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_571, [1, 0]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_44: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_10);  convert_element_type_103 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_22: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_44, torch.bfloat16);  mul_tensor_44 = None
        mul_tensor_45: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_258, convert_element_type_default_22);  add_258 = convert_element_type_default_22 = None
        sum_dim_int_list_22: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1], True);  mul_tensor_45 = None
        reshape_default_44: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, [2048]);  sum_dim_int_list_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_88: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_574, [1, 0]);  view_574 = None
        permute_default_89: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_576, [1, 0]);  view_576 = None
        permute_default_90: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_578, [1, 0]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_46: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_9);  convert_element_type_93 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_23: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_46, torch.bfloat16);  mul_tensor_46 = None
        mul_tensor_47: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_263, convert_element_type_default_23);  add_263 = convert_element_type_default_23 = None
        sum_dim_int_list_23: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1], True);  mul_tensor_47 = None
        reshape_default_45: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [2048]);  sum_dim_int_list_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_91: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_581, [1, 0]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_92: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_46: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_92, [4, 512, -1]);  permute_default_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_47: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_46, [2048, 2048]);  reshape_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_93: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_587, [1, 0]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_94: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_590, [1, 0]);  view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_95: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_593, [1, 0]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_48: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_8);  convert_element_type_83 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_24: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_48, torch.bfloat16);  mul_tensor_48 = None
        mul_tensor_49: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_271, convert_element_type_default_24);  add_271 = convert_element_type_default_24 = None
        sum_dim_int_list_24: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_49, [0, 1], True);  mul_tensor_49 = None
        reshape_default_48: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, [2048]);  sum_dim_int_list_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_96: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_596, [1, 0]);  view_596 = None
        permute_default_97: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_598, [1, 0]);  view_598 = None
        permute_default_98: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_600, [1, 0]);  view_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_50: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_7);  convert_element_type_73 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_25: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_50, torch.bfloat16);  mul_tensor_50 = None
        mul_tensor_51: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_276, convert_element_type_default_25);  add_276 = convert_element_type_default_25 = None
        sum_dim_int_list_25: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [0, 1], True);  mul_tensor_51 = None
        reshape_default_49: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, [2048]);  sum_dim_int_list_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_99: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_603, [1, 0]);  view_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_100: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_50: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_100, [4, 512, -1]);  permute_default_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_51: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_50, [2048, 2048]);  reshape_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_101: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_609, [1, 0]);  view_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_102: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_612, [1, 0]);  view_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_103: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_615, [1, 0]);  view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_52: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_6);  convert_element_type_63 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_26: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_52, torch.bfloat16);  mul_tensor_52 = None
        mul_tensor_53: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_284, convert_element_type_default_26);  add_284 = convert_element_type_default_26 = None
        sum_dim_int_list_26: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [0, 1], True);  mul_tensor_53 = None
        reshape_default_52: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, [2048]);  sum_dim_int_list_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_104: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_618, [1, 0]);  view_618 = None
        permute_default_105: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_620, [1, 0]);  view_620 = None
        permute_default_106: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_622, [1, 0]);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_54: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_5);  convert_element_type_53 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_27: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_54, torch.bfloat16);  mul_tensor_54 = None
        mul_tensor_55: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_289, convert_element_type_default_27);  add_289 = convert_element_type_default_27 = None
        sum_dim_int_list_27: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_55, [0, 1], True);  mul_tensor_55 = None
        reshape_default_53: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, [2048]);  sum_dim_int_list_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_107: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_625, [1, 0]);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_108: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_54: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_108, [4, 512, -1]);  permute_default_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_55: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_54, [2048, 2048]);  reshape_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_109: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_631, [1, 0]);  view_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_110: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_634, [1, 0]);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_111: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_637, [1, 0]);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_56: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_43, rsqrt_4);  convert_element_type_43 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_28: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_56, torch.bfloat16);  mul_tensor_56 = None
        mul_tensor_57: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_297, convert_element_type_default_28);  add_297 = convert_element_type_default_28 = None
        sum_dim_int_list_28: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_57, [0, 1], True);  mul_tensor_57 = None
        reshape_default_56: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, [2048]);  sum_dim_int_list_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_112: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_640, [1, 0]);  view_640 = None
        permute_default_113: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_642, [1, 0]);  view_642 = None
        permute_default_114: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_644, [1, 0]);  view_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_58: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_33, rsqrt_3);  convert_element_type_33 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_29: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_58, torch.bfloat16);  mul_tensor_58 = None
        mul_tensor_59: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_302, convert_element_type_default_29);  add_302 = convert_element_type_default_29 = None
        sum_dim_int_list_29: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_59, [0, 1], True);  mul_tensor_59 = None
        reshape_default_57: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, [2048]);  sum_dim_int_list_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_115: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_647, [1, 0]);  view_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_116: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_58: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_116, [4, 512, -1]);  permute_default_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_59: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_58, [2048, 2048]);  reshape_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_117: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_653, [1, 0]);  view_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_118: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_656, [1, 0]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_119: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_659, [1, 0]);  view_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_60: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_23, rsqrt_2);  convert_element_type_23 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_30: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_60, torch.bfloat16);  mul_tensor_60 = None
        mul_tensor_61: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_310, convert_element_type_default_30);  add_310 = convert_element_type_default_30 = None
        sum_dim_int_list_30: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_61, [0, 1], True);  mul_tensor_61 = None
        reshape_default_60: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, [2048]);  sum_dim_int_list_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_120: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_662, [1, 0]);  view_662 = None
        permute_default_121: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_664, [1, 0]);  view_664 = None
        permute_default_122: "bf16[8192, 2048]" = torch.ops.aten.permute.default(view_666, [1, 0]);  view_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_62: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_13, rsqrt_1);  convert_element_type_13 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_31: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_62, torch.bfloat16);  mul_tensor_62 = None
        mul_tensor_63: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_315, convert_element_type_default_31);  add_315 = convert_element_type_default_31 = None
        sum_dim_int_list_31: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_63, [0, 1], True);  mul_tensor_63 = None
        reshape_default_61: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, [2048]);  sum_dim_int_list_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_123: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_669, [1, 0]);  view_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_124: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_62: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_124, [4, 512, -1]);  permute_default_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_63: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_62, [2048, 2048]);  reshape_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_125: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_675, [1, 0]);  view_675 = None
        reshape_default_64: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_334, [4, 512, 2048]);  mm_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_126: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_678, [1, 0]);  view_678 = None
        reshape_default_65: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_336, [4, 512, 2048]);  mm_336 = None
        add_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(reshape_default_64, reshape_default_65);  reshape_default_64 = reshape_default_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_127: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_681, [1, 0]);  view_681 = None
        reshape_default_66: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_338, [4, 512, 2048]);  mm_338 = None
        add_tensor_1: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_66);  add_tensor = reshape_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_64: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_32: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_65: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_32, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_33: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_65, torch.bfloat16);  mul_tensor_65 = None
        mul_tensor_66: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, convert_element_type_default_33);  add_tensor_1 = convert_element_type_default_33 = None
        sum_dim_int_list_32: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_66, [0, 1], True);  mul_tensor_66 = None
        reshape_default_67: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, [2048]);  sum_dim_int_list_32 = None
        convert_element_type_default_34: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_64, torch.float32);  mul_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_67: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_34, convert_element_type_default_32)
        mul_tensor_68: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_34, rsqrt);  convert_element_type_default_34 = None
        sum_dim_int_list_33: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_67, [2], True);  mul_tensor_67 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_33, -0.5);  sum_dim_int_list_33 = None
        mul_tensor_69: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_69, [4, 512, 2048]);  mul_tensor_69 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_32, 1.0);  convert_element_type_default_32 = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_70: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_70);  mul_tensor_68 = mul_tensor_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_35: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_317, convert_element_type_default_35);  add_317 = convert_element_type_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:389 in forward, code: inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        convert_element_type_default_36: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[4, 512, 2048]" = torch.ops.aten.where.self(unsqueeze_default, full_default_34, convert_element_type_default_36);  unsqueeze_default = full_default_34 = convert_element_type_default_36 = None
        full_default: "f32[128256, 2048]" = torch.ops.aten.full.default([128256, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[128256, 2048]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        convert_element_type_default_37: "bf16[128256, 2048]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.bfloat16);  index_put_default = None
        add_tensor_4: "bf16[128256, 2048]" = torch.ops.aten.add.Tensor(mm_113, convert_element_type_default_37);  mm_113 = convert_element_type_default_37 = None
        return (reshape_default, permute_default, permute_default_1, permute_default_2, reshape_default_1, permute_default_3, reshape_default_3, permute_default_5, permute_default_6, permute_default_7, reshape_default_4, permute_default_8, permute_default_9, permute_default_10, reshape_default_5, permute_default_11, reshape_default_7, permute_default_13, permute_default_14, permute_default_15, reshape_default_8, permute_default_16, permute_default_17, permute_default_18, reshape_default_9, permute_default_19, reshape_default_11, permute_default_21, permute_default_22, permute_default_23, reshape_default_12, permute_default_24, permute_default_25, permute_default_26, reshape_default_13, permute_default_27, reshape_default_15, permute_default_29, permute_default_30, permute_default_31, reshape_default_16, permute_default_32, permute_default_33, permute_default_34, reshape_default_17, permute_default_35, reshape_default_19, permute_default_37, permute_default_38, permute_default_39, reshape_default_20, permute_default_40, permute_default_41, permute_default_42, reshape_default_21, permute_default_43, reshape_default_23, permute_default_45, permute_default_46, permute_default_47, reshape_default_24, permute_default_48, permute_default_49, permute_default_50, reshape_default_25, permute_default_51, reshape_default_27, permute_default_53, permute_default_54, permute_default_55, reshape_default_28, permute_default_56, permute_default_57, permute_default_58, reshape_default_29, permute_default_59, reshape_default_31, permute_default_61, permute_default_62, permute_default_63, reshape_default_32, permute_default_64, permute_default_65, permute_default_66, reshape_default_33, permute_default_67, reshape_default_35, permute_default_69, permute_default_70, permute_default_71, reshape_default_36, permute_default_72, permute_default_73, permute_default_74, reshape_default_37, permute_default_75, reshape_default_39, permute_default_77, permute_default_78, permute_default_79, reshape_default_40, permute_default_80, permute_default_81, permute_default_82, reshape_default_41, permute_default_83, reshape_default_43, permute_default_85, permute_default_86, permute_default_87, reshape_default_44, permute_default_88, permute_default_89, permute_default_90, reshape_default_45, permute_default_91, reshape_default_47, permute_default_93, permute_default_94, permute_default_95, reshape_default_48, permute_default_96, permute_default_97, permute_default_98, reshape_default_49, permute_default_99, reshape_default_51, permute_default_101, permute_default_102, permute_default_103, reshape_default_52, permute_default_104, permute_default_105, permute_default_106, reshape_default_53, permute_default_107, reshape_default_55, permute_default_109, permute_default_110, permute_default_111, reshape_default_56, permute_default_112, permute_default_113, permute_default_114, reshape_default_57, permute_default_115, reshape_default_59, permute_default_117, permute_default_118, permute_default_119, reshape_default_60, permute_default_120, permute_default_121, permute_default_122, reshape_default_61, permute_default_123, reshape_default_63, permute_default_125, permute_default_126, permute_default_127, reshape_default_67, add_tensor_4)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
