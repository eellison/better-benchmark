"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: b56260d6c9ba
Shape hash: 44680a37
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 32000], f32), T([16, 512], i64, gen=Index(512)), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 16384, 1024], f32), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([1, 8192, 1024], f32), T([1, 1024, 1024], f32), T([256, 512, 512], f32), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 1024], f32, stride=(64, 1, 16384)), T([256, 512, 64], f32, stride=(64, 16384, 1)), T([256, 64, 512], f32, stride=(64, 1, 16384)), T([1, 8192, 1024], f32), S([16, 512, 32000]), S([-1, 32000]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_48: "f32[8192, 32000]", primals_364: "i64[16, 512]", rsqrt_47: "f32[512, 16, 1]", rsqrt_46: "f32[512, 16, 1]", view_904: "f32[1, 8192, 1024]", view_905: "f32[1, 1024, 1024]", view_900: "f32[256, 512, 512]", view_901: "f32[256, 512, 64]", view_894: "f32[256, 512, 64]", view_895: "f32[256, 64, 1024]", view_890: "f32[256, 512, 64]", view_891: "f32[256, 64, 512]", view_12: "f32[1, 16384, 1024]", view_874: "f32[1, 8192, 1024]", rsqrt_45: "f32[512, 16, 1]", rsqrt_44: "f32[512, 16, 1]", view_866: "f32[1, 8192, 1024]", view_867: "f32[1, 1024, 1024]", view_862: "f32[256, 512, 512]", view_863: "f32[256, 512, 64]", view_856: "f32[256, 512, 64]", view_857: "f32[256, 64, 1024]", view_852: "f32[256, 512, 64]", view_853: "f32[256, 64, 512]", view_836: "f32[1, 8192, 1024]", rsqrt_43: "f32[512, 16, 1]", rsqrt_42: "f32[512, 16, 1]", view_828: "f32[1, 8192, 1024]", view_829: "f32[1, 1024, 1024]", view_824: "f32[256, 512, 512]", view_825: "f32[256, 512, 64]", view_818: "f32[256, 512, 64]", view_819: "f32[256, 64, 1024]", view_814: "f32[256, 512, 64]", view_815: "f32[256, 64, 512]", view_798: "f32[1, 8192, 1024]", rsqrt_41: "f32[512, 16, 1]", rsqrt_40: "f32[512, 16, 1]", view_790: "f32[1, 8192, 1024]", view_791: "f32[1, 1024, 1024]", view_786: "f32[256, 512, 512]", view_787: "f32[256, 512, 64]", view_780: "f32[256, 512, 64]", view_781: "f32[256, 64, 1024]", view_776: "f32[256, 512, 64]", view_777: "f32[256, 64, 512]", view_760: "f32[1, 8192, 1024]", rsqrt_39: "f32[512, 16, 1]", rsqrt_38: "f32[512, 16, 1]", view_752: "f32[1, 8192, 1024]", view_753: "f32[1, 1024, 1024]", view_748: "f32[256, 512, 512]", view_749: "f32[256, 512, 64]", view_742: "f32[256, 512, 64]", view_743: "f32[256, 64, 1024]", view_738: "f32[256, 512, 64]", view_739: "f32[256, 64, 512]", view_722: "f32[1, 8192, 1024]", rsqrt_37: "f32[512, 16, 1]", rsqrt_36: "f32[512, 16, 1]", view_714: "f32[1, 8192, 1024]", view_715: "f32[1, 1024, 1024]", view_710: "f32[256, 512, 512]", view_711: "f32[256, 512, 64]", view_704: "f32[256, 512, 64]", view_705: "f32[256, 64, 1024]", view_700: "f32[256, 512, 64]", view_701: "f32[256, 64, 512]", view_684: "f32[1, 8192, 1024]", rsqrt_35: "f32[512, 16, 1]", rsqrt_34: "f32[512, 16, 1]", view_676: "f32[1, 8192, 1024]", view_677: "f32[1, 1024, 1024]", view_672: "f32[256, 512, 512]", view_673: "f32[256, 512, 64]", view_666: "f32[256, 512, 64]", view_667: "f32[256, 64, 1024]", view_662: "f32[256, 512, 64]", view_663: "f32[256, 64, 512]", view_646: "f32[1, 8192, 1024]", rsqrt_33: "f32[512, 16, 1]", rsqrt_32: "f32[512, 16, 1]", view_638: "f32[1, 8192, 1024]", view_639: "f32[1, 1024, 1024]", view_634: "f32[256, 512, 512]", view_635: "f32[256, 512, 64]", view_628: "f32[256, 512, 64]", view_629: "f32[256, 64, 1024]", view_624: "f32[256, 512, 64]", view_625: "f32[256, 64, 512]", view_608: "f32[1, 8192, 1024]", rsqrt_31: "f32[512, 16, 1]", rsqrt_30: "f32[512, 16, 1]", view_600: "f32[1, 8192, 1024]", view_601: "f32[1, 1024, 1024]", view_596: "f32[256, 512, 512]", view_597: "f32[256, 512, 64]", view_590: "f32[256, 512, 64]", view_591: "f32[256, 64, 1024]", view_586: "f32[256, 512, 64]", view_587: "f32[256, 64, 512]", view_570: "f32[1, 8192, 1024]", rsqrt_29: "f32[512, 16, 1]", rsqrt_28: "f32[512, 16, 1]", view_562: "f32[1, 8192, 1024]", view_563: "f32[1, 1024, 1024]", view_558: "f32[256, 512, 512]", view_559: "f32[256, 512, 64]", view_552: "f32[256, 512, 64]", view_553: "f32[256, 64, 1024]", view_548: "f32[256, 512, 64]", view_549: "f32[256, 64, 512]", view_532: "f32[1, 8192, 1024]", rsqrt_27: "f32[512, 16, 1]", rsqrt_26: "f32[512, 16, 1]", view_524: "f32[1, 8192, 1024]", view_525: "f32[1, 1024, 1024]", view_520: "f32[256, 512, 512]", view_521: "f32[256, 512, 64]", view_514: "f32[256, 512, 64]", view_515: "f32[256, 64, 1024]", view_510: "f32[256, 512, 64]", view_511: "f32[256, 64, 512]", view_494: "f32[1, 8192, 1024]", rsqrt_25: "f32[512, 16, 1]", rsqrt_24: "f32[512, 16, 1]", view_486: "f32[1, 8192, 1024]", view_487: "f32[1, 1024, 1024]", view_482: "f32[256, 512, 512]", view_483: "f32[256, 512, 64]", view_476: "f32[256, 512, 64]", view_477: "f32[256, 64, 1024]", view_472: "f32[256, 512, 64]", view_473: "f32[256, 64, 512]", view_456: "f32[1, 8192, 1024]", rsqrt_23: "f32[512, 16, 1]", rsqrt_22: "f32[512, 16, 1]", view_448: "f32[1, 8192, 1024]", view_449: "f32[1, 1024, 1024]", view_444: "f32[256, 512, 512]", view_445: "f32[256, 512, 64]", view_438: "f32[256, 512, 64]", view_439: "f32[256, 64, 1024]", view_434: "f32[256, 512, 64]", view_435: "f32[256, 64, 512]", view_418: "f32[1, 8192, 1024]", rsqrt_21: "f32[512, 16, 1]", rsqrt_20: "f32[512, 16, 1]", view_410: "f32[1, 8192, 1024]", view_411: "f32[1, 1024, 1024]", view_406: "f32[256, 512, 512]", view_407: "f32[256, 512, 64]", view_400: "f32[256, 512, 64]", view_401: "f32[256, 64, 1024]", view_396: "f32[256, 512, 64]", view_397: "f32[256, 64, 512]", view_380: "f32[1, 8192, 1024]", rsqrt_19: "f32[512, 16, 1]", rsqrt_18: "f32[512, 16, 1]", view_372: "f32[1, 8192, 1024]", view_373: "f32[1, 1024, 1024]", view_368: "f32[256, 512, 512]", view_369: "f32[256, 512, 64]", view_362: "f32[256, 512, 64]", view_363: "f32[256, 64, 1024]", view_358: "f32[256, 512, 64]", view_359: "f32[256, 64, 512]", view_342: "f32[1, 8192, 1024]", rsqrt_17: "f32[512, 16, 1]", rsqrt_16: "f32[512, 16, 1]", view_334: "f32[1, 8192, 1024]", view_335: "f32[1, 1024, 1024]", view_330: "f32[256, 512, 512]", view_331: "f32[256, 512, 64]", view_324: "f32[256, 512, 64]", view_325: "f32[256, 64, 1024]", view_320: "f32[256, 512, 64]", view_321: "f32[256, 64, 512]", view_304: "f32[1, 8192, 1024]", rsqrt_15: "f32[512, 16, 1]", rsqrt_14: "f32[512, 16, 1]", view_296: "f32[1, 8192, 1024]", view_297: "f32[1, 1024, 1024]", view_292: "f32[256, 512, 512]", view_293: "f32[256, 512, 64]", view_286: "f32[256, 512, 64]", view_287: "f32[256, 64, 1024]", view_282: "f32[256, 512, 64]", view_283: "f32[256, 64, 512]", view_266: "f32[1, 8192, 1024]", rsqrt_13: "f32[512, 16, 1]", rsqrt_12: "f32[512, 16, 1]", view_258: "f32[1, 8192, 1024]", view_259: "f32[1, 1024, 1024]", view_254: "f32[256, 512, 512]", view_255: "f32[256, 512, 64]", view_248: "f32[256, 512, 64]", view_249: "f32[256, 64, 1024]", view_244: "f32[256, 512, 64]", view_245: "f32[256, 64, 512]", view_228: "f32[1, 8192, 1024]", rsqrt_11: "f32[512, 16, 1]", rsqrt_10: "f32[512, 16, 1]", view_220: "f32[1, 8192, 1024]", view_221: "f32[1, 1024, 1024]", view_216: "f32[256, 512, 512]", view_217: "f32[256, 512, 64]", view_210: "f32[256, 512, 64]", view_211: "f32[256, 64, 1024]", view_206: "f32[256, 512, 64]", view_207: "f32[256, 64, 512]", view_190: "f32[1, 8192, 1024]", rsqrt_9: "f32[512, 16, 1]", rsqrt_8: "f32[512, 16, 1]", view_182: "f32[1, 8192, 1024]", view_183: "f32[1, 1024, 1024]", view_178: "f32[256, 512, 512]", view_179: "f32[256, 512, 64]", view_172: "f32[256, 512, 64]", view_173: "f32[256, 64, 1024]", view_168: "f32[256, 512, 64]", view_169: "f32[256, 64, 512]", view_152: "f32[1, 8192, 1024]", rsqrt_7: "f32[512, 16, 1]", rsqrt_6: "f32[512, 16, 1]", view_144: "f32[1, 8192, 1024]", view_145: "f32[1, 1024, 1024]", view_140: "f32[256, 512, 512]", view_141: "f32[256, 512, 64]", view_134: "f32[256, 512, 64]", view_135: "f32[256, 64, 1024]", view_130: "f32[256, 512, 64]", view_131: "f32[256, 64, 512]", view_114: "f32[1, 8192, 1024]", rsqrt_5: "f32[512, 16, 1]", rsqrt_4: "f32[512, 16, 1]", view_106: "f32[1, 8192, 1024]", view_107: "f32[1, 1024, 1024]", view_102: "f32[256, 512, 512]", view_103: "f32[256, 512, 64]", view_96: "f32[256, 512, 64]", view_97: "f32[256, 64, 1024]", view_92: "f32[256, 512, 64]", view_93: "f32[256, 64, 512]", view_76: "f32[1, 8192, 1024]", rsqrt_3: "f32[512, 16, 1]", rsqrt_2: "f32[512, 16, 1]", view_68: "f32[1, 8192, 1024]", view_69: "f32[1, 1024, 1024]", view_64: "f32[256, 512, 512]", view_65: "f32[256, 512, 64]", view_58: "f32[256, 512, 64]", view_59: "f32[256, 64, 1024]", view_54: "f32[256, 512, 64]", view_55: "f32[256, 64, 512]", view_38: "f32[1, 8192, 1024]", rsqrt_1: "f32[512, 16, 1]", rsqrt: "f32[512, 16, 1]", view_30: "f32[1, 8192, 1024]", view_31: "f32[1, 1024, 1024]", view_26: "f32[256, 512, 512]", view_27: "f32[256, 512, 64]", view_20: "f32[256, 512, 64]", view_21: "f32[256, 64, 1024]", view_16: "f32[256, 512, 64]", view_17: "f32[256, 64, 512]", view: "f32[1, 8192, 1024]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        reshape_default: "f32[16, 512, 32000]" = torch.ops.aten.reshape.default(addmm_48, _shape_param_0);  addmm_48 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1432 in forward, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        reshape_default_1: "f32[8192, 32000]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[8192]" = torch.ops.aten.reshape.default(primals_364, [-1]);  primals_364 = None
        amax_default: "f32[8192, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[8192, 32000]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8192, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[8192]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8192]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[8192]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[8192]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8192]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_2: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_904, [0, 2, 1]);  view_904 = None
        squeeze_dim_1: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default, 0);  permute_default = None
        permute_default_1: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_905, [0, 2, 1]);  view_905 = None
        squeeze_dim_2: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_1, 0);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_2: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_900, [0, 2, 1]);  view_900 = None
        permute_default_3: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_901, [0, 2, 1]);  view_901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_4: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_894, [0, 2, 1]);  view_894 = None
        permute_default_5: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_895, [0, 2, 1]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_6: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_890, [0, 2, 1]);  view_890 = None
        permute_default_7: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_891, [0, 2, 1]);  view_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        permute_default_8: "f32[1, 1024, 16384]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        squeeze_dim_3: "f32[1024, 16384]" = torch.ops.aten.squeeze.dim(permute_default_8, 0);  permute_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_9: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_874, [0, 2, 1]);  view_874 = None
        squeeze_dim_4: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_9, 0);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_3: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_4: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_10: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_866, [0, 2, 1]);  view_866 = None
        squeeze_dim_5: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_10, 0);  permute_default_10 = None
        permute_default_11: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_867, [0, 2, 1]);  view_867 = None
        squeeze_dim_6: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_11, 0);  permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_12: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_862, [0, 2, 1]);  view_862 = None
        permute_default_13: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_863, [0, 2, 1]);  view_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_14: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_856, [0, 2, 1]);  view_856 = None
        permute_default_15: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_857, [0, 2, 1]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_16: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_852, [0, 2, 1]);  view_852 = None
        permute_default_17: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_853, [0, 2, 1]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_18: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_836, [0, 2, 1]);  view_836 = None
        squeeze_dim_7: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_18, 0);  permute_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_5: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_6: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_19: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_828, [0, 2, 1]);  view_828 = None
        squeeze_dim_8: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_19, 0);  permute_default_19 = None
        permute_default_20: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_829, [0, 2, 1]);  view_829 = None
        squeeze_dim_9: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_20, 0);  permute_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_21: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_824, [0, 2, 1]);  view_824 = None
        permute_default_22: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_825, [0, 2, 1]);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_23: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_818, [0, 2, 1]);  view_818 = None
        permute_default_24: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_819, [0, 2, 1]);  view_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_25: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_814, [0, 2, 1]);  view_814 = None
        permute_default_26: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_815, [0, 2, 1]);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_27: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_798, [0, 2, 1]);  view_798 = None
        squeeze_dim_10: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_27, 0);  permute_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_7: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_8: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_28: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_790, [0, 2, 1]);  view_790 = None
        squeeze_dim_11: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_28, 0);  permute_default_28 = None
        permute_default_29: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_791, [0, 2, 1]);  view_791 = None
        squeeze_dim_12: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_29, 0);  permute_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_30: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_786, [0, 2, 1]);  view_786 = None
        permute_default_31: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_787, [0, 2, 1]);  view_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_32: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_780, [0, 2, 1]);  view_780 = None
        permute_default_33: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_781, [0, 2, 1]);  view_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_34: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_776, [0, 2, 1]);  view_776 = None
        permute_default_35: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_777, [0, 2, 1]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_36: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_760, [0, 2, 1]);  view_760 = None
        squeeze_dim_13: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_36, 0);  permute_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_9: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_10: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_37: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_752, [0, 2, 1]);  view_752 = None
        squeeze_dim_14: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_37, 0);  permute_default_37 = None
        permute_default_38: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_753, [0, 2, 1]);  view_753 = None
        squeeze_dim_15: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_38, 0);  permute_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_39: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_748, [0, 2, 1]);  view_748 = None
        permute_default_40: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_749, [0, 2, 1]);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_41: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_742, [0, 2, 1]);  view_742 = None
        permute_default_42: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_743, [0, 2, 1]);  view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_43: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_738, [0, 2, 1]);  view_738 = None
        permute_default_44: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_739, [0, 2, 1]);  view_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_45: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_722, [0, 2, 1]);  view_722 = None
        squeeze_dim_16: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_45, 0);  permute_default_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_11: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_12: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_46: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_714, [0, 2, 1]);  view_714 = None
        squeeze_dim_17: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_46, 0);  permute_default_46 = None
        permute_default_47: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
        squeeze_dim_18: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_47, 0);  permute_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_48: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_710, [0, 2, 1]);  view_710 = None
        permute_default_49: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_711, [0, 2, 1]);  view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_50: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_704, [0, 2, 1]);  view_704 = None
        permute_default_51: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_705, [0, 2, 1]);  view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_52: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_700, [0, 2, 1]);  view_700 = None
        permute_default_53: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_701, [0, 2, 1]);  view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_54: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_684, [0, 2, 1]);  view_684 = None
        squeeze_dim_19: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_54, 0);  permute_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_13: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_14: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_55: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_676, [0, 2, 1]);  view_676 = None
        squeeze_dim_20: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_55, 0);  permute_default_55 = None
        permute_default_56: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_677, [0, 2, 1]);  view_677 = None
        squeeze_dim_21: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_56, 0);  permute_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_57: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_672, [0, 2, 1]);  view_672 = None
        permute_default_58: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_673, [0, 2, 1]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_59: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_666, [0, 2, 1]);  view_666 = None
        permute_default_60: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_667, [0, 2, 1]);  view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_61: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_662, [0, 2, 1]);  view_662 = None
        permute_default_62: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_663, [0, 2, 1]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_63: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_646, [0, 2, 1]);  view_646 = None
        squeeze_dim_22: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_63, 0);  permute_default_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_15: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_16: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_64: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
        squeeze_dim_23: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_64, 0);  permute_default_64 = None
        permute_default_65: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_639, [0, 2, 1]);  view_639 = None
        squeeze_dim_24: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_65, 0);  permute_default_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_66: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_634, [0, 2, 1]);  view_634 = None
        permute_default_67: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_68: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_628, [0, 2, 1]);  view_628 = None
        permute_default_69: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_629, [0, 2, 1]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_70: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_624, [0, 2, 1]);  view_624 = None
        permute_default_71: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_625, [0, 2, 1]);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_72: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_608, [0, 2, 1]);  view_608 = None
        squeeze_dim_25: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_72, 0);  permute_default_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_17: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_18: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_73: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_600, [0, 2, 1]);  view_600 = None
        squeeze_dim_26: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_73, 0);  permute_default_73 = None
        permute_default_74: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_601, [0, 2, 1]);  view_601 = None
        squeeze_dim_27: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_74, 0);  permute_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_75: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_596, [0, 2, 1]);  view_596 = None
        permute_default_76: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_597, [0, 2, 1]);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_77: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_590, [0, 2, 1]);  view_590 = None
        permute_default_78: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_591, [0, 2, 1]);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_79: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_586, [0, 2, 1]);  view_586 = None
        permute_default_80: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_587, [0, 2, 1]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_81: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_570, [0, 2, 1]);  view_570 = None
        squeeze_dim_28: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_81, 0);  permute_default_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_19: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_20: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_82: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_562, [0, 2, 1]);  view_562 = None
        squeeze_dim_29: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_82, 0);  permute_default_82 = None
        permute_default_83: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        squeeze_dim_30: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_83, 0);  permute_default_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_84: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_558, [0, 2, 1]);  view_558 = None
        permute_default_85: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_559, [0, 2, 1]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_86: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_552, [0, 2, 1]);  view_552 = None
        permute_default_87: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_88: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None
        permute_default_89: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_549, [0, 2, 1]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_90: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_532, [0, 2, 1]);  view_532 = None
        squeeze_dim_31: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_90, 0);  permute_default_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_21: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_22: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_91: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_524, [0, 2, 1]);  view_524 = None
        squeeze_dim_32: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_91, 0);  permute_default_91 = None
        permute_default_92: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_525, [0, 2, 1]);  view_525 = None
        squeeze_dim_33: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_92, 0);  permute_default_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_93: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
        permute_default_94: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_95: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_514, [0, 2, 1]);  view_514 = None
        permute_default_96: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_97: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_510, [0, 2, 1]);  view_510 = None
        permute_default_98: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_99: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
        squeeze_dim_34: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_99, 0);  permute_default_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_23: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_24: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_100: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_486, [0, 2, 1]);  view_486 = None
        squeeze_dim_35: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_100, 0);  permute_default_100 = None
        permute_default_101: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_487, [0, 2, 1]);  view_487 = None
        squeeze_dim_36: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_101, 0);  permute_default_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_102: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_482, [0, 2, 1]);  view_482 = None
        permute_default_103: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_483, [0, 2, 1]);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_104: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_476, [0, 2, 1]);  view_476 = None
        permute_default_105: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_477, [0, 2, 1]);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_106: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
        permute_default_107: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_108: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
        squeeze_dim_37: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_108, 0);  permute_default_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_25: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_26: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_109: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
        squeeze_dim_38: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_109, 0);  permute_default_109 = None
        permute_default_110: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        squeeze_dim_39: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_110, 0);  permute_default_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_111: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
        permute_default_112: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_113: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
        permute_default_114: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_115: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_434, [0, 2, 1]);  view_434 = None
        permute_default_116: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_435, [0, 2, 1]);  view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_117: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        squeeze_dim_40: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_117, 0);  permute_default_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_27: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_28: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_118: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
        squeeze_dim_41: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_118, 0);  permute_default_118 = None
        permute_default_119: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
        squeeze_dim_42: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_119, 0);  permute_default_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_120: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
        permute_default_121: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_122: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_400, [0, 2, 1]);  view_400 = None
        permute_default_123: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_124: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_396, [0, 2, 1]);  view_396 = None
        permute_default_125: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_126: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
        squeeze_dim_43: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_126, 0);  permute_default_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_29: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_30: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_127: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
        squeeze_dim_44: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_127, 0);  permute_default_127 = None
        permute_default_128: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
        squeeze_dim_45: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_128, 0);  permute_default_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_129: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
        permute_default_130: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_131: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None
        permute_default_132: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_133: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
        permute_default_134: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_135: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
        squeeze_dim_46: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_135, 0);  permute_default_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_31: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_32: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_136: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
        squeeze_dim_47: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_136, 0);  permute_default_136 = None
        permute_default_137: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
        squeeze_dim_48: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_137, 0);  permute_default_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_138: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None
        permute_default_139: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_140: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
        permute_default_141: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_142: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        permute_default_143: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_144: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
        squeeze_dim_49: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_144, 0);  permute_default_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_33: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_34: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_145: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
        squeeze_dim_50: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_145, 0);  permute_default_145 = None
        permute_default_146: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
        squeeze_dim_51: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_146, 0);  permute_default_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_147: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
        permute_default_148: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_149: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
        permute_default_150: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_151: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_282, [0, 2, 1]);  view_282 = None
        permute_default_152: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_153: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        squeeze_dim_52: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_153, 0);  permute_default_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_35: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_36: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_154: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
        squeeze_dim_53: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_154, 0);  permute_default_154 = None
        permute_default_155: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
        squeeze_dim_54: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_155, 0);  permute_default_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_156: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_default_157: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_158: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
        permute_default_159: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_160: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
        permute_default_161: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_162: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        squeeze_dim_55: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_162, 0);  permute_default_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_37: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_38: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_163: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
        squeeze_dim_56: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_163, 0);  permute_default_163 = None
        permute_default_164: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
        squeeze_dim_57: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_164, 0);  permute_default_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_165: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
        permute_default_166: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_167: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_default_168: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_169: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
        permute_default_170: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_171: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        squeeze_dim_58: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_171, 0);  permute_default_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_39: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_40: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_172: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
        squeeze_dim_59: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_172, 0);  permute_default_172 = None
        permute_default_173: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        squeeze_dim_60: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_173, 0);  permute_default_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_174: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
        permute_default_175: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_176: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        permute_default_177: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_178: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_default_179: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_180: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
        squeeze_dim_61: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_180, 0);  permute_default_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_41: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_42: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_181: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        squeeze_dim_62: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_181, 0);  permute_default_181 = None
        permute_default_182: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        squeeze_dim_63: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_182, 0);  permute_default_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_183: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
        permute_default_184: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_185: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
        permute_default_186: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_187: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_default_188: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_189: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
        squeeze_dim_64: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_189, 0);  permute_default_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_43: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_44: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_190: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        squeeze_dim_65: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_190, 0);  permute_default_190 = None
        permute_default_191: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        squeeze_dim_66: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_191, 0);  permute_default_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_192: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        permute_default_193: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_194: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
        permute_default_195: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_196: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
        permute_default_197: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_198: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
        squeeze_dim_67: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_198, 0);  permute_default_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_45: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_46: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_199: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        squeeze_dim_68: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_199, 0);  permute_default_199 = None
        permute_default_200: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        squeeze_dim_69: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_200, 0);  permute_default_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_201: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_default_202: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_203: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_default_204: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_205: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
        permute_default_206: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_207: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
        squeeze_dim_70: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_207, 0);  permute_default_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_tensor_47: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_tensor_48: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_default_208: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        squeeze_dim_71: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_208, 0);  permute_default_208 = None
        permute_default_209: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        squeeze_dim_72: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_209, 0);  permute_default_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_default_210: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
        permute_default_211: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_default_212: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
        permute_default_213: "f32[256, 1024, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_default_214: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_default_215: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_default_216: "f32[1, 1024, 8192]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        squeeze_dim_73: "f32[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_default_216, 0);  permute_default_216 = None
        return (div_tensor, div_tensor_1, div_tensor_2, squeeze_dim_1, squeeze_dim_2, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, squeeze_dim_3, squeeze_dim_4, div_tensor_3, div_tensor_4, squeeze_dim_5, squeeze_dim_6, permute_default_12, permute_default_13, permute_default_14, permute_default_15, permute_default_16, permute_default_17, squeeze_dim_7, div_tensor_5, div_tensor_6, squeeze_dim_8, squeeze_dim_9, permute_default_21, permute_default_22, permute_default_23, permute_default_24, permute_default_25, permute_default_26, squeeze_dim_10, div_tensor_7, div_tensor_8, squeeze_dim_11, squeeze_dim_12, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, permute_default_35, squeeze_dim_13, div_tensor_9, div_tensor_10, squeeze_dim_14, squeeze_dim_15, permute_default_39, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, squeeze_dim_16, div_tensor_11, div_tensor_12, squeeze_dim_17, squeeze_dim_18, permute_default_48, permute_default_49, permute_default_50, permute_default_51, permute_default_52, permute_default_53, squeeze_dim_19, div_tensor_13, div_tensor_14, squeeze_dim_20, squeeze_dim_21, permute_default_57, permute_default_58, permute_default_59, permute_default_60, permute_default_61, permute_default_62, squeeze_dim_22, div_tensor_15, div_tensor_16, squeeze_dim_23, squeeze_dim_24, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_70, permute_default_71, squeeze_dim_25, div_tensor_17, div_tensor_18, squeeze_dim_26, squeeze_dim_27, permute_default_75, permute_default_76, permute_default_77, permute_default_78, permute_default_79, permute_default_80, squeeze_dim_28, div_tensor_19, div_tensor_20, squeeze_dim_29, squeeze_dim_30, permute_default_84, permute_default_85, permute_default_86, permute_default_87, permute_default_88, permute_default_89, squeeze_dim_31, div_tensor_21, div_tensor_22, squeeze_dim_32, squeeze_dim_33, permute_default_93, permute_default_94, permute_default_95, permute_default_96, permute_default_97, permute_default_98, squeeze_dim_34, div_tensor_23, div_tensor_24, squeeze_dim_35, squeeze_dim_36, permute_default_102, permute_default_103, permute_default_104, permute_default_105, permute_default_106, permute_default_107, squeeze_dim_37, div_tensor_25, div_tensor_26, squeeze_dim_38, squeeze_dim_39, permute_default_111, permute_default_112, permute_default_113, permute_default_114, permute_default_115, permute_default_116, squeeze_dim_40, div_tensor_27, div_tensor_28, squeeze_dim_41, squeeze_dim_42, permute_default_120, permute_default_121, permute_default_122, permute_default_123, permute_default_124, permute_default_125, squeeze_dim_43, div_tensor_29, div_tensor_30, squeeze_dim_44, squeeze_dim_45, permute_default_129, permute_default_130, permute_default_131, permute_default_132, permute_default_133, permute_default_134, squeeze_dim_46, div_tensor_31, div_tensor_32, squeeze_dim_47, squeeze_dim_48, permute_default_138, permute_default_139, permute_default_140, permute_default_141, permute_default_142, permute_default_143, squeeze_dim_49, div_tensor_33, div_tensor_34, squeeze_dim_50, squeeze_dim_51, permute_default_147, permute_default_148, permute_default_149, permute_default_150, permute_default_151, permute_default_152, squeeze_dim_52, div_tensor_35, div_tensor_36, squeeze_dim_53, squeeze_dim_54, permute_default_156, permute_default_157, permute_default_158, permute_default_159, permute_default_160, permute_default_161, squeeze_dim_55, div_tensor_37, div_tensor_38, squeeze_dim_56, squeeze_dim_57, permute_default_165, permute_default_166, permute_default_167, permute_default_168, permute_default_169, permute_default_170, squeeze_dim_58, div_tensor_39, div_tensor_40, squeeze_dim_59, squeeze_dim_60, permute_default_174, permute_default_175, permute_default_176, permute_default_177, permute_default_178, permute_default_179, squeeze_dim_61, div_tensor_41, div_tensor_42, squeeze_dim_62, squeeze_dim_63, permute_default_183, permute_default_184, permute_default_185, permute_default_186, permute_default_187, permute_default_188, squeeze_dim_64, div_tensor_43, div_tensor_44, squeeze_dim_65, squeeze_dim_66, permute_default_192, permute_default_193, permute_default_194, permute_default_195, permute_default_196, permute_default_197, squeeze_dim_67, div_tensor_45, div_tensor_46, squeeze_dim_68, squeeze_dim_69, permute_default_201, permute_default_202, permute_default_203, permute_default_204, permute_default_205, permute_default_206, squeeze_dim_70, div_tensor_47, div_tensor_48, squeeze_dim_71, squeeze_dim_72, permute_default_210, permute_default_211, permute_default_212, permute_default_213, permute_default_214, permute_default_215, squeeze_dim_73)


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
