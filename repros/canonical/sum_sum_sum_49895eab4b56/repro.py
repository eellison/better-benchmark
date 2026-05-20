"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 49895eab4b56
Shape hash: 61b861ba
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 32000], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([512, 16, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([512, 16, 16, 64], f32, stride=(64, 524288, 32768, 1)), T([1024, 1024], f32), T([1024, 1024], f32), T([8192, 1024], f32), T([512, 16, 1024], f32), T([1024, 1024], f32), T([8192, 1024], f32), T([1024, 1024], f32), T([8192, 1024], f32), T([512, 16, 1024], b8), T([512, 16], i64, max=32000), T([], f32), T([32000, 1024], f32), S([32000]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([1024]), S([4096]), S([64, 16, 1024, 1, 1]), S([16, 64]), S([16, 64]), S([1024, 16, 64, 1, 1]), S([1024, 16, 64, 1, 1]), S([512, 16, 1024, 1, 1]), S([1024, 16, 64, 1, 1]), S([512, 16, 1024, 1, 1]), S([1024, 16, 64, 1, 1]), S([512, 16, 1024, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, view_917: "f32[8192, 32000]", clone_50: "f32[512, 16, 1024]", mul_388: "f32[512, 16, 1024]", view_920: "f32[8192, 1024]", view_923: "f32[8192, 4096]", add_267: "f32[512, 16, 1024]", mul_379: "f32[512, 16, 1024]", mm_default_215: "f32[1024, 1024]", squeeze_7: "f32[512, 16, 16, 64]", squeeze_9: "f32[512, 16, 16, 64]", mm_default_213: "f32[1024, 1024]", mm_default_212: "f32[1024, 1024]", mm_default_210: "f32[1024, 1024]", mm_default_208: "f32[1024, 1024]", add_271: "f32[512, 16, 1024]", mul_372: "f32[512, 16, 1024]", view_961: "f32[8192, 1024]", view_964: "f32[8192, 4096]", add_274: "f32[512, 16, 1024]", mul_363: "f32[512, 16, 1024]", mm_default_206: "f32[1024, 1024]", squeeze_30: "f32[512, 16, 16, 64]", squeeze_32: "f32[512, 16, 16, 64]", mm_default_204: "f32[1024, 1024]", mm_default_203: "f32[1024, 1024]", mm_default_201: "f32[1024, 1024]", mm_default_199: "f32[1024, 1024]", add_278: "f32[512, 16, 1024]", mul_356: "f32[512, 16, 1024]", view_1002: "f32[8192, 1024]", view_1005: "f32[8192, 4096]", add_281: "f32[512, 16, 1024]", mul_347: "f32[512, 16, 1024]", mm_default_197: "f32[1024, 1024]", squeeze_53: "f32[512, 16, 16, 64]", squeeze_55: "f32[512, 16, 16, 64]", mm_default_195: "f32[1024, 1024]", mm_default_194: "f32[1024, 1024]", mm_default_192: "f32[1024, 1024]", mm_default_190: "f32[1024, 1024]", add_285: "f32[512, 16, 1024]", mul_340: "f32[512, 16, 1024]", view_1043: "f32[8192, 1024]", view_1046: "f32[8192, 4096]", add_288: "f32[512, 16, 1024]", mul_331: "f32[512, 16, 1024]", mm_default_188: "f32[1024, 1024]", squeeze_76: "f32[512, 16, 16, 64]", squeeze_78: "f32[512, 16, 16, 64]", mm_default_186: "f32[1024, 1024]", mm_default_185: "f32[1024, 1024]", mm_default_183: "f32[1024, 1024]", mm_default_181: "f32[1024, 1024]", add_292: "f32[512, 16, 1024]", mul_324: "f32[512, 16, 1024]", view_1084: "f32[8192, 1024]", view_1087: "f32[8192, 4096]", add_295: "f32[512, 16, 1024]", mul_315: "f32[512, 16, 1024]", mm_default_179: "f32[1024, 1024]", squeeze_99: "f32[512, 16, 16, 64]", squeeze_101: "f32[512, 16, 16, 64]", mm_default_177: "f32[1024, 1024]", mm_default_176: "f32[1024, 1024]", mm_default_174: "f32[1024, 1024]", mm_default_172: "f32[1024, 1024]", add_299: "f32[512, 16, 1024]", mul_308: "f32[512, 16, 1024]", view_1125: "f32[8192, 1024]", view_1128: "f32[8192, 4096]", add_302: "f32[512, 16, 1024]", mul_299: "f32[512, 16, 1024]", mm_default_170: "f32[1024, 1024]", squeeze_122: "f32[512, 16, 16, 64]", squeeze_124: "f32[512, 16, 16, 64]", mm_default_168: "f32[1024, 1024]", mm_default_167: "f32[1024, 1024]", mm_default_165: "f32[1024, 1024]", mm_default_163: "f32[1024, 1024]", add_306: "f32[512, 16, 1024]", mul_292: "f32[512, 16, 1024]", view_1166: "f32[8192, 1024]", view_1169: "f32[8192, 4096]", add_309: "f32[512, 16, 1024]", mul_283: "f32[512, 16, 1024]", mm_default_161: "f32[1024, 1024]", squeeze_145: "f32[512, 16, 16, 64]", squeeze_147: "f32[512, 16, 16, 64]", mm_default_159: "f32[1024, 1024]", mm_default_158: "f32[1024, 1024]", mm_default_156: "f32[1024, 1024]", mm_default_154: "f32[1024, 1024]", add_313: "f32[512, 16, 1024]", mul_276: "f32[512, 16, 1024]", view_1207: "f32[8192, 1024]", view_1210: "f32[8192, 4096]", add_316: "f32[512, 16, 1024]", mul_267: "f32[512, 16, 1024]", mm_default_152: "f32[1024, 1024]", squeeze_168: "f32[512, 16, 16, 64]", squeeze_170: "f32[512, 16, 16, 64]", mm_default_150: "f32[1024, 1024]", mm_default_149: "f32[1024, 1024]", mm_default_147: "f32[1024, 1024]", mm_default_145: "f32[1024, 1024]", add_320: "f32[512, 16, 1024]", mul_260: "f32[512, 16, 1024]", view_1248: "f32[8192, 1024]", view_1251: "f32[8192, 4096]", add_323: "f32[512, 16, 1024]", mul_251: "f32[512, 16, 1024]", mm_default_143: "f32[1024, 1024]", squeeze_191: "f32[512, 16, 16, 64]", squeeze_193: "f32[512, 16, 16, 64]", mm_default_141: "f32[1024, 1024]", mm_default_140: "f32[1024, 1024]", mm_default_138: "f32[1024, 1024]", mm_default_136: "f32[1024, 1024]", add_327: "f32[512, 16, 1024]", mul_244: "f32[512, 16, 1024]", view_1289: "f32[8192, 1024]", view_1292: "f32[8192, 4096]", add_330: "f32[512, 16, 1024]", mul_235: "f32[512, 16, 1024]", mm_default_134: "f32[1024, 1024]", squeeze_214: "f32[512, 16, 16, 64]", squeeze_216: "f32[512, 16, 16, 64]", mm_default_132: "f32[1024, 1024]", mm_default_131: "f32[1024, 1024]", mm_default_129: "f32[1024, 1024]", mm_default_127: "f32[1024, 1024]", add_334: "f32[512, 16, 1024]", mul_228: "f32[512, 16, 1024]", view_1330: "f32[8192, 1024]", view_1333: "f32[8192, 4096]", add_337: "f32[512, 16, 1024]", mul_219: "f32[512, 16, 1024]", mm_default_125: "f32[1024, 1024]", squeeze_237: "f32[512, 16, 16, 64]", squeeze_239: "f32[512, 16, 16, 64]", mm_default_123: "f32[1024, 1024]", mm_default_122: "f32[1024, 1024]", mm_default_120: "f32[1024, 1024]", mm_default_118: "f32[1024, 1024]", add_341: "f32[512, 16, 1024]", mul_212: "f32[512, 16, 1024]", view_1371: "f32[8192, 1024]", view_1374: "f32[8192, 4096]", add_344: "f32[512, 16, 1024]", mul_203: "f32[512, 16, 1024]", mm_default_116: "f32[1024, 1024]", squeeze_260: "f32[512, 16, 16, 64]", squeeze_262: "f32[512, 16, 16, 64]", mm_default_114: "f32[1024, 1024]", mm_default_113: "f32[1024, 1024]", mm_default_111: "f32[1024, 1024]", mm_default_109: "f32[1024, 1024]", add_348: "f32[512, 16, 1024]", mul_196: "f32[512, 16, 1024]", view_1412: "f32[8192, 1024]", view_1415: "f32[8192, 4096]", add_351: "f32[512, 16, 1024]", mul_187: "f32[512, 16, 1024]", mm_default_107: "f32[1024, 1024]", squeeze_283: "f32[512, 16, 16, 64]", squeeze_285: "f32[512, 16, 16, 64]", mm_default_105: "f32[1024, 1024]", mm_default_104: "f32[1024, 1024]", mm_default_102: "f32[1024, 1024]", mm_default_100: "f32[1024, 1024]", add_355: "f32[512, 16, 1024]", mul_180: "f32[512, 16, 1024]", view_1453: "f32[8192, 1024]", view_1456: "f32[8192, 4096]", add_358: "f32[512, 16, 1024]", mul_171: "f32[512, 16, 1024]", mm_default_98: "f32[1024, 1024]", squeeze_306: "f32[512, 16, 16, 64]", squeeze_308: "f32[512, 16, 16, 64]", mm_default_96: "f32[1024, 1024]", mm_default_95: "f32[1024, 1024]", mm_default_93: "f32[1024, 1024]", mm_default_91: "f32[1024, 1024]", add_362: "f32[512, 16, 1024]", mul_164: "f32[512, 16, 1024]", view_1494: "f32[8192, 1024]", view_1497: "f32[8192, 4096]", add_365: "f32[512, 16, 1024]", mul_155: "f32[512, 16, 1024]", mm_default_89: "f32[1024, 1024]", squeeze_329: "f32[512, 16, 16, 64]", squeeze_331: "f32[512, 16, 16, 64]", mm_default_87: "f32[1024, 1024]", mm_default_86: "f32[1024, 1024]", mm_default_84: "f32[1024, 1024]", mm_default_82: "f32[1024, 1024]", add_369: "f32[512, 16, 1024]", mul_148: "f32[512, 16, 1024]", view_1535: "f32[8192, 1024]", view_1538: "f32[8192, 4096]", add_372: "f32[512, 16, 1024]", mul_139: "f32[512, 16, 1024]", mm_default_80: "f32[1024, 1024]", squeeze_352: "f32[512, 16, 16, 64]", squeeze_354: "f32[512, 16, 16, 64]", mm_default_78: "f32[1024, 1024]", mm_default_77: "f32[1024, 1024]", mm_default_75: "f32[1024, 1024]", mm_default_73: "f32[1024, 1024]", add_376: "f32[512, 16, 1024]", mul_132: "f32[512, 16, 1024]", view_1576: "f32[8192, 1024]", view_1579: "f32[8192, 4096]", add_379: "f32[512, 16, 1024]", mul_123: "f32[512, 16, 1024]", mm_default_71: "f32[1024, 1024]", squeeze_375: "f32[512, 16, 16, 64]", squeeze_377: "f32[512, 16, 16, 64]", mm_default_69: "f32[1024, 1024]", mm_default_68: "f32[1024, 1024]", mm_default_66: "f32[1024, 1024]", mm_default_64: "f32[1024, 1024]", add_383: "f32[512, 16, 1024]", mul_116: "f32[512, 16, 1024]", view_1617: "f32[8192, 1024]", view_1620: "f32[8192, 4096]", add_386: "f32[512, 16, 1024]", mul_107: "f32[512, 16, 1024]", mm_default_62: "f32[1024, 1024]", squeeze_398: "f32[512, 16, 16, 64]", squeeze_400: "f32[512, 16, 16, 64]", mm_default_60: "f32[1024, 1024]", mm_default_59: "f32[1024, 1024]", mm_default_57: "f32[1024, 1024]", mm_default_55: "f32[1024, 1024]", add_390: "f32[512, 16, 1024]", mul_100: "f32[512, 16, 1024]", view_1658: "f32[8192, 1024]", view_1661: "f32[8192, 4096]", add_393: "f32[512, 16, 1024]", mul_91: "f32[512, 16, 1024]", mm_default_53: "f32[1024, 1024]", squeeze_421: "f32[512, 16, 16, 64]", squeeze_423: "f32[512, 16, 16, 64]", mm_default_51: "f32[1024, 1024]", mm_default_50: "f32[1024, 1024]", mm_default_48: "f32[1024, 1024]", mm_default_46: "f32[1024, 1024]", add_397: "f32[512, 16, 1024]", mul_84: "f32[512, 16, 1024]", view_1699: "f32[8192, 1024]", view_1702: "f32[8192, 4096]", add_400: "f32[512, 16, 1024]", mul_75: "f32[512, 16, 1024]", mm_default_44: "f32[1024, 1024]", squeeze_444: "f32[512, 16, 16, 64]", squeeze_446: "f32[512, 16, 16, 64]", mm_default_42: "f32[1024, 1024]", mm_default_41: "f32[1024, 1024]", mm_default_39: "f32[1024, 1024]", mm_default_37: "f32[1024, 1024]", add_404: "f32[512, 16, 1024]", mul_68: "f32[512, 16, 1024]", view_1740: "f32[8192, 1024]", view_1743: "f32[8192, 4096]", add_407: "f32[512, 16, 1024]", mul_59: "f32[512, 16, 1024]", mm_default_35: "f32[1024, 1024]", squeeze_467: "f32[512, 16, 16, 64]", squeeze_469: "f32[512, 16, 16, 64]", mm_default_33: "f32[1024, 1024]", mm_default_32: "f32[1024, 1024]", mm_default_30: "f32[1024, 1024]", mm_default_28: "f32[1024, 1024]", add_411: "f32[512, 16, 1024]", mul_52: "f32[512, 16, 1024]", view_1781: "f32[8192, 1024]", view_1784: "f32[8192, 4096]", add_414: "f32[512, 16, 1024]", mul_43: "f32[512, 16, 1024]", mm_default_26: "f32[1024, 1024]", squeeze_490: "f32[512, 16, 16, 64]", squeeze_492: "f32[512, 16, 16, 64]", mm_default_24: "f32[1024, 1024]", mm_default_23: "f32[1024, 1024]", mm_default_21: "f32[1024, 1024]", mm_default_19: "f32[1024, 1024]", add_418: "f32[512, 16, 1024]", mul_36: "f32[512, 16, 1024]", view_1822: "f32[8192, 1024]", view_1825: "f32[8192, 4096]", add_421: "f32[512, 16, 1024]", mul_27: "f32[512, 16, 1024]", mm_default_17: "f32[1024, 1024]", squeeze_513: "f32[512, 16, 16, 64]", squeeze_515: "f32[512, 16, 16, 64]", mm_default_15: "f32[1024, 1024]", mm_default_14: "f32[1024, 1024]", mm_default_12: "f32[1024, 1024]", mm_default_10: "f32[1024, 1024]", add_425: "f32[512, 16, 1024]", mul_20: "f32[512, 16, 1024]", view_1863: "f32[8192, 1024]", view_1866: "f32[8192, 4096]", add_428: "f32[512, 16, 1024]", mul_11: "f32[512, 16, 1024]", mm_default_8: "f32[1024, 1024]", squeeze_536: "f32[512, 16, 16, 64]", squeeze_538: "f32[512, 16, 16, 64]", mm_default_6: "f32[1024, 1024]", mm_default_5: "f32[1024, 1024]", mm_default_4: "f32[8192, 1024]", mul_1132: "f32[512, 16, 1024]", mm_default_3: "f32[1024, 1024]", mm_default_2: "f32[8192, 1024]", mm_default_1: "f32[1024, 1024]", mm_default: "f32[8192, 1024]", gt: "b8[512, 16, 1024]", clone: "i64[512, 16]", full_default_1: "f32[]", mm_1: "f32[32000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        sum_dim_int_list: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_917, [0], True);  view_917 = None
        reshape_default: "f32[32000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(clone_50, mul_388);  mul_388 = None
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(clone_50, [0, 1]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_920, [1, 0])
        sum_dim_int_list_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
        reshape_default_1: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_1: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_923, [1, 0])
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_923, [0], True);  view_923 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_267, mul_379);  mul_379 = None
        sum_dim_int_list_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_215, 0);  mm_default_215 = None
        reshape_default_3: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_3);  unsqueeze_default = _shape_param_3 = None
        permute_default_2: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_3, [3, 4, 2, 0, 1]);  reshape_default_3 = None
        permute_default_3: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_2, [2, 4, 3, 0, 1]);  permute_default_2 = None
        squeeze_dim: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_3, 4);  permute_default_3 = None
        squeeze_dim_1: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim, 3);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_7: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_7, [0, 1], True);  squeeze_7 = None
        reshape_default_4: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_4);  sum_dim_int_list_7 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_8: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_9, [0, 1], True);  squeeze_9 = None
        reshape_default_5: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_5);  sum_dim_int_list_8 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_1: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_213, 0);  mm_default_213 = None
        reshape_default_6: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_6);  unsqueeze_default_1 = _shape_param_6 = None
        squeeze_dim_2: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_6, 4);  reshape_default_6 = None
        squeeze_dim_3: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_2, 3);  squeeze_dim_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_2: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_212, 0);  mm_default_212 = None
        reshape_default_7: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_2, _shape_param_7);  unsqueeze_default_2 = _shape_param_7 = None
        squeeze_dim_4: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_7, 4);  reshape_default_7 = None
        squeeze_dim_5: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_4, 3);  squeeze_dim_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_3: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_210, 0);  mm_default_210 = None
        reshape_default_8: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_8);  unsqueeze_default_3 = _shape_param_8 = None
        squeeze_dim_6: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_8, 4);  reshape_default_8 = None
        squeeze_dim_7: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_6, 3);  squeeze_dim_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_4: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_208, 0);  mm_default_208 = None
        reshape_default_9: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_4, _shape_param_9);  unsqueeze_default_4 = _shape_param_9 = None
        squeeze_dim_8: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_9, 4);  reshape_default_9 = None
        squeeze_dim_9: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_8, 3);  squeeze_dim_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_271, mul_372);  mul_372 = None
        sum_dim_int_list_9: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_271, [0, 1]);  add_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_4: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_961, [1, 0])
        sum_dim_int_list_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
        reshape_default_10: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_10);  sum_dim_int_list_11 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_5: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_964, [1, 0])
        sum_dim_int_list_12: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
        reshape_default_11: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_11);  sum_dim_int_list_12 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_274, mul_363);  mul_363 = None
        sum_dim_int_list_13: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_5: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_206, 0);  mm_default_206 = None
        reshape_default_12: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_12);  unsqueeze_default_5 = _shape_param_12 = None
        permute_default_6: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_12, [3, 4, 2, 0, 1]);  reshape_default_12 = None
        permute_default_7: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_6, [2, 4, 3, 0, 1]);  permute_default_6 = None
        squeeze_dim_10: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_7, 4);  permute_default_7 = None
        squeeze_dim_11: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_10, 3);  squeeze_dim_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_15: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_30, [0, 1], True);  squeeze_30 = None
        reshape_default_13: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_13);  sum_dim_int_list_15 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_16: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_32, [0, 1], True);  squeeze_32 = None
        reshape_default_14: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_14);  sum_dim_int_list_16 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_6: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_204, 0);  mm_default_204 = None
        reshape_default_15: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_6, _shape_param_15);  unsqueeze_default_6 = _shape_param_15 = None
        squeeze_dim_12: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_15, 4);  reshape_default_15 = None
        squeeze_dim_13: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_12, 3);  squeeze_dim_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_7: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_203, 0);  mm_default_203 = None
        reshape_default_16: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_7, _shape_param_16);  unsqueeze_default_7 = _shape_param_16 = None
        squeeze_dim_14: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_16, 4);  reshape_default_16 = None
        squeeze_dim_15: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_14, 3);  squeeze_dim_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_8: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_201, 0);  mm_default_201 = None
        reshape_default_17: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_8, _shape_param_17);  unsqueeze_default_8 = _shape_param_17 = None
        squeeze_dim_16: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_17, 4);  reshape_default_17 = None
        squeeze_dim_17: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_16, 3);  squeeze_dim_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_9: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_199, 0);  mm_default_199 = None
        reshape_default_18: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_9, _shape_param_18);  unsqueeze_default_9 = _shape_param_18 = None
        squeeze_dim_18: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_18, 4);  reshape_default_18 = None
        squeeze_dim_19: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_18, 3);  squeeze_dim_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_278, mul_356);  mul_356 = None
        sum_dim_int_list_17: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_18: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_8: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1002, [1, 0])
        sum_dim_int_list_19: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True);  view_1002 = None
        reshape_default_19: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_19);  sum_dim_int_list_19 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_9: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1005, [1, 0])
        sum_dim_int_list_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
        reshape_default_20: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_20);  sum_dim_int_list_20 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_281, mul_347);  mul_347 = None
        sum_dim_int_list_21: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_10: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_197, 0);  mm_default_197 = None
        reshape_default_21: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_10, _shape_param_21);  unsqueeze_default_10 = _shape_param_21 = None
        permute_default_10: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_21, [3, 4, 2, 0, 1]);  reshape_default_21 = None
        permute_default_11: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_10, [2, 4, 3, 0, 1]);  permute_default_10 = None
        squeeze_dim_20: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_11, 4);  permute_default_11 = None
        squeeze_dim_21: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_20, 3);  squeeze_dim_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_23: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_53, [0, 1], True);  squeeze_53 = None
        reshape_default_22: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_22);  sum_dim_int_list_23 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_24: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_55, [0, 1], True);  squeeze_55 = None
        reshape_default_23: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_23);  sum_dim_int_list_24 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_11: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_195, 0);  mm_default_195 = None
        reshape_default_24: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_11, _shape_param_24);  unsqueeze_default_11 = _shape_param_24 = None
        squeeze_dim_22: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_24, 4);  reshape_default_24 = None
        squeeze_dim_23: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_22, 3);  squeeze_dim_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_12: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_194, 0);  mm_default_194 = None
        reshape_default_25: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_12, _shape_param_25);  unsqueeze_default_12 = _shape_param_25 = None
        squeeze_dim_24: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_25, 4);  reshape_default_25 = None
        squeeze_dim_25: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_24, 3);  squeeze_dim_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_13: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_192, 0);  mm_default_192 = None
        reshape_default_26: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_13, _shape_param_26);  unsqueeze_default_13 = _shape_param_26 = None
        squeeze_dim_26: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_26, 4);  reshape_default_26 = None
        squeeze_dim_27: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_26, 3);  squeeze_dim_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_14: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_190, 0);  mm_default_190 = None
        reshape_default_27: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_14, _shape_param_27);  unsqueeze_default_14 = _shape_param_27 = None
        squeeze_dim_28: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_27, 4);  reshape_default_27 = None
        squeeze_dim_29: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_28, 3);  squeeze_dim_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_6: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_285, mul_340);  mul_340 = None
        sum_dim_int_list_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_285, [0, 1]);  add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_12: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1043, [1, 0])
        sum_dim_int_list_27: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
        reshape_default_28: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_28);  sum_dim_int_list_27 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_13: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1046, [1, 0])
        sum_dim_int_list_28: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1046, [0], True);  view_1046 = None
        reshape_default_29: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_29);  sum_dim_int_list_28 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_7: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_288, mul_331);  mul_331 = None
        sum_dim_int_list_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_30: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_288, [0, 1]);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_15: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_188, 0);  mm_default_188 = None
        reshape_default_30: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_15, _shape_param_30);  unsqueeze_default_15 = _shape_param_30 = None
        permute_default_14: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_30, [3, 4, 2, 0, 1]);  reshape_default_30 = None
        permute_default_15: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_14, [2, 4, 3, 0, 1]);  permute_default_14 = None
        squeeze_dim_30: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_15, 4);  permute_default_15 = None
        squeeze_dim_31: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_30, 3);  squeeze_dim_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_31: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_76, [0, 1], True);  squeeze_76 = None
        reshape_default_31: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_31);  sum_dim_int_list_31 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_32: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_78, [0, 1], True);  squeeze_78 = None
        reshape_default_32: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_32);  sum_dim_int_list_32 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_16: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_186, 0);  mm_default_186 = None
        reshape_default_33: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_16, _shape_param_33);  unsqueeze_default_16 = _shape_param_33 = None
        squeeze_dim_32: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_33, 4);  reshape_default_33 = None
        squeeze_dim_33: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_32, 3);  squeeze_dim_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_17: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_185, 0);  mm_default_185 = None
        reshape_default_34: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_17, _shape_param_34);  unsqueeze_default_17 = _shape_param_34 = None
        squeeze_dim_34: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_34, 4);  reshape_default_34 = None
        squeeze_dim_35: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_34, 3);  squeeze_dim_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_18: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_183, 0);  mm_default_183 = None
        reshape_default_35: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_18, _shape_param_35);  unsqueeze_default_18 = _shape_param_35 = None
        squeeze_dim_36: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_35, 4);  reshape_default_35 = None
        squeeze_dim_37: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_36, 3);  squeeze_dim_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_19: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_181, 0);  mm_default_181 = None
        reshape_default_36: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_19, _shape_param_36);  unsqueeze_default_19 = _shape_param_36 = None
        squeeze_dim_38: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_36, 4);  reshape_default_36 = None
        squeeze_dim_39: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_38, 3);  squeeze_dim_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_8: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_292, mul_324);  mul_324 = None
        sum_dim_int_list_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_292, [0, 1]);  add_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_16: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1084, [1, 0])
        sum_dim_int_list_35: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1084, [0], True);  view_1084 = None
        reshape_default_37: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_37);  sum_dim_int_list_35 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_17: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1087, [1, 0])
        sum_dim_int_list_36: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1087, [0], True);  view_1087 = None
        reshape_default_38: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_38);  sum_dim_int_list_36 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_9: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_295, mul_315);  mul_315 = None
        sum_dim_int_list_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_295, [0, 1]);  add_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_20: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_179, 0);  mm_default_179 = None
        reshape_default_39: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_20, _shape_param_39);  unsqueeze_default_20 = _shape_param_39 = None
        permute_default_18: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_39, [3, 4, 2, 0, 1]);  reshape_default_39 = None
        permute_default_19: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_18, [2, 4, 3, 0, 1]);  permute_default_18 = None
        squeeze_dim_40: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_19, 4);  permute_default_19 = None
        squeeze_dim_41: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_40, 3);  squeeze_dim_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_39: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_99, [0, 1], True);  squeeze_99 = None
        reshape_default_40: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_40);  sum_dim_int_list_39 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_40: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_101, [0, 1], True);  squeeze_101 = None
        reshape_default_41: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_41);  sum_dim_int_list_40 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_21: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_177, 0);  mm_default_177 = None
        reshape_default_42: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_21, _shape_param_42);  unsqueeze_default_21 = _shape_param_42 = None
        squeeze_dim_42: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_42, 4);  reshape_default_42 = None
        squeeze_dim_43: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_42, 3);  squeeze_dim_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_22: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_176, 0);  mm_default_176 = None
        reshape_default_43: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_22, _shape_param_43);  unsqueeze_default_22 = _shape_param_43 = None
        squeeze_dim_44: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_43, 4);  reshape_default_43 = None
        squeeze_dim_45: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_44, 3);  squeeze_dim_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_23: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_174, 0);  mm_default_174 = None
        reshape_default_44: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_23, _shape_param_44);  unsqueeze_default_23 = _shape_param_44 = None
        squeeze_dim_46: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_44, 4);  reshape_default_44 = None
        squeeze_dim_47: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_46, 3);  squeeze_dim_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_24: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_172, 0);  mm_default_172 = None
        reshape_default_45: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_24, _shape_param_45);  unsqueeze_default_24 = _shape_param_45 = None
        squeeze_dim_48: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_45, 4);  reshape_default_45 = None
        squeeze_dim_49: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_48, 3);  squeeze_dim_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_10: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_299, mul_308);  mul_308 = None
        sum_dim_int_list_41: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_42: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_20: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1125, [1, 0])
        sum_dim_int_list_43: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1125, [0], True);  view_1125 = None
        reshape_default_46: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_46);  sum_dim_int_list_43 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_21: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1128, [1, 0])
        sum_dim_int_list_44: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1128, [0], True);  view_1128 = None
        reshape_default_47: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_47);  sum_dim_int_list_44 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_11: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_302, mul_299);  mul_299 = None
        sum_dim_int_list_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_25: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_170, 0);  mm_default_170 = None
        reshape_default_48: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_25, _shape_param_48);  unsqueeze_default_25 = _shape_param_48 = None
        permute_default_22: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_48, [3, 4, 2, 0, 1]);  reshape_default_48 = None
        permute_default_23: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_22, [2, 4, 3, 0, 1]);  permute_default_22 = None
        squeeze_dim_50: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_23, 4);  permute_default_23 = None
        squeeze_dim_51: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_50, 3);  squeeze_dim_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_47: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_122, [0, 1], True);  squeeze_122 = None
        reshape_default_49: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_49);  sum_dim_int_list_47 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_48: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_124, [0, 1], True);  squeeze_124 = None
        reshape_default_50: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_50);  sum_dim_int_list_48 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_26: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_168, 0);  mm_default_168 = None
        reshape_default_51: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_26, _shape_param_51);  unsqueeze_default_26 = _shape_param_51 = None
        squeeze_dim_52: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_51, 4);  reshape_default_51 = None
        squeeze_dim_53: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_52, 3);  squeeze_dim_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_27: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_167, 0);  mm_default_167 = None
        reshape_default_52: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_27, _shape_param_52);  unsqueeze_default_27 = _shape_param_52 = None
        squeeze_dim_54: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_52, 4);  reshape_default_52 = None
        squeeze_dim_55: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_54, 3);  squeeze_dim_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_28: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_165, 0);  mm_default_165 = None
        reshape_default_53: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_28, _shape_param_53);  unsqueeze_default_28 = _shape_param_53 = None
        squeeze_dim_56: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_53, 4);  reshape_default_53 = None
        squeeze_dim_57: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_56, 3);  squeeze_dim_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_29: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_163, 0);  mm_default_163 = None
        reshape_default_54: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_29, _shape_param_54);  unsqueeze_default_29 = _shape_param_54 = None
        squeeze_dim_58: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_54, 4);  reshape_default_54 = None
        squeeze_dim_59: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_58, 3);  squeeze_dim_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_12: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_306, mul_292);  mul_292 = None
        sum_dim_int_list_49: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_50: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_306, [0, 1]);  add_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_24: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1166, [1, 0])
        sum_dim_int_list_51: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1166, [0], True);  view_1166 = None
        reshape_default_55: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_55);  sum_dim_int_list_51 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_25: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1169, [1, 0])
        sum_dim_int_list_52: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1169, [0], True);  view_1169 = None
        reshape_default_56: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_56);  sum_dim_int_list_52 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_13: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_309, mul_283);  mul_283 = None
        sum_dim_int_list_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_54: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_309, [0, 1]);  add_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_30: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_161, 0);  mm_default_161 = None
        reshape_default_57: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_30, _shape_param_57);  unsqueeze_default_30 = _shape_param_57 = None
        permute_default_26: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_57, [3, 4, 2, 0, 1]);  reshape_default_57 = None
        permute_default_27: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_26, [2, 4, 3, 0, 1]);  permute_default_26 = None
        squeeze_dim_60: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_27, 4);  permute_default_27 = None
        squeeze_dim_61: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_60, 3);  squeeze_dim_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_55: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_145, [0, 1], True);  squeeze_145 = None
        reshape_default_58: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_58);  sum_dim_int_list_55 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_56: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_147, [0, 1], True);  squeeze_147 = None
        reshape_default_59: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_59);  sum_dim_int_list_56 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_31: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_159, 0);  mm_default_159 = None
        reshape_default_60: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_31, _shape_param_60);  unsqueeze_default_31 = _shape_param_60 = None
        squeeze_dim_62: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_60, 4);  reshape_default_60 = None
        squeeze_dim_63: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_62, 3);  squeeze_dim_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_32: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_158, 0);  mm_default_158 = None
        reshape_default_61: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_32, _shape_param_61);  unsqueeze_default_32 = _shape_param_61 = None
        squeeze_dim_64: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_61, 4);  reshape_default_61 = None
        squeeze_dim_65: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_64, 3);  squeeze_dim_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_33: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_156, 0);  mm_default_156 = None
        reshape_default_62: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_33, _shape_param_62);  unsqueeze_default_33 = _shape_param_62 = None
        squeeze_dim_66: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_62, 4);  reshape_default_62 = None
        squeeze_dim_67: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_66, 3);  squeeze_dim_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_34: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_154, 0);  mm_default_154 = None
        reshape_default_63: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_34, _shape_param_63);  unsqueeze_default_34 = _shape_param_63 = None
        squeeze_dim_68: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_63, 4);  reshape_default_63 = None
        squeeze_dim_69: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_68, 3);  squeeze_dim_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_14: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_313, mul_276);  mul_276 = None
        sum_dim_int_list_57: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_313, [0, 1]);  add_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_28: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1207, [1, 0])
        sum_dim_int_list_59: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1207, [0], True);  view_1207 = None
        reshape_default_64: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_64);  sum_dim_int_list_59 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_29: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        sum_dim_int_list_60: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        reshape_default_65: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_65);  sum_dim_int_list_60 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_15: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_316, mul_267);  mul_267 = None
        sum_dim_int_list_61: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_62: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_316, [0, 1]);  add_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_35: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_152, 0);  mm_default_152 = None
        reshape_default_66: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_35, _shape_param_66);  unsqueeze_default_35 = _shape_param_66 = None
        permute_default_30: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_66, [3, 4, 2, 0, 1]);  reshape_default_66 = None
        permute_default_31: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_30, [2, 4, 3, 0, 1]);  permute_default_30 = None
        squeeze_dim_70: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_31, 4);  permute_default_31 = None
        squeeze_dim_71: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_70, 3);  squeeze_dim_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_63: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_168, [0, 1], True);  squeeze_168 = None
        reshape_default_67: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_67);  sum_dim_int_list_63 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_64: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_170, [0, 1], True);  squeeze_170 = None
        reshape_default_68: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_68);  sum_dim_int_list_64 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_36: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_150, 0);  mm_default_150 = None
        reshape_default_69: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_36, _shape_param_69);  unsqueeze_default_36 = _shape_param_69 = None
        squeeze_dim_72: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_69, 4);  reshape_default_69 = None
        squeeze_dim_73: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_72, 3);  squeeze_dim_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_37: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_149, 0);  mm_default_149 = None
        reshape_default_70: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_37, _shape_param_70);  unsqueeze_default_37 = _shape_param_70 = None
        squeeze_dim_74: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_70, 4);  reshape_default_70 = None
        squeeze_dim_75: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_74, 3);  squeeze_dim_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_38: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_147, 0);  mm_default_147 = None
        reshape_default_71: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_38, _shape_param_71);  unsqueeze_default_38 = _shape_param_71 = None
        squeeze_dim_76: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_71, 4);  reshape_default_71 = None
        squeeze_dim_77: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_76, 3);  squeeze_dim_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_39: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_145, 0);  mm_default_145 = None
        reshape_default_72: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_39, _shape_param_72);  unsqueeze_default_39 = _shape_param_72 = None
        squeeze_dim_78: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_72, 4);  reshape_default_72 = None
        squeeze_dim_79: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_78, 3);  squeeze_dim_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_16: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_320, mul_260);  mul_260 = None
        sum_dim_int_list_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_66: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_320, [0, 1]);  add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_32: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1248, [1, 0])
        sum_dim_int_list_67: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1248, [0], True);  view_1248 = None
        reshape_default_73: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_73);  sum_dim_int_list_67 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_33: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1251, [1, 0])
        sum_dim_int_list_68: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1251, [0], True);  view_1251 = None
        reshape_default_74: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_74);  sum_dim_int_list_68 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_17: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_323, mul_251);  mul_251 = None
        sum_dim_int_list_69: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_70: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_40: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_143, 0);  mm_default_143 = None
        reshape_default_75: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_40, _shape_param_75);  unsqueeze_default_40 = _shape_param_75 = None
        permute_default_34: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_75, [3, 4, 2, 0, 1]);  reshape_default_75 = None
        permute_default_35: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_34, [2, 4, 3, 0, 1]);  permute_default_34 = None
        squeeze_dim_80: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_35, 4);  permute_default_35 = None
        squeeze_dim_81: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_80, 3);  squeeze_dim_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_71: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_191, [0, 1], True);  squeeze_191 = None
        reshape_default_76: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_76);  sum_dim_int_list_71 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_72: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_193, [0, 1], True);  squeeze_193 = None
        reshape_default_77: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_77);  sum_dim_int_list_72 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_41: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_141, 0);  mm_default_141 = None
        reshape_default_78: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_41, _shape_param_78);  unsqueeze_default_41 = _shape_param_78 = None
        squeeze_dim_82: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_78, 4);  reshape_default_78 = None
        squeeze_dim_83: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_82, 3);  squeeze_dim_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_42: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_140, 0);  mm_default_140 = None
        reshape_default_79: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_42, _shape_param_79);  unsqueeze_default_42 = _shape_param_79 = None
        squeeze_dim_84: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_79, 4);  reshape_default_79 = None
        squeeze_dim_85: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_84, 3);  squeeze_dim_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_43: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_138, 0);  mm_default_138 = None
        reshape_default_80: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_43, _shape_param_80);  unsqueeze_default_43 = _shape_param_80 = None
        squeeze_dim_86: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_80, 4);  reshape_default_80 = None
        squeeze_dim_87: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_86, 3);  squeeze_dim_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_44: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_136, 0);  mm_default_136 = None
        reshape_default_81: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_44, _shape_param_81);  unsqueeze_default_44 = _shape_param_81 = None
        squeeze_dim_88: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_81, 4);  reshape_default_81 = None
        squeeze_dim_89: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_88, 3);  squeeze_dim_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_18: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_327, mul_244);  mul_244 = None
        sum_dim_int_list_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_327, [0, 1]);  add_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_36: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1289, [1, 0])
        sum_dim_int_list_75: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1289, [0], True);  view_1289 = None
        reshape_default_82: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_82);  sum_dim_int_list_75 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_37: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1292, [1, 0])
        sum_dim_int_list_76: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1292, [0], True);  view_1292 = None
        reshape_default_83: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_83);  sum_dim_int_list_76 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_19: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_330, mul_235);  mul_235 = None
        sum_dim_int_list_77: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_78: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_330, [0, 1]);  add_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_45: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_134, 0);  mm_default_134 = None
        reshape_default_84: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_45, _shape_param_84);  unsqueeze_default_45 = _shape_param_84 = None
        permute_default_38: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_84, [3, 4, 2, 0, 1]);  reshape_default_84 = None
        permute_default_39: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_38, [2, 4, 3, 0, 1]);  permute_default_38 = None
        squeeze_dim_90: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_39, 4);  permute_default_39 = None
        squeeze_dim_91: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_90, 3);  squeeze_dim_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_79: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_214, [0, 1], True);  squeeze_214 = None
        reshape_default_85: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_85);  sum_dim_int_list_79 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_80: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_216, [0, 1], True);  squeeze_216 = None
        reshape_default_86: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_86);  sum_dim_int_list_80 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_46: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_132, 0);  mm_default_132 = None
        reshape_default_87: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_46, _shape_param_87);  unsqueeze_default_46 = _shape_param_87 = None
        squeeze_dim_92: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_87, 4);  reshape_default_87 = None
        squeeze_dim_93: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_92, 3);  squeeze_dim_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_47: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_131, 0);  mm_default_131 = None
        reshape_default_88: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_47, _shape_param_88);  unsqueeze_default_47 = _shape_param_88 = None
        squeeze_dim_94: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_88, 4);  reshape_default_88 = None
        squeeze_dim_95: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_94, 3);  squeeze_dim_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_48: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_129, 0);  mm_default_129 = None
        reshape_default_89: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_48, _shape_param_89);  unsqueeze_default_48 = _shape_param_89 = None
        squeeze_dim_96: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_89, 4);  reshape_default_89 = None
        squeeze_dim_97: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_96, 3);  squeeze_dim_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_49: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_127, 0);  mm_default_127 = None
        reshape_default_90: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_49, _shape_param_90);  unsqueeze_default_49 = _shape_param_90 = None
        squeeze_dim_98: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_90, 4);  reshape_default_90 = None
        squeeze_dim_99: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_98, 3);  squeeze_dim_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_20: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_334, mul_228);  mul_228 = None
        sum_dim_int_list_81: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_334, [0, 1]);  add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_40: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1330, [1, 0])
        sum_dim_int_list_83: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1330, [0], True);  view_1330 = None
        reshape_default_91: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_91);  sum_dim_int_list_83 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_41: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1333, [1, 0])
        sum_dim_int_list_84: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1333, [0], True);  view_1333 = None
        reshape_default_92: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_92);  sum_dim_int_list_84 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_21: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_337, mul_219);  mul_219 = None
        sum_dim_int_list_85: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_86: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_337, [0, 1]);  add_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_50: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_125, 0);  mm_default_125 = None
        reshape_default_93: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_50, _shape_param_93);  unsqueeze_default_50 = _shape_param_93 = None
        permute_default_42: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_93, [3, 4, 2, 0, 1]);  reshape_default_93 = None
        permute_default_43: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_42, [2, 4, 3, 0, 1]);  permute_default_42 = None
        squeeze_dim_100: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_43, 4);  permute_default_43 = None
        squeeze_dim_101: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_100, 3);  squeeze_dim_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_87: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_237, [0, 1], True);  squeeze_237 = None
        reshape_default_94: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_94);  sum_dim_int_list_87 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_88: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_239, [0, 1], True);  squeeze_239 = None
        reshape_default_95: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_95);  sum_dim_int_list_88 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_51: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_123, 0);  mm_default_123 = None
        reshape_default_96: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_51, _shape_param_96);  unsqueeze_default_51 = _shape_param_96 = None
        squeeze_dim_102: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_96, 4);  reshape_default_96 = None
        squeeze_dim_103: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_102, 3);  squeeze_dim_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_52: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_122, 0);  mm_default_122 = None
        reshape_default_97: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_52, _shape_param_97);  unsqueeze_default_52 = _shape_param_97 = None
        squeeze_dim_104: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_97, 4);  reshape_default_97 = None
        squeeze_dim_105: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_104, 3);  squeeze_dim_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_53: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_120, 0);  mm_default_120 = None
        reshape_default_98: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_53, _shape_param_98);  unsqueeze_default_53 = _shape_param_98 = None
        squeeze_dim_106: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_98, 4);  reshape_default_98 = None
        squeeze_dim_107: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_106, 3);  squeeze_dim_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_54: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_118, 0);  mm_default_118 = None
        reshape_default_99: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_54, _shape_param_99);  unsqueeze_default_54 = _shape_param_99 = None
        squeeze_dim_108: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_99, 4);  reshape_default_99 = None
        squeeze_dim_109: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_108, 3);  squeeze_dim_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_22: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_341, mul_212);  mul_212 = None
        sum_dim_int_list_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_90: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 1]);  add_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_44: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1371, [1, 0])
        sum_dim_int_list_91: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1371, [0], True);  view_1371 = None
        reshape_default_100: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_100);  sum_dim_int_list_91 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_45: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1374, [1, 0])
        sum_dim_int_list_92: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1374, [0], True);  view_1374 = None
        reshape_default_101: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_101);  sum_dim_int_list_92 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_23: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_344, mul_203);  mul_203 = None
        sum_dim_int_list_93: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_94: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 1]);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_55: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_116, 0);  mm_default_116 = None
        reshape_default_102: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_55, _shape_param_102);  unsqueeze_default_55 = _shape_param_102 = None
        permute_default_46: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_102, [3, 4, 2, 0, 1]);  reshape_default_102 = None
        permute_default_47: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_46, [2, 4, 3, 0, 1]);  permute_default_46 = None
        squeeze_dim_110: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_47, 4);  permute_default_47 = None
        squeeze_dim_111: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_110, 3);  squeeze_dim_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_95: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_260, [0, 1], True);  squeeze_260 = None
        reshape_default_103: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_103);  sum_dim_int_list_95 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_96: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_262, [0, 1], True);  squeeze_262 = None
        reshape_default_104: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_104);  sum_dim_int_list_96 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_56: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_114, 0);  mm_default_114 = None
        reshape_default_105: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_56, _shape_param_105);  unsqueeze_default_56 = _shape_param_105 = None
        squeeze_dim_112: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_105, 4);  reshape_default_105 = None
        squeeze_dim_113: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_112, 3);  squeeze_dim_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_57: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_113, 0);  mm_default_113 = None
        reshape_default_106: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_57, _shape_param_106);  unsqueeze_default_57 = _shape_param_106 = None
        squeeze_dim_114: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_106, 4);  reshape_default_106 = None
        squeeze_dim_115: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_114, 3);  squeeze_dim_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_58: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_111, 0);  mm_default_111 = None
        reshape_default_107: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_58, _shape_param_107);  unsqueeze_default_58 = _shape_param_107 = None
        squeeze_dim_116: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_107, 4);  reshape_default_107 = None
        squeeze_dim_117: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_116, 3);  squeeze_dim_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_59: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_109, 0);  mm_default_109 = None
        reshape_default_108: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_59, _shape_param_108);  unsqueeze_default_59 = _shape_param_108 = None
        squeeze_dim_118: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_108, 4);  reshape_default_108 = None
        squeeze_dim_119: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_118, 3);  squeeze_dim_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_24: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_348, mul_196);  mul_196 = None
        sum_dim_int_list_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_348, [0, 1]);  add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_48: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1412, [1, 0])
        sum_dim_int_list_99: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1412, [0], True);  view_1412 = None
        reshape_default_109: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_109);  sum_dim_int_list_99 = _shape_param_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_49: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1415, [1, 0])
        sum_dim_int_list_100: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1415, [0], True);  view_1415 = None
        reshape_default_110: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_110);  sum_dim_int_list_100 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_25: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_351, mul_187);  mul_187 = None
        sum_dim_int_list_101: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1]);  mul_tensor_25 = None
        sum_dim_int_list_102: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_351, [0, 1]);  add_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_60: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_107, 0);  mm_default_107 = None
        reshape_default_111: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_60, _shape_param_111);  unsqueeze_default_60 = _shape_param_111 = None
        permute_default_50: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_111, [3, 4, 2, 0, 1]);  reshape_default_111 = None
        permute_default_51: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_50, [2, 4, 3, 0, 1]);  permute_default_50 = None
        squeeze_dim_120: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_51, 4);  permute_default_51 = None
        squeeze_dim_121: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_120, 3);  squeeze_dim_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_103: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_283, [0, 1], True);  squeeze_283 = None
        reshape_default_112: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_112);  sum_dim_int_list_103 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_104: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_285, [0, 1], True);  squeeze_285 = None
        reshape_default_113: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_113);  sum_dim_int_list_104 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_61: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_105, 0);  mm_default_105 = None
        reshape_default_114: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_61, _shape_param_114);  unsqueeze_default_61 = _shape_param_114 = None
        squeeze_dim_122: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_114, 4);  reshape_default_114 = None
        squeeze_dim_123: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_122, 3);  squeeze_dim_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_62: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_104, 0);  mm_default_104 = None
        reshape_default_115: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_62, _shape_param_115);  unsqueeze_default_62 = _shape_param_115 = None
        squeeze_dim_124: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_115, 4);  reshape_default_115 = None
        squeeze_dim_125: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_124, 3);  squeeze_dim_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_63: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_102, 0);  mm_default_102 = None
        reshape_default_116: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_63, _shape_param_116);  unsqueeze_default_63 = _shape_param_116 = None
        squeeze_dim_126: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_116, 4);  reshape_default_116 = None
        squeeze_dim_127: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_126, 3);  squeeze_dim_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_64: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_100, 0);  mm_default_100 = None
        reshape_default_117: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_64, _shape_param_117);  unsqueeze_default_64 = _shape_param_117 = None
        squeeze_dim_128: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_117, 4);  reshape_default_117 = None
        squeeze_dim_129: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_128, 3);  squeeze_dim_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_26: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_355, mul_180);  mul_180 = None
        sum_dim_int_list_105: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_106: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_355, [0, 1]);  add_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_52: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1453, [1, 0])
        sum_dim_int_list_107: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1453, [0], True);  view_1453 = None
        reshape_default_118: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_118);  sum_dim_int_list_107 = _shape_param_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_53: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1456, [1, 0])
        sum_dim_int_list_108: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1456, [0], True);  view_1456 = None
        reshape_default_119: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_119);  sum_dim_int_list_108 = _shape_param_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_27: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_358, mul_171);  mul_171 = None
        sum_dim_int_list_109: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1]);  mul_tensor_27 = None
        sum_dim_int_list_110: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_358, [0, 1]);  add_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_65: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_98, 0);  mm_default_98 = None
        reshape_default_120: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_65, _shape_param_120);  unsqueeze_default_65 = _shape_param_120 = None
        permute_default_54: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_120, [3, 4, 2, 0, 1]);  reshape_default_120 = None
        permute_default_55: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_54, [2, 4, 3, 0, 1]);  permute_default_54 = None
        squeeze_dim_130: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_55, 4);  permute_default_55 = None
        squeeze_dim_131: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_130, 3);  squeeze_dim_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_111: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_306, [0, 1], True);  squeeze_306 = None
        reshape_default_121: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_121);  sum_dim_int_list_111 = _shape_param_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_112: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_308, [0, 1], True);  squeeze_308 = None
        reshape_default_122: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_122);  sum_dim_int_list_112 = _shape_param_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_66: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_96, 0);  mm_default_96 = None
        reshape_default_123: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_66, _shape_param_123);  unsqueeze_default_66 = _shape_param_123 = None
        squeeze_dim_132: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_123, 4);  reshape_default_123 = None
        squeeze_dim_133: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_132, 3);  squeeze_dim_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_67: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_95, 0);  mm_default_95 = None
        reshape_default_124: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_67, _shape_param_124);  unsqueeze_default_67 = _shape_param_124 = None
        squeeze_dim_134: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_124, 4);  reshape_default_124 = None
        squeeze_dim_135: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_134, 3);  squeeze_dim_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_68: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_93, 0);  mm_default_93 = None
        reshape_default_125: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_68, _shape_param_125);  unsqueeze_default_68 = _shape_param_125 = None
        squeeze_dim_136: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_125, 4);  reshape_default_125 = None
        squeeze_dim_137: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_136, 3);  squeeze_dim_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_69: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_91, 0);  mm_default_91 = None
        reshape_default_126: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_69, _shape_param_126);  unsqueeze_default_69 = _shape_param_126 = None
        squeeze_dim_138: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_126, 4);  reshape_default_126 = None
        squeeze_dim_139: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_138, 3);  squeeze_dim_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_28: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_362, mul_164);  mul_164 = None
        sum_dim_int_list_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_114: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_362, [0, 1]);  add_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_56: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1494, [1, 0])
        sum_dim_int_list_115: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1494, [0], True);  view_1494 = None
        reshape_default_127: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_127);  sum_dim_int_list_115 = _shape_param_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_57: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1497, [1, 0])
        sum_dim_int_list_116: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1497, [0], True);  view_1497 = None
        reshape_default_128: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_128);  sum_dim_int_list_116 = _shape_param_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_29: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_365, mul_155);  mul_155 = None
        sum_dim_int_list_117: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1]);  mul_tensor_29 = None
        sum_dim_int_list_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_365, [0, 1]);  add_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_70: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_89, 0);  mm_default_89 = None
        reshape_default_129: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_70, _shape_param_129);  unsqueeze_default_70 = _shape_param_129 = None
        permute_default_58: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_129, [3, 4, 2, 0, 1]);  reshape_default_129 = None
        permute_default_59: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_58, [2, 4, 3, 0, 1]);  permute_default_58 = None
        squeeze_dim_140: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_59, 4);  permute_default_59 = None
        squeeze_dim_141: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_140, 3);  squeeze_dim_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_119: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_329, [0, 1], True);  squeeze_329 = None
        reshape_default_130: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_130);  sum_dim_int_list_119 = _shape_param_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_120: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_331, [0, 1], True);  squeeze_331 = None
        reshape_default_131: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_131);  sum_dim_int_list_120 = _shape_param_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_71: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_87, 0);  mm_default_87 = None
        reshape_default_132: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_71, _shape_param_132);  unsqueeze_default_71 = _shape_param_132 = None
        squeeze_dim_142: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_132, 4);  reshape_default_132 = None
        squeeze_dim_143: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_142, 3);  squeeze_dim_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_72: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_86, 0);  mm_default_86 = None
        reshape_default_133: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_72, _shape_param_133);  unsqueeze_default_72 = _shape_param_133 = None
        squeeze_dim_144: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_133, 4);  reshape_default_133 = None
        squeeze_dim_145: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_144, 3);  squeeze_dim_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_73: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_84, 0);  mm_default_84 = None
        reshape_default_134: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_73, _shape_param_134);  unsqueeze_default_73 = _shape_param_134 = None
        squeeze_dim_146: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_134, 4);  reshape_default_134 = None
        squeeze_dim_147: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_146, 3);  squeeze_dim_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_74: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_82, 0);  mm_default_82 = None
        reshape_default_135: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_74, _shape_param_135);  unsqueeze_default_74 = _shape_param_135 = None
        squeeze_dim_148: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_135, 4);  reshape_default_135 = None
        squeeze_dim_149: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_148, 3);  squeeze_dim_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_30: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_369, mul_148);  mul_148 = None
        sum_dim_int_list_121: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_122: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_369, [0, 1]);  add_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_60: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1535, [1, 0])
        sum_dim_int_list_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1535, [0], True);  view_1535 = None
        reshape_default_136: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_136);  sum_dim_int_list_123 = _shape_param_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_61: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1538, [1, 0])
        sum_dim_int_list_124: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1538, [0], True);  view_1538 = None
        reshape_default_137: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_124, _shape_param_137);  sum_dim_int_list_124 = _shape_param_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_31: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_372, mul_139);  mul_139 = None
        sum_dim_int_list_125: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1]);  mul_tensor_31 = None
        sum_dim_int_list_126: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_372, [0, 1]);  add_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_75: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_80, 0);  mm_default_80 = None
        reshape_default_138: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_75, _shape_param_138);  unsqueeze_default_75 = _shape_param_138 = None
        permute_default_62: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_138, [3, 4, 2, 0, 1]);  reshape_default_138 = None
        permute_default_63: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_62, [2, 4, 3, 0, 1]);  permute_default_62 = None
        squeeze_dim_150: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_63, 4);  permute_default_63 = None
        squeeze_dim_151: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_150, 3);  squeeze_dim_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_127: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_352, [0, 1], True);  squeeze_352 = None
        reshape_default_139: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_127, _shape_param_139);  sum_dim_int_list_127 = _shape_param_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_128: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_354, [0, 1], True);  squeeze_354 = None
        reshape_default_140: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_128, _shape_param_140);  sum_dim_int_list_128 = _shape_param_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_76: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_78, 0);  mm_default_78 = None
        reshape_default_141: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_76, _shape_param_141);  unsqueeze_default_76 = _shape_param_141 = None
        squeeze_dim_152: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_141, 4);  reshape_default_141 = None
        squeeze_dim_153: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_152, 3);  squeeze_dim_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_77: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_77, 0);  mm_default_77 = None
        reshape_default_142: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_77, _shape_param_142);  unsqueeze_default_77 = _shape_param_142 = None
        squeeze_dim_154: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_142, 4);  reshape_default_142 = None
        squeeze_dim_155: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_154, 3);  squeeze_dim_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_78: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_75, 0);  mm_default_75 = None
        reshape_default_143: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_78, _shape_param_143);  unsqueeze_default_78 = _shape_param_143 = None
        squeeze_dim_156: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_143, 4);  reshape_default_143 = None
        squeeze_dim_157: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_156, 3);  squeeze_dim_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_79: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_73, 0);  mm_default_73 = None
        reshape_default_144: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_79, _shape_param_144);  unsqueeze_default_79 = _shape_param_144 = None
        squeeze_dim_158: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_144, 4);  reshape_default_144 = None
        squeeze_dim_159: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_158, 3);  squeeze_dim_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_32: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_376, mul_132);  mul_132 = None
        sum_dim_int_list_129: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_130: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_376, [0, 1]);  add_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_64: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1576, [1, 0])
        sum_dim_int_list_131: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1576, [0], True);  view_1576 = None
        reshape_default_145: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_131, _shape_param_145);  sum_dim_int_list_131 = _shape_param_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_65: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1579, [1, 0])
        sum_dim_int_list_132: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1579, [0], True);  view_1579 = None
        reshape_default_146: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_146);  sum_dim_int_list_132 = _shape_param_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_33: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_379, mul_123);  mul_123 = None
        sum_dim_int_list_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1]);  mul_tensor_33 = None
        sum_dim_int_list_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_379, [0, 1]);  add_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_80: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_71, 0);  mm_default_71 = None
        reshape_default_147: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_80, _shape_param_147);  unsqueeze_default_80 = _shape_param_147 = None
        permute_default_66: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_147, [3, 4, 2, 0, 1]);  reshape_default_147 = None
        permute_default_67: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_66, [2, 4, 3, 0, 1]);  permute_default_66 = None
        squeeze_dim_160: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_67, 4);  permute_default_67 = None
        squeeze_dim_161: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_160, 3);  squeeze_dim_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_135: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_375, [0, 1], True);  squeeze_375 = None
        reshape_default_148: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_135, _shape_param_148);  sum_dim_int_list_135 = _shape_param_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_136: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_377, [0, 1], True);  squeeze_377 = None
        reshape_default_149: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_136, _shape_param_149);  sum_dim_int_list_136 = _shape_param_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_81: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_69, 0);  mm_default_69 = None
        reshape_default_150: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_81, _shape_param_150);  unsqueeze_default_81 = _shape_param_150 = None
        squeeze_dim_162: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_150, 4);  reshape_default_150 = None
        squeeze_dim_163: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_162, 3);  squeeze_dim_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_82: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_68, 0);  mm_default_68 = None
        reshape_default_151: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_82, _shape_param_151);  unsqueeze_default_82 = _shape_param_151 = None
        squeeze_dim_164: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_151, 4);  reshape_default_151 = None
        squeeze_dim_165: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_164, 3);  squeeze_dim_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_83: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_66, 0);  mm_default_66 = None
        reshape_default_152: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_83, _shape_param_152);  unsqueeze_default_83 = _shape_param_152 = None
        squeeze_dim_166: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_152, 4);  reshape_default_152 = None
        squeeze_dim_167: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_166, 3);  squeeze_dim_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_84: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_64, 0);  mm_default_64 = None
        reshape_default_153: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_84, _shape_param_153);  unsqueeze_default_84 = _shape_param_153 = None
        squeeze_dim_168: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_153, 4);  reshape_default_153 = None
        squeeze_dim_169: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_168, 3);  squeeze_dim_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_34: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_383, mul_116);  mul_116 = None
        sum_dim_int_list_137: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_138: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_383, [0, 1]);  add_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_68: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1617, [1, 0])
        sum_dim_int_list_139: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1617, [0], True);  view_1617 = None
        reshape_default_154: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_139, _shape_param_154);  sum_dim_int_list_139 = _shape_param_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_69: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1620, [1, 0])
        sum_dim_int_list_140: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1620, [0], True);  view_1620 = None
        reshape_default_155: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_140, _shape_param_155);  sum_dim_int_list_140 = _shape_param_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_35: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_386, mul_107);  mul_107 = None
        sum_dim_int_list_141: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1]);  mul_tensor_35 = None
        sum_dim_int_list_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_386, [0, 1]);  add_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_85: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_62, 0);  mm_default_62 = None
        reshape_default_156: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_85, _shape_param_156);  unsqueeze_default_85 = _shape_param_156 = None
        permute_default_70: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_156, [3, 4, 2, 0, 1]);  reshape_default_156 = None
        permute_default_71: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_70, [2, 4, 3, 0, 1]);  permute_default_70 = None
        squeeze_dim_170: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_71, 4);  permute_default_71 = None
        squeeze_dim_171: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_170, 3);  squeeze_dim_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_143: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_398, [0, 1], True);  squeeze_398 = None
        reshape_default_157: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_143, _shape_param_157);  sum_dim_int_list_143 = _shape_param_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_144: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_400, [0, 1], True);  squeeze_400 = None
        reshape_default_158: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_144, _shape_param_158);  sum_dim_int_list_144 = _shape_param_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_86: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_60, 0);  mm_default_60 = None
        reshape_default_159: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_86, _shape_param_159);  unsqueeze_default_86 = _shape_param_159 = None
        squeeze_dim_172: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_159, 4);  reshape_default_159 = None
        squeeze_dim_173: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_172, 3);  squeeze_dim_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_87: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_59, 0);  mm_default_59 = None
        reshape_default_160: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_87, _shape_param_160);  unsqueeze_default_87 = _shape_param_160 = None
        squeeze_dim_174: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_160, 4);  reshape_default_160 = None
        squeeze_dim_175: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_174, 3);  squeeze_dim_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_88: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_57, 0);  mm_default_57 = None
        reshape_default_161: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_88, _shape_param_161);  unsqueeze_default_88 = _shape_param_161 = None
        squeeze_dim_176: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_161, 4);  reshape_default_161 = None
        squeeze_dim_177: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_176, 3);  squeeze_dim_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_89: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_55, 0);  mm_default_55 = None
        reshape_default_162: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_89, _shape_param_162);  unsqueeze_default_89 = _shape_param_162 = None
        squeeze_dim_178: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_162, 4);  reshape_default_162 = None
        squeeze_dim_179: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_178, 3);  squeeze_dim_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_36: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_390, mul_100);  mul_100 = None
        sum_dim_int_list_145: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_146: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_390, [0, 1]);  add_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_72: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1658, [1, 0])
        sum_dim_int_list_147: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1658, [0], True);  view_1658 = None
        reshape_default_163: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_147, _shape_param_163);  sum_dim_int_list_147 = _shape_param_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_73: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1661, [1, 0])
        sum_dim_int_list_148: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1661, [0], True);  view_1661 = None
        reshape_default_164: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_148, _shape_param_164);  sum_dim_int_list_148 = _shape_param_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_37: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_393, mul_91);  mul_91 = None
        sum_dim_int_list_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1]);  mul_tensor_37 = None
        sum_dim_int_list_150: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_393, [0, 1]);  add_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_90: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_53, 0);  mm_default_53 = None
        reshape_default_165: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_90, _shape_param_165);  unsqueeze_default_90 = _shape_param_165 = None
        permute_default_74: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_165, [3, 4, 2, 0, 1]);  reshape_default_165 = None
        permute_default_75: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_74, [2, 4, 3, 0, 1]);  permute_default_74 = None
        squeeze_dim_180: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_75, 4);  permute_default_75 = None
        squeeze_dim_181: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_180, 3);  squeeze_dim_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_151: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_421, [0, 1], True);  squeeze_421 = None
        reshape_default_166: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_151, _shape_param_166);  sum_dim_int_list_151 = _shape_param_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_152: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_423, [0, 1], True);  squeeze_423 = None
        reshape_default_167: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_152, _shape_param_167);  sum_dim_int_list_152 = _shape_param_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_91: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_51, 0);  mm_default_51 = None
        reshape_default_168: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_91, _shape_param_168);  unsqueeze_default_91 = _shape_param_168 = None
        squeeze_dim_182: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_168, 4);  reshape_default_168 = None
        squeeze_dim_183: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_182, 3);  squeeze_dim_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_92: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_50, 0);  mm_default_50 = None
        reshape_default_169: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_92, _shape_param_169);  unsqueeze_default_92 = _shape_param_169 = None
        squeeze_dim_184: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_169, 4);  reshape_default_169 = None
        squeeze_dim_185: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_184, 3);  squeeze_dim_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_93: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_48, 0);  mm_default_48 = None
        reshape_default_170: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_93, _shape_param_170);  unsqueeze_default_93 = _shape_param_170 = None
        squeeze_dim_186: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_170, 4);  reshape_default_170 = None
        squeeze_dim_187: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_186, 3);  squeeze_dim_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_94: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_46, 0);  mm_default_46 = None
        reshape_default_171: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_94, _shape_param_171);  unsqueeze_default_94 = _shape_param_171 = None
        squeeze_dim_188: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_171, 4);  reshape_default_171 = None
        squeeze_dim_189: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_188, 3);  squeeze_dim_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_38: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_397, mul_84);  mul_84 = None
        sum_dim_int_list_153: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_154: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_397, [0, 1]);  add_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_76: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1699, [1, 0])
        sum_dim_int_list_155: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1699, [0], True);  view_1699 = None
        reshape_default_172: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_155, _shape_param_172);  sum_dim_int_list_155 = _shape_param_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_77: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1702, [1, 0])
        sum_dim_int_list_156: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1702, [0], True);  view_1702 = None
        reshape_default_173: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_156, _shape_param_173);  sum_dim_int_list_156 = _shape_param_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_39: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_400, mul_75);  mul_75 = None
        sum_dim_int_list_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1]);  mul_tensor_39 = None
        sum_dim_int_list_158: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_400, [0, 1]);  add_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_95: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_44, 0);  mm_default_44 = None
        reshape_default_174: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_95, _shape_param_174);  unsqueeze_default_95 = _shape_param_174 = None
        permute_default_78: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_174, [3, 4, 2, 0, 1]);  reshape_default_174 = None
        permute_default_79: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_78, [2, 4, 3, 0, 1]);  permute_default_78 = None
        squeeze_dim_190: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_79, 4);  permute_default_79 = None
        squeeze_dim_191: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_190, 3);  squeeze_dim_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_159: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_444, [0, 1], True);  squeeze_444 = None
        reshape_default_175: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_159, _shape_param_175);  sum_dim_int_list_159 = _shape_param_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_160: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_446, [0, 1], True);  squeeze_446 = None
        reshape_default_176: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_160, _shape_param_176);  sum_dim_int_list_160 = _shape_param_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_96: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_42, 0);  mm_default_42 = None
        reshape_default_177: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_96, _shape_param_177);  unsqueeze_default_96 = _shape_param_177 = None
        squeeze_dim_192: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_177, 4);  reshape_default_177 = None
        squeeze_dim_193: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_192, 3);  squeeze_dim_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_97: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_41, 0);  mm_default_41 = None
        reshape_default_178: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_97, _shape_param_178);  unsqueeze_default_97 = _shape_param_178 = None
        squeeze_dim_194: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_178, 4);  reshape_default_178 = None
        squeeze_dim_195: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_194, 3);  squeeze_dim_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_98: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_39, 0);  mm_default_39 = None
        reshape_default_179: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_98, _shape_param_179);  unsqueeze_default_98 = _shape_param_179 = None
        squeeze_dim_196: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_179, 4);  reshape_default_179 = None
        squeeze_dim_197: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_196, 3);  squeeze_dim_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_99: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_37, 0);  mm_default_37 = None
        reshape_default_180: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_99, _shape_param_180);  unsqueeze_default_99 = _shape_param_180 = None
        squeeze_dim_198: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_180, 4);  reshape_default_180 = None
        squeeze_dim_199: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_198, 3);  squeeze_dim_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_40: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_404, mul_68);  mul_68 = None
        sum_dim_int_list_161: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_162: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_404, [0, 1]);  add_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_80: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1740, [1, 0])
        sum_dim_int_list_163: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1740, [0], True);  view_1740 = None
        reshape_default_181: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_163, _shape_param_181);  sum_dim_int_list_163 = _shape_param_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_81: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1743, [1, 0])
        sum_dim_int_list_164: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1743, [0], True);  view_1743 = None
        reshape_default_182: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_164, _shape_param_182);  sum_dim_int_list_164 = _shape_param_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_41: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_407, mul_59);  mul_59 = None
        sum_dim_int_list_165: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1]);  mul_tensor_41 = None
        sum_dim_int_list_166: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_407, [0, 1]);  add_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_100: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_35, 0);  mm_default_35 = None
        reshape_default_183: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_100, _shape_param_183);  unsqueeze_default_100 = _shape_param_183 = None
        permute_default_82: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_183, [3, 4, 2, 0, 1]);  reshape_default_183 = None
        permute_default_83: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_82, [2, 4, 3, 0, 1]);  permute_default_82 = None
        squeeze_dim_200: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_83, 4);  permute_default_83 = None
        squeeze_dim_201: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_200, 3);  squeeze_dim_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_167: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_467, [0, 1], True);  squeeze_467 = None
        reshape_default_184: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_167, _shape_param_184);  sum_dim_int_list_167 = _shape_param_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_168: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_469, [0, 1], True);  squeeze_469 = None
        reshape_default_185: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_168, _shape_param_185);  sum_dim_int_list_168 = _shape_param_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_101: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_33, 0);  mm_default_33 = None
        reshape_default_186: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_101, _shape_param_186);  unsqueeze_default_101 = _shape_param_186 = None
        squeeze_dim_202: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_186, 4);  reshape_default_186 = None
        squeeze_dim_203: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_202, 3);  squeeze_dim_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_102: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_32, 0);  mm_default_32 = None
        reshape_default_187: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_102, _shape_param_187);  unsqueeze_default_102 = _shape_param_187 = None
        squeeze_dim_204: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_187, 4);  reshape_default_187 = None
        squeeze_dim_205: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_204, 3);  squeeze_dim_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_103: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_30, 0);  mm_default_30 = None
        reshape_default_188: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_103, _shape_param_188);  unsqueeze_default_103 = _shape_param_188 = None
        squeeze_dim_206: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_188, 4);  reshape_default_188 = None
        squeeze_dim_207: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_206, 3);  squeeze_dim_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_104: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_28, 0);  mm_default_28 = None
        reshape_default_189: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_104, _shape_param_189);  unsqueeze_default_104 = _shape_param_189 = None
        squeeze_dim_208: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_189, 4);  reshape_default_189 = None
        squeeze_dim_209: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_208, 3);  squeeze_dim_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_42: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_411, mul_52);  mul_52 = None
        sum_dim_int_list_169: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_170: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_411, [0, 1]);  add_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_84: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1781, [1, 0])
        sum_dim_int_list_171: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1781, [0], True);  view_1781 = None
        reshape_default_190: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_171, _shape_param_190);  sum_dim_int_list_171 = _shape_param_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_85: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1784, [1, 0])
        sum_dim_int_list_172: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1784, [0], True);  view_1784 = None
        reshape_default_191: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_172, _shape_param_191);  sum_dim_int_list_172 = _shape_param_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_43: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_414, mul_43);  mul_43 = None
        sum_dim_int_list_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1]);  mul_tensor_43 = None
        sum_dim_int_list_174: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_414, [0, 1]);  add_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_105: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_26, 0);  mm_default_26 = None
        reshape_default_192: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_105, _shape_param_192);  unsqueeze_default_105 = _shape_param_192 = None
        permute_default_86: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_192, [3, 4, 2, 0, 1]);  reshape_default_192 = None
        permute_default_87: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_86, [2, 4, 3, 0, 1]);  permute_default_86 = None
        squeeze_dim_210: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_87, 4);  permute_default_87 = None
        squeeze_dim_211: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_210, 3);  squeeze_dim_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_175: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_490, [0, 1], True);  squeeze_490 = None
        reshape_default_193: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_175, _shape_param_193);  sum_dim_int_list_175 = _shape_param_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_176: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_492, [0, 1], True);  squeeze_492 = None
        reshape_default_194: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_176, _shape_param_194);  sum_dim_int_list_176 = _shape_param_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_106: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_24, 0);  mm_default_24 = None
        reshape_default_195: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_106, _shape_param_195);  unsqueeze_default_106 = _shape_param_195 = None
        squeeze_dim_212: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_195, 4);  reshape_default_195 = None
        squeeze_dim_213: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_212, 3);  squeeze_dim_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_107: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_23, 0);  mm_default_23 = None
        reshape_default_196: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_107, _shape_param_196);  unsqueeze_default_107 = _shape_param_196 = None
        squeeze_dim_214: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_196, 4);  reshape_default_196 = None
        squeeze_dim_215: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_214, 3);  squeeze_dim_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_108: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_21, 0);  mm_default_21 = None
        reshape_default_197: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_108, _shape_param_197);  unsqueeze_default_108 = _shape_param_197 = None
        squeeze_dim_216: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_197, 4);  reshape_default_197 = None
        squeeze_dim_217: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_216, 3);  squeeze_dim_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_109: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_19, 0);  mm_default_19 = None
        reshape_default_198: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_109, _shape_param_198);  unsqueeze_default_109 = _shape_param_198 = None
        squeeze_dim_218: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_198, 4);  reshape_default_198 = None
        squeeze_dim_219: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_218, 3);  squeeze_dim_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_44: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_418, mul_36);  mul_36 = None
        sum_dim_int_list_177: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_418, [0, 1]);  add_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_88: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1822, [1, 0])
        sum_dim_int_list_179: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1822, [0], True);  view_1822 = None
        reshape_default_199: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_179, _shape_param_199);  sum_dim_int_list_179 = _shape_param_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_89: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1825, [1, 0])
        sum_dim_int_list_180: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1825, [0], True);  view_1825 = None
        reshape_default_200: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_180, _shape_param_200);  sum_dim_int_list_180 = _shape_param_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_45: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_421, mul_27);  mul_27 = None
        sum_dim_int_list_181: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1]);  mul_tensor_45 = None
        sum_dim_int_list_182: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_421, [0, 1]);  add_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_110: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_17, 0);  mm_default_17 = None
        reshape_default_201: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_110, _shape_param_201);  unsqueeze_default_110 = _shape_param_201 = None
        permute_default_90: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_201, [3, 4, 2, 0, 1]);  reshape_default_201 = None
        permute_default_91: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_90, [2, 4, 3, 0, 1]);  permute_default_90 = None
        squeeze_dim_220: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_91, 4);  permute_default_91 = None
        squeeze_dim_221: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_220, 3);  squeeze_dim_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_183: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_513, [0, 1], True);  squeeze_513 = None
        reshape_default_202: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_183, _shape_param_202);  sum_dim_int_list_183 = _shape_param_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_184: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_515, [0, 1], True);  squeeze_515 = None
        reshape_default_203: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_184, _shape_param_203);  sum_dim_int_list_184 = _shape_param_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_111: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_15, 0);  mm_default_15 = None
        reshape_default_204: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_111, _shape_param_204);  unsqueeze_default_111 = _shape_param_204 = None
        squeeze_dim_222: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_204, 4);  reshape_default_204 = None
        squeeze_dim_223: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_222, 3);  squeeze_dim_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_112: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_14, 0);  mm_default_14 = None
        reshape_default_205: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_112, _shape_param_205);  unsqueeze_default_112 = _shape_param_205 = None
        squeeze_dim_224: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_205, 4);  reshape_default_205 = None
        squeeze_dim_225: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_224, 3);  squeeze_dim_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_113: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_12, 0);  mm_default_12 = None
        reshape_default_206: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_113, _shape_param_206);  unsqueeze_default_113 = _shape_param_206 = None
        squeeze_dim_226: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_206, 4);  reshape_default_206 = None
        squeeze_dim_227: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_226, 3);  squeeze_dim_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_114: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_10, 0);  mm_default_10 = None
        reshape_default_207: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_114, _shape_param_207);  unsqueeze_default_114 = _shape_param_207 = None
        squeeze_dim_228: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_207, 4);  reshape_default_207 = None
        squeeze_dim_229: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_228, 3);  squeeze_dim_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_46: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_425, mul_20);  mul_20 = None
        sum_dim_int_list_185: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_186: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_425, [0, 1]);  add_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_default_92: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1863, [1, 0])
        sum_dim_int_list_187: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1863, [0], True);  view_1863 = None
        reshape_default_208: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_187, _shape_param_208);  sum_dim_int_list_187 = _shape_param_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_default_93: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1866, [1, 0])
        sum_dim_int_list_188: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1866, [0], True);  view_1866 = None
        reshape_default_209: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_188, _shape_param_209);  sum_dim_int_list_188 = _shape_param_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        mul_tensor_47: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_428, mul_11);  mul_11 = None
        sum_dim_int_list_189: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1]);  mul_tensor_47 = None
        sum_dim_int_list_190: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_428, [0, 1]);  add_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default_115: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_8, 0);  mm_default_8 = None
        reshape_default_210: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_115, _shape_param_210);  unsqueeze_default_115 = _shape_param_210 = None
        permute_default_94: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(reshape_default_210, [3, 4, 2, 0, 1]);  reshape_default_210 = None
        permute_default_95: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_94, [2, 4, 3, 0, 1]);  permute_default_94 = None
        squeeze_dim_230: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_default_95, 4);  permute_default_95 = None
        squeeze_dim_231: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_230, 3);  squeeze_dim_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        sum_dim_int_list_191: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_536, [0, 1], True);  squeeze_536 = None
        reshape_default_211: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_191, _shape_param_211);  sum_dim_int_list_191 = _shape_param_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        sum_dim_int_list_192: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_538, [0, 1], True);  squeeze_538 = None
        reshape_default_212: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_192, _shape_param_212);  sum_dim_int_list_192 = _shape_param_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_116: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_6, 0);  mm_default_6 = None
        reshape_default_213: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_116, _shape_param_213);  unsqueeze_default_116 = _shape_param_213 = None
        squeeze_dim_232: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_213, 4);  reshape_default_213 = None
        squeeze_dim_233: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_232, 3);  squeeze_dim_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_117: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_5, 0);  mm_default_5 = None
        unsqueeze_default_118: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None
        reshape_default_214: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_117, _shape_param_214);  unsqueeze_default_117 = _shape_param_214 = None
        reshape_default_215: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_118, _shape_param_215);  unsqueeze_default_118 = _shape_param_215 = None
        squeeze_dim_234: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_214, 4);  reshape_default_214 = None
        squeeze_dim_235: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_234, 3);  squeeze_dim_234 = None
        squeeze_dim_236: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default_215, 4);  reshape_default_215 = None
        squeeze_dim_237: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim_236, 3);  squeeze_dim_236 = None
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_1132, squeeze_dim_237);  mul_1132 = squeeze_dim_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_119: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_3, 0);  mm_default_3 = None
        unsqueeze_default_120: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_2, 0);  mm_default_2 = None
        reshape_default_216: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_119, _shape_param_216);  unsqueeze_default_119 = _shape_param_216 = None
        reshape_default_217: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_120, _shape_param_217);  unsqueeze_default_120 = _shape_param_217 = None
        squeeze_dim_238: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_216, 4);  reshape_default_216 = None
        squeeze_dim_239: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_238, 3);  squeeze_dim_238 = None
        squeeze_dim_240: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default_217, 4);  reshape_default_217 = None
        squeeze_dim_241: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim_240, 3);  squeeze_dim_240 = None
        add_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_tensor, squeeze_dim_241);  add_tensor = squeeze_dim_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_121: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        unsqueeze_default_122: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        reshape_default_218: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_121, _shape_param_218);  unsqueeze_default_121 = _shape_param_218 = None
        reshape_default_219: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_122, _shape_param_219);  unsqueeze_default_122 = _shape_param_219 = None
        squeeze_dim_242: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default_218, 4);  reshape_default_218 = None
        squeeze_dim_243: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim_242, 3);  squeeze_dim_242 = None
        squeeze_dim_244: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default_219, 4);  reshape_default_219 = None
        squeeze_dim_245: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim_244, 3);  squeeze_dim_244 = None
        add_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, squeeze_dim_245);  add_tensor_1 = squeeze_dim_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1116 in forward, code: output_h = self.dropout(word_emb_k)
        convert_element_type_default: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_48: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_49: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_48);  add_tensor_2 = mul_tensor_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1115 in forward, code: word_emb_k = self.word_embedding(input_ids)
        eq_scalar: "b8[512, 16]" = torch.ops.aten.eq.Scalar(clone, -1)
        unsqueeze_default_123: "b8[512, 16, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[512, 16, 1024]" = torch.ops.aten.where.self(unsqueeze_default_123, full_default_1, mul_tensor_49);  unsqueeze_default_123 = full_default_1 = mul_tensor_49 = None
        full_default: "f32[32000, 1024]" = torch.ops.aten.full.default([32000, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32000, 1024]" = torch.ops.aten.index_put.default(full_default, [clone], where_self, True);  full_default = clone = where_self = None
        add_tensor_3: "f32[32000, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_default);  mm_1 = index_put_default = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, permute_default_1, reshape_default_2, sum_dim_int_list_5, sum_dim_int_list_6, squeeze_dim_1, reshape_default_4, reshape_default_5, squeeze_dim_3, squeeze_dim_5, squeeze_dim_7, squeeze_dim_9, sum_dim_int_list_9, sum_dim_int_list_10, permute_default_4, reshape_default_10, permute_default_5, reshape_default_11, sum_dim_int_list_13, sum_dim_int_list_14, squeeze_dim_11, reshape_default_13, reshape_default_14, squeeze_dim_13, squeeze_dim_15, squeeze_dim_17, squeeze_dim_19, sum_dim_int_list_17, sum_dim_int_list_18, permute_default_8, reshape_default_19, permute_default_9, reshape_default_20, sum_dim_int_list_21, sum_dim_int_list_22, squeeze_dim_21, reshape_default_22, reshape_default_23, squeeze_dim_23, squeeze_dim_25, squeeze_dim_27, squeeze_dim_29, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_12, reshape_default_28, permute_default_13, reshape_default_29, sum_dim_int_list_29, sum_dim_int_list_30, squeeze_dim_31, reshape_default_31, reshape_default_32, squeeze_dim_33, squeeze_dim_35, squeeze_dim_37, squeeze_dim_39, sum_dim_int_list_33, sum_dim_int_list_34, permute_default_16, reshape_default_37, permute_default_17, reshape_default_38, sum_dim_int_list_37, sum_dim_int_list_38, squeeze_dim_41, reshape_default_40, reshape_default_41, squeeze_dim_43, squeeze_dim_45, squeeze_dim_47, squeeze_dim_49, sum_dim_int_list_41, sum_dim_int_list_42, permute_default_20, reshape_default_46, permute_default_21, reshape_default_47, sum_dim_int_list_45, sum_dim_int_list_46, squeeze_dim_51, reshape_default_49, reshape_default_50, squeeze_dim_53, squeeze_dim_55, squeeze_dim_57, squeeze_dim_59, sum_dim_int_list_49, sum_dim_int_list_50, permute_default_24, reshape_default_55, permute_default_25, reshape_default_56, sum_dim_int_list_53, sum_dim_int_list_54, squeeze_dim_61, reshape_default_58, reshape_default_59, squeeze_dim_63, squeeze_dim_65, squeeze_dim_67, squeeze_dim_69, sum_dim_int_list_57, sum_dim_int_list_58, permute_default_28, reshape_default_64, permute_default_29, reshape_default_65, sum_dim_int_list_61, sum_dim_int_list_62, squeeze_dim_71, reshape_default_67, reshape_default_68, squeeze_dim_73, squeeze_dim_75, squeeze_dim_77, squeeze_dim_79, sum_dim_int_list_65, sum_dim_int_list_66, permute_default_32, reshape_default_73, permute_default_33, reshape_default_74, sum_dim_int_list_69, sum_dim_int_list_70, squeeze_dim_81, reshape_default_76, reshape_default_77, squeeze_dim_83, squeeze_dim_85, squeeze_dim_87, squeeze_dim_89, sum_dim_int_list_73, sum_dim_int_list_74, permute_default_36, reshape_default_82, permute_default_37, reshape_default_83, sum_dim_int_list_77, sum_dim_int_list_78, squeeze_dim_91, reshape_default_85, reshape_default_86, squeeze_dim_93, squeeze_dim_95, squeeze_dim_97, squeeze_dim_99, sum_dim_int_list_81, sum_dim_int_list_82, permute_default_40, reshape_default_91, permute_default_41, reshape_default_92, sum_dim_int_list_85, sum_dim_int_list_86, squeeze_dim_101, reshape_default_94, reshape_default_95, squeeze_dim_103, squeeze_dim_105, squeeze_dim_107, squeeze_dim_109, sum_dim_int_list_89, sum_dim_int_list_90, permute_default_44, reshape_default_100, permute_default_45, reshape_default_101, sum_dim_int_list_93, sum_dim_int_list_94, squeeze_dim_111, reshape_default_103, reshape_default_104, squeeze_dim_113, squeeze_dim_115, squeeze_dim_117, squeeze_dim_119, sum_dim_int_list_97, sum_dim_int_list_98, permute_default_48, reshape_default_109, permute_default_49, reshape_default_110, sum_dim_int_list_101, sum_dim_int_list_102, squeeze_dim_121, reshape_default_112, reshape_default_113, squeeze_dim_123, squeeze_dim_125, squeeze_dim_127, squeeze_dim_129, sum_dim_int_list_105, sum_dim_int_list_106, permute_default_52, reshape_default_118, permute_default_53, reshape_default_119, sum_dim_int_list_109, sum_dim_int_list_110, squeeze_dim_131, reshape_default_121, reshape_default_122, squeeze_dim_133, squeeze_dim_135, squeeze_dim_137, squeeze_dim_139, sum_dim_int_list_113, sum_dim_int_list_114, permute_default_56, reshape_default_127, permute_default_57, reshape_default_128, sum_dim_int_list_117, sum_dim_int_list_118, squeeze_dim_141, reshape_default_130, reshape_default_131, squeeze_dim_143, squeeze_dim_145, squeeze_dim_147, squeeze_dim_149, sum_dim_int_list_121, sum_dim_int_list_122, permute_default_60, reshape_default_136, permute_default_61, reshape_default_137, sum_dim_int_list_125, sum_dim_int_list_126, squeeze_dim_151, reshape_default_139, reshape_default_140, squeeze_dim_153, squeeze_dim_155, squeeze_dim_157, squeeze_dim_159, sum_dim_int_list_129, sum_dim_int_list_130, permute_default_64, reshape_default_145, permute_default_65, reshape_default_146, sum_dim_int_list_133, sum_dim_int_list_134, squeeze_dim_161, reshape_default_148, reshape_default_149, squeeze_dim_163, squeeze_dim_165, squeeze_dim_167, squeeze_dim_169, sum_dim_int_list_137, sum_dim_int_list_138, permute_default_68, reshape_default_154, permute_default_69, reshape_default_155, sum_dim_int_list_141, sum_dim_int_list_142, squeeze_dim_171, reshape_default_157, reshape_default_158, squeeze_dim_173, squeeze_dim_175, squeeze_dim_177, squeeze_dim_179, sum_dim_int_list_145, sum_dim_int_list_146, permute_default_72, reshape_default_163, permute_default_73, reshape_default_164, sum_dim_int_list_149, sum_dim_int_list_150, squeeze_dim_181, reshape_default_166, reshape_default_167, squeeze_dim_183, squeeze_dim_185, squeeze_dim_187, squeeze_dim_189, sum_dim_int_list_153, sum_dim_int_list_154, permute_default_76, reshape_default_172, permute_default_77, reshape_default_173, sum_dim_int_list_157, sum_dim_int_list_158, squeeze_dim_191, reshape_default_175, reshape_default_176, squeeze_dim_193, squeeze_dim_195, squeeze_dim_197, squeeze_dim_199, sum_dim_int_list_161, sum_dim_int_list_162, permute_default_80, reshape_default_181, permute_default_81, reshape_default_182, sum_dim_int_list_165, sum_dim_int_list_166, squeeze_dim_201, reshape_default_184, reshape_default_185, squeeze_dim_203, squeeze_dim_205, squeeze_dim_207, squeeze_dim_209, sum_dim_int_list_169, sum_dim_int_list_170, permute_default_84, reshape_default_190, permute_default_85, reshape_default_191, sum_dim_int_list_173, sum_dim_int_list_174, squeeze_dim_211, reshape_default_193, reshape_default_194, squeeze_dim_213, squeeze_dim_215, squeeze_dim_217, squeeze_dim_219, sum_dim_int_list_177, sum_dim_int_list_178, permute_default_88, reshape_default_199, permute_default_89, reshape_default_200, sum_dim_int_list_181, sum_dim_int_list_182, squeeze_dim_221, reshape_default_202, reshape_default_203, squeeze_dim_223, squeeze_dim_225, squeeze_dim_227, squeeze_dim_229, sum_dim_int_list_185, sum_dim_int_list_186, permute_default_92, reshape_default_208, permute_default_93, reshape_default_209, sum_dim_int_list_189, sum_dim_int_list_190, squeeze_dim_231, reshape_default_211, reshape_default_212, squeeze_dim_233, squeeze_dim_235, squeeze_dim_239, squeeze_dim_243, add_tensor_3)


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
