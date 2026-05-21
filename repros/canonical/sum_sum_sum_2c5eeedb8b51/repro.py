"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
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
_shapes_config = "(T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([2048, 1024], bf16), T([4, 16, 512, 128], bf16, stride=(1048576, 128, 2048, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([4, 512, 16, 128], f32), T([4, 512, 16, 1], f32), T([4, 512, 16, 128], bf16, stride=(1048576, 128, 65536, 1)), T([2048, 2048], bf16), T([2048, 1024], bf16), T([1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([4, 512], i64, gen=Index(100)), T([], f32), T([151936, 1024], bf16))"

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_675: "f32[4, 512, 1024]", rsqrt_112: "f32[4, 512, 1]", view_570: "bf16[4, 512, 1024]", view_572: "bf16[2048, 1024]", view_574: "bf16[2048, 3072]", view_576: "bf16[2048, 3072]", convert_element_type_665: "f32[4, 512, 1024]", rsqrt_111: "f32[4, 512, 1]", add_260: "bf16[4, 512, 1024]", view_579: "bf16[2048, 1024]", getitem_243: "bf16[4, 16, 512, 128]", view_585: "bf16[2048, 1024]", convert_element_type_659: "f32[4, 512, 8, 128]", rsqrt_110: "f32[4, 512, 8, 1]", permute_336: "bf16[4, 512, 8, 128]", view_589: "bf16[2048, 1024]", convert_element_type_655: "f32[4, 512, 16, 128]", rsqrt_109: "f32[4, 512, 16, 1]", permute_341: "bf16[4, 512, 16, 128]", view_593: "bf16[2048, 2048]", convert_element_type_651: "f32[4, 512, 1024]", rsqrt_108: "f32[4, 512, 1]", add_270: "bf16[4, 512, 1024]", view_596: "bf16[2048, 1024]", view_598: "bf16[2048, 3072]", view_600: "bf16[2048, 3072]", convert_element_type_641: "f32[4, 512, 1024]", rsqrt_107: "f32[4, 512, 1]", add_275: "bf16[4, 512, 1024]", view_603: "bf16[2048, 1024]", getitem_234: "bf16[4, 16, 512, 128]", view_609: "bf16[2048, 1024]", convert_element_type_635: "f32[4, 512, 8, 128]", rsqrt_106: "f32[4, 512, 8, 1]", permute_368: "bf16[4, 512, 8, 128]", view_613: "bf16[2048, 1024]", convert_element_type_631: "f32[4, 512, 16, 128]", rsqrt_105: "f32[4, 512, 16, 1]", permute_373: "bf16[4, 512, 16, 128]", view_617: "bf16[2048, 2048]", convert_element_type_627: "f32[4, 512, 1024]", rsqrt_104: "f32[4, 512, 1]", add_285: "bf16[4, 512, 1024]", view_620: "bf16[2048, 1024]", view_622: "bf16[2048, 3072]", view_624: "bf16[2048, 3072]", convert_element_type_617: "f32[4, 512, 1024]", rsqrt_103: "f32[4, 512, 1]", add_290: "bf16[4, 512, 1024]", view_627: "bf16[2048, 1024]", getitem_225: "bf16[4, 16, 512, 128]", view_633: "bf16[2048, 1024]", convert_element_type_611: "f32[4, 512, 8, 128]", rsqrt_102: "f32[4, 512, 8, 1]", permute_400: "bf16[4, 512, 8, 128]", view_637: "bf16[2048, 1024]", convert_element_type_607: "f32[4, 512, 16, 128]", rsqrt_101: "f32[4, 512, 16, 1]", permute_405: "bf16[4, 512, 16, 128]", view_641: "bf16[2048, 2048]", convert_element_type_603: "f32[4, 512, 1024]", rsqrt_100: "f32[4, 512, 1]", add_300: "bf16[4, 512, 1024]", view_644: "bf16[2048, 1024]", view_646: "bf16[2048, 3072]", view_648: "bf16[2048, 3072]", convert_element_type_593: "f32[4, 512, 1024]", rsqrt_99: "f32[4, 512, 1]", add_305: "bf16[4, 512, 1024]", view_651: "bf16[2048, 1024]", getitem_216: "bf16[4, 16, 512, 128]", view_657: "bf16[2048, 1024]", convert_element_type_587: "f32[4, 512, 8, 128]", rsqrt_98: "f32[4, 512, 8, 1]", permute_432: "bf16[4, 512, 8, 128]", view_661: "bf16[2048, 1024]", convert_element_type_583: "f32[4, 512, 16, 128]", rsqrt_97: "f32[4, 512, 16, 1]", permute_437: "bf16[4, 512, 16, 128]", view_665: "bf16[2048, 2048]", convert_element_type_579: "f32[4, 512, 1024]", rsqrt_96: "f32[4, 512, 1]", add_315: "bf16[4, 512, 1024]", view_668: "bf16[2048, 1024]", view_670: "bf16[2048, 3072]", view_672: "bf16[2048, 3072]", convert_element_type_569: "f32[4, 512, 1024]", rsqrt_95: "f32[4, 512, 1]", add_320: "bf16[4, 512, 1024]", view_675: "bf16[2048, 1024]", getitem_207: "bf16[4, 16, 512, 128]", view_681: "bf16[2048, 1024]", convert_element_type_563: "f32[4, 512, 8, 128]", rsqrt_94: "f32[4, 512, 8, 1]", permute_464: "bf16[4, 512, 8, 128]", view_685: "bf16[2048, 1024]", convert_element_type_559: "f32[4, 512, 16, 128]", rsqrt_93: "f32[4, 512, 16, 1]", permute_469: "bf16[4, 512, 16, 128]", view_689: "bf16[2048, 2048]", convert_element_type_555: "f32[4, 512, 1024]", rsqrt_92: "f32[4, 512, 1]", add_330: "bf16[4, 512, 1024]", view_692: "bf16[2048, 1024]", view_694: "bf16[2048, 3072]", view_696: "bf16[2048, 3072]", convert_element_type_545: "f32[4, 512, 1024]", rsqrt_91: "f32[4, 512, 1]", add_335: "bf16[4, 512, 1024]", view_699: "bf16[2048, 1024]", getitem_198: "bf16[4, 16, 512, 128]", view_705: "bf16[2048, 1024]", convert_element_type_539: "f32[4, 512, 8, 128]", rsqrt_90: "f32[4, 512, 8, 1]", permute_496: "bf16[4, 512, 8, 128]", view_709: "bf16[2048, 1024]", convert_element_type_535: "f32[4, 512, 16, 128]", rsqrt_89: "f32[4, 512, 16, 1]", permute_501: "bf16[4, 512, 16, 128]", view_713: "bf16[2048, 2048]", convert_element_type_531: "f32[4, 512, 1024]", rsqrt_88: "f32[4, 512, 1]", add_345: "bf16[4, 512, 1024]", view_716: "bf16[2048, 1024]", view_718: "bf16[2048, 3072]", view_720: "bf16[2048, 3072]", convert_element_type_521: "f32[4, 512, 1024]", rsqrt_87: "f32[4, 512, 1]", add_350: "bf16[4, 512, 1024]", view_723: "bf16[2048, 1024]", getitem_189: "bf16[4, 16, 512, 128]", view_729: "bf16[2048, 1024]", convert_element_type_515: "f32[4, 512, 8, 128]", rsqrt_86: "f32[4, 512, 8, 1]", permute_528: "bf16[4, 512, 8, 128]", view_733: "bf16[2048, 1024]", convert_element_type_511: "f32[4, 512, 16, 128]", rsqrt_85: "f32[4, 512, 16, 1]", permute_533: "bf16[4, 512, 16, 128]", view_737: "bf16[2048, 2048]", convert_element_type_507: "f32[4, 512, 1024]", rsqrt_84: "f32[4, 512, 1]", add_360: "bf16[4, 512, 1024]", view_740: "bf16[2048, 1024]", view_742: "bf16[2048, 3072]", view_744: "bf16[2048, 3072]", convert_element_type_497: "f32[4, 512, 1024]", rsqrt_83: "f32[4, 512, 1]", add_365: "bf16[4, 512, 1024]", view_747: "bf16[2048, 1024]", getitem_180: "bf16[4, 16, 512, 128]", view_753: "bf16[2048, 1024]", convert_element_type_491: "f32[4, 512, 8, 128]", rsqrt_82: "f32[4, 512, 8, 1]", permute_560: "bf16[4, 512, 8, 128]", view_757: "bf16[2048, 1024]", convert_element_type_487: "f32[4, 512, 16, 128]", rsqrt_81: "f32[4, 512, 16, 1]", permute_565: "bf16[4, 512, 16, 128]", view_761: "bf16[2048, 2048]", convert_element_type_483: "f32[4, 512, 1024]", rsqrt_80: "f32[4, 512, 1]", add_375: "bf16[4, 512, 1024]", view_764: "bf16[2048, 1024]", view_766: "bf16[2048, 3072]", view_768: "bf16[2048, 3072]", convert_element_type_473: "f32[4, 512, 1024]", rsqrt_79: "f32[4, 512, 1]", add_380: "bf16[4, 512, 1024]", view_771: "bf16[2048, 1024]", getitem_171: "bf16[4, 16, 512, 128]", view_777: "bf16[2048, 1024]", convert_element_type_467: "f32[4, 512, 8, 128]", rsqrt_78: "f32[4, 512, 8, 1]", permute_592: "bf16[4, 512, 8, 128]", view_781: "bf16[2048, 1024]", convert_element_type_463: "f32[4, 512, 16, 128]", rsqrt_77: "f32[4, 512, 16, 1]", permute_597: "bf16[4, 512, 16, 128]", view_785: "bf16[2048, 2048]", convert_element_type_459: "f32[4, 512, 1024]", rsqrt_76: "f32[4, 512, 1]", add_390: "bf16[4, 512, 1024]", view_788: "bf16[2048, 1024]", view_790: "bf16[2048, 3072]", view_792: "bf16[2048, 3072]", convert_element_type_449: "f32[4, 512, 1024]", rsqrt_75: "f32[4, 512, 1]", add_395: "bf16[4, 512, 1024]", view_795: "bf16[2048, 1024]", getitem_162: "bf16[4, 16, 512, 128]", view_801: "bf16[2048, 1024]", convert_element_type_443: "f32[4, 512, 8, 128]", rsqrt_74: "f32[4, 512, 8, 1]", permute_624: "bf16[4, 512, 8, 128]", view_805: "bf16[2048, 1024]", convert_element_type_439: "f32[4, 512, 16, 128]", rsqrt_73: "f32[4, 512, 16, 1]", permute_629: "bf16[4, 512, 16, 128]", view_809: "bf16[2048, 2048]", convert_element_type_435: "f32[4, 512, 1024]", rsqrt_72: "f32[4, 512, 1]", add_405: "bf16[4, 512, 1024]", view_812: "bf16[2048, 1024]", view_814: "bf16[2048, 3072]", view_816: "bf16[2048, 3072]", convert_element_type_425: "f32[4, 512, 1024]", rsqrt_71: "f32[4, 512, 1]", add_410: "bf16[4, 512, 1024]", view_819: "bf16[2048, 1024]", getitem_153: "bf16[4, 16, 512, 128]", view_825: "bf16[2048, 1024]", convert_element_type_419: "f32[4, 512, 8, 128]", rsqrt_70: "f32[4, 512, 8, 1]", permute_656: "bf16[4, 512, 8, 128]", view_829: "bf16[2048, 1024]", convert_element_type_415: "f32[4, 512, 16, 128]", rsqrt_69: "f32[4, 512, 16, 1]", permute_661: "bf16[4, 512, 16, 128]", view_833: "bf16[2048, 2048]", convert_element_type_411: "f32[4, 512, 1024]", rsqrt_68: "f32[4, 512, 1]", add_420: "bf16[4, 512, 1024]", view_836: "bf16[2048, 1024]", view_838: "bf16[2048, 3072]", view_840: "bf16[2048, 3072]", convert_element_type_401: "f32[4, 512, 1024]", rsqrt_67: "f32[4, 512, 1]", add_425: "bf16[4, 512, 1024]", view_843: "bf16[2048, 1024]", getitem_144: "bf16[4, 16, 512, 128]", view_849: "bf16[2048, 1024]", convert_element_type_395: "f32[4, 512, 8, 128]", rsqrt_66: "f32[4, 512, 8, 1]", permute_688: "bf16[4, 512, 8, 128]", view_853: "bf16[2048, 1024]", convert_element_type_391: "f32[4, 512, 16, 128]", rsqrt_65: "f32[4, 512, 16, 1]", permute_693: "bf16[4, 512, 16, 128]", view_857: "bf16[2048, 2048]", convert_element_type_387: "f32[4, 512, 1024]", rsqrt_64: "f32[4, 512, 1]", add_435: "bf16[4, 512, 1024]", view_860: "bf16[2048, 1024]", view_862: "bf16[2048, 3072]", view_864: "bf16[2048, 3072]", convert_element_type_377: "f32[4, 512, 1024]", rsqrt_63: "f32[4, 512, 1]", add_440: "bf16[4, 512, 1024]", view_867: "bf16[2048, 1024]", getitem_135: "bf16[4, 16, 512, 128]", view_873: "bf16[2048, 1024]", convert_element_type_371: "f32[4, 512, 8, 128]", rsqrt_62: "f32[4, 512, 8, 1]", permute_720: "bf16[4, 512, 8, 128]", view_877: "bf16[2048, 1024]", convert_element_type_367: "f32[4, 512, 16, 128]", rsqrt_61: "f32[4, 512, 16, 1]", permute_725: "bf16[4, 512, 16, 128]", view_881: "bf16[2048, 2048]", convert_element_type_363: "f32[4, 512, 1024]", rsqrt_60: "f32[4, 512, 1]", add_450: "bf16[4, 512, 1024]", view_884: "bf16[2048, 1024]", view_886: "bf16[2048, 3072]", view_888: "bf16[2048, 3072]", convert_element_type_353: "f32[4, 512, 1024]", rsqrt_59: "f32[4, 512, 1]", add_455: "bf16[4, 512, 1024]", view_891: "bf16[2048, 1024]", getitem_126: "bf16[4, 16, 512, 128]", view_897: "bf16[2048, 1024]", convert_element_type_347: "f32[4, 512, 8, 128]", rsqrt_58: "f32[4, 512, 8, 1]", permute_752: "bf16[4, 512, 8, 128]", view_901: "bf16[2048, 1024]", convert_element_type_343: "f32[4, 512, 16, 128]", rsqrt_57: "f32[4, 512, 16, 1]", permute_757: "bf16[4, 512, 16, 128]", view_905: "bf16[2048, 2048]", convert_element_type_339: "f32[4, 512, 1024]", rsqrt_56: "f32[4, 512, 1]", add_465: "bf16[4, 512, 1024]", view_908: "bf16[2048, 1024]", view_910: "bf16[2048, 3072]", view_912: "bf16[2048, 3072]", convert_element_type_329: "f32[4, 512, 1024]", rsqrt_55: "f32[4, 512, 1]", add_470: "bf16[4, 512, 1024]", view_915: "bf16[2048, 1024]", getitem_117: "bf16[4, 16, 512, 128]", view_921: "bf16[2048, 1024]", convert_element_type_323: "f32[4, 512, 8, 128]", rsqrt_54: "f32[4, 512, 8, 1]", permute_784: "bf16[4, 512, 8, 128]", view_925: "bf16[2048, 1024]", convert_element_type_319: "f32[4, 512, 16, 128]", rsqrt_53: "f32[4, 512, 16, 1]", permute_789: "bf16[4, 512, 16, 128]", view_929: "bf16[2048, 2048]", convert_element_type_315: "f32[4, 512, 1024]", rsqrt_52: "f32[4, 512, 1]", add_480: "bf16[4, 512, 1024]", view_932: "bf16[2048, 1024]", view_934: "bf16[2048, 3072]", view_936: "bf16[2048, 3072]", convert_element_type_305: "f32[4, 512, 1024]", rsqrt_51: "f32[4, 512, 1]", add_485: "bf16[4, 512, 1024]", view_939: "bf16[2048, 1024]", getitem_108: "bf16[4, 16, 512, 128]", view_945: "bf16[2048, 1024]", convert_element_type_299: "f32[4, 512, 8, 128]", rsqrt_50: "f32[4, 512, 8, 1]", permute_816: "bf16[4, 512, 8, 128]", view_949: "bf16[2048, 1024]", convert_element_type_295: "f32[4, 512, 16, 128]", rsqrt_49: "f32[4, 512, 16, 1]", permute_821: "bf16[4, 512, 16, 128]", view_953: "bf16[2048, 2048]", convert_element_type_291: "f32[4, 512, 1024]", rsqrt_48: "f32[4, 512, 1]", add_495: "bf16[4, 512, 1024]", view_956: "bf16[2048, 1024]", view_958: "bf16[2048, 3072]", view_960: "bf16[2048, 3072]", convert_element_type_281: "f32[4, 512, 1024]", rsqrt_47: "f32[4, 512, 1]", add_500: "bf16[4, 512, 1024]", view_963: "bf16[2048, 1024]", getitem_99: "bf16[4, 16, 512, 128]", view_969: "bf16[2048, 1024]", convert_element_type_275: "f32[4, 512, 8, 128]", rsqrt_46: "f32[4, 512, 8, 1]", permute_848: "bf16[4, 512, 8, 128]", view_973: "bf16[2048, 1024]", convert_element_type_271: "f32[4, 512, 16, 128]", rsqrt_45: "f32[4, 512, 16, 1]", permute_853: "bf16[4, 512, 16, 128]", view_977: "bf16[2048, 2048]", convert_element_type_267: "f32[4, 512, 1024]", rsqrt_44: "f32[4, 512, 1]", add_510: "bf16[4, 512, 1024]", view_980: "bf16[2048, 1024]", view_982: "bf16[2048, 3072]", view_984: "bf16[2048, 3072]", convert_element_type_257: "f32[4, 512, 1024]", rsqrt_43: "f32[4, 512, 1]", add_515: "bf16[4, 512, 1024]", view_987: "bf16[2048, 1024]", getitem_90: "bf16[4, 16, 512, 128]", view_993: "bf16[2048, 1024]", convert_element_type_251: "f32[4, 512, 8, 128]", rsqrt_42: "f32[4, 512, 8, 1]", permute_880: "bf16[4, 512, 8, 128]", view_997: "bf16[2048, 1024]", convert_element_type_247: "f32[4, 512, 16, 128]", rsqrt_41: "f32[4, 512, 16, 1]", permute_885: "bf16[4, 512, 16, 128]", view_1001: "bf16[2048, 2048]", convert_element_type_243: "f32[4, 512, 1024]", rsqrt_40: "f32[4, 512, 1]", add_525: "bf16[4, 512, 1024]", view_1004: "bf16[2048, 1024]", view_1006: "bf16[2048, 3072]", view_1008: "bf16[2048, 3072]", convert_element_type_233: "f32[4, 512, 1024]", rsqrt_39: "f32[4, 512, 1]", add_530: "bf16[4, 512, 1024]", view_1011: "bf16[2048, 1024]", getitem_81: "bf16[4, 16, 512, 128]", view_1017: "bf16[2048, 1024]", convert_element_type_227: "f32[4, 512, 8, 128]", rsqrt_38: "f32[4, 512, 8, 1]", permute_912: "bf16[4, 512, 8, 128]", view_1021: "bf16[2048, 1024]", convert_element_type_223: "f32[4, 512, 16, 128]", rsqrt_37: "f32[4, 512, 16, 1]", permute_917: "bf16[4, 512, 16, 128]", view_1025: "bf16[2048, 2048]", convert_element_type_219: "f32[4, 512, 1024]", rsqrt_36: "f32[4, 512, 1]", add_540: "bf16[4, 512, 1024]", view_1028: "bf16[2048, 1024]", view_1030: "bf16[2048, 3072]", view_1032: "bf16[2048, 3072]", convert_element_type_209: "f32[4, 512, 1024]", rsqrt_35: "f32[4, 512, 1]", add_545: "bf16[4, 512, 1024]", view_1035: "bf16[2048, 1024]", getitem_72: "bf16[4, 16, 512, 128]", view_1041: "bf16[2048, 1024]", convert_element_type_203: "f32[4, 512, 8, 128]", rsqrt_34: "f32[4, 512, 8, 1]", permute_944: "bf16[4, 512, 8, 128]", view_1045: "bf16[2048, 1024]", convert_element_type_199: "f32[4, 512, 16, 128]", rsqrt_33: "f32[4, 512, 16, 1]", permute_949: "bf16[4, 512, 16, 128]", view_1049: "bf16[2048, 2048]", convert_element_type_195: "f32[4, 512, 1024]", rsqrt_32: "f32[4, 512, 1]", add_555: "bf16[4, 512, 1024]", view_1052: "bf16[2048, 1024]", view_1054: "bf16[2048, 3072]", view_1056: "bf16[2048, 3072]", convert_element_type_185: "f32[4, 512, 1024]", rsqrt_31: "f32[4, 512, 1]", add_560: "bf16[4, 512, 1024]", view_1059: "bf16[2048, 1024]", getitem_63: "bf16[4, 16, 512, 128]", view_1065: "bf16[2048, 1024]", convert_element_type_179: "f32[4, 512, 8, 128]", rsqrt_30: "f32[4, 512, 8, 1]", permute_976: "bf16[4, 512, 8, 128]", view_1069: "bf16[2048, 1024]", convert_element_type_175: "f32[4, 512, 16, 128]", rsqrt_29: "f32[4, 512, 16, 1]", permute_981: "bf16[4, 512, 16, 128]", view_1073: "bf16[2048, 2048]", convert_element_type_171: "f32[4, 512, 1024]", rsqrt_28: "f32[4, 512, 1]", add_570: "bf16[4, 512, 1024]", view_1076: "bf16[2048, 1024]", view_1078: "bf16[2048, 3072]", view_1080: "bf16[2048, 3072]", convert_element_type_161: "f32[4, 512, 1024]", rsqrt_27: "f32[4, 512, 1]", add_575: "bf16[4, 512, 1024]", view_1083: "bf16[2048, 1024]", getitem_54: "bf16[4, 16, 512, 128]", view_1089: "bf16[2048, 1024]", convert_element_type_155: "f32[4, 512, 8, 128]", rsqrt_26: "f32[4, 512, 8, 1]", permute_1008: "bf16[4, 512, 8, 128]", view_1093: "bf16[2048, 1024]", convert_element_type_151: "f32[4, 512, 16, 128]", rsqrt_25: "f32[4, 512, 16, 1]", permute_1013: "bf16[4, 512, 16, 128]", view_1097: "bf16[2048, 2048]", convert_element_type_147: "f32[4, 512, 1024]", rsqrt_24: "f32[4, 512, 1]", add_585: "bf16[4, 512, 1024]", view_1100: "bf16[2048, 1024]", view_1102: "bf16[2048, 3072]", view_1104: "bf16[2048, 3072]", convert_element_type_137: "f32[4, 512, 1024]", rsqrt_23: "f32[4, 512, 1]", add_590: "bf16[4, 512, 1024]", view_1107: "bf16[2048, 1024]", getitem_45: "bf16[4, 16, 512, 128]", view_1113: "bf16[2048, 1024]", convert_element_type_131: "f32[4, 512, 8, 128]", rsqrt_22: "f32[4, 512, 8, 1]", permute_1040: "bf16[4, 512, 8, 128]", view_1117: "bf16[2048, 1024]", convert_element_type_127: "f32[4, 512, 16, 128]", rsqrt_21: "f32[4, 512, 16, 1]", permute_1045: "bf16[4, 512, 16, 128]", view_1121: "bf16[2048, 2048]", convert_element_type_123: "f32[4, 512, 1024]", rsqrt_20: "f32[4, 512, 1]", add_600: "bf16[4, 512, 1024]", view_1124: "bf16[2048, 1024]", view_1126: "bf16[2048, 3072]", view_1128: "bf16[2048, 3072]", convert_element_type_113: "f32[4, 512, 1024]", rsqrt_19: "f32[4, 512, 1]", add_605: "bf16[4, 512, 1024]", view_1131: "bf16[2048, 1024]", getitem_36: "bf16[4, 16, 512, 128]", view_1137: "bf16[2048, 1024]", convert_element_type_107: "f32[4, 512, 8, 128]", rsqrt_18: "f32[4, 512, 8, 1]", permute_1072: "bf16[4, 512, 8, 128]", view_1141: "bf16[2048, 1024]", convert_element_type_103: "f32[4, 512, 16, 128]", rsqrt_17: "f32[4, 512, 16, 1]", permute_1077: "bf16[4, 512, 16, 128]", view_1145: "bf16[2048, 2048]", convert_element_type_99: "f32[4, 512, 1024]", rsqrt_16: "f32[4, 512, 1]", add_615: "bf16[4, 512, 1024]", view_1148: "bf16[2048, 1024]", view_1150: "bf16[2048, 3072]", view_1152: "bf16[2048, 3072]", convert_element_type_89: "f32[4, 512, 1024]", rsqrt_15: "f32[4, 512, 1]", add_620: "bf16[4, 512, 1024]", view_1155: "bf16[2048, 1024]", getitem_27: "bf16[4, 16, 512, 128]", view_1161: "bf16[2048, 1024]", convert_element_type_83: "f32[4, 512, 8, 128]", rsqrt_14: "f32[4, 512, 8, 1]", permute_1104: "bf16[4, 512, 8, 128]", view_1165: "bf16[2048, 1024]", convert_element_type_79: "f32[4, 512, 16, 128]", rsqrt_13: "f32[4, 512, 16, 1]", permute_1109: "bf16[4, 512, 16, 128]", view_1169: "bf16[2048, 2048]", convert_element_type_75: "f32[4, 512, 1024]", rsqrt_12: "f32[4, 512, 1]", add_630: "bf16[4, 512, 1024]", view_1172: "bf16[2048, 1024]", view_1174: "bf16[2048, 3072]", view_1176: "bf16[2048, 3072]", convert_element_type_65: "f32[4, 512, 1024]", rsqrt_11: "f32[4, 512, 1]", add_635: "bf16[4, 512, 1024]", view_1179: "bf16[2048, 1024]", getitem_18: "bf16[4, 16, 512, 128]", view_1185: "bf16[2048, 1024]", convert_element_type_59: "f32[4, 512, 8, 128]", rsqrt_10: "f32[4, 512, 8, 1]", permute_1136: "bf16[4, 512, 8, 128]", view_1189: "bf16[2048, 1024]", convert_element_type_55: "f32[4, 512, 16, 128]", rsqrt_9: "f32[4, 512, 16, 1]", permute_1141: "bf16[4, 512, 16, 128]", view_1193: "bf16[2048, 2048]", convert_element_type_51: "f32[4, 512, 1024]", rsqrt_8: "f32[4, 512, 1]", add_645: "bf16[4, 512, 1024]", view_1196: "bf16[2048, 1024]", view_1198: "bf16[2048, 3072]", view_1200: "bf16[2048, 3072]", convert_element_type_41: "f32[4, 512, 1024]", rsqrt_7: "f32[4, 512, 1]", add_650: "bf16[4, 512, 1024]", view_1203: "bf16[2048, 1024]", getitem_9: "bf16[4, 16, 512, 128]", view_1209: "bf16[2048, 1024]", convert_element_type_35: "f32[4, 512, 8, 128]", rsqrt_6: "f32[4, 512, 8, 1]", permute_1168: "bf16[4, 512, 8, 128]", view_1213: "bf16[2048, 1024]", convert_element_type_31: "f32[4, 512, 16, 128]", rsqrt_5: "f32[4, 512, 16, 1]", permute_1173: "bf16[4, 512, 16, 128]", view_1217: "bf16[2048, 2048]", convert_element_type_27: "f32[4, 512, 1024]", rsqrt_4: "f32[4, 512, 1]", add_660: "bf16[4, 512, 1024]", view_1220: "bf16[2048, 1024]", view_1222: "bf16[2048, 3072]", view_1224: "bf16[2048, 3072]", convert_element_type_17: "f32[4, 512, 1024]", rsqrt_3: "f32[4, 512, 1]", add_665: "bf16[4, 512, 1024]", view_1227: "bf16[2048, 1024]", getitem: "bf16[4, 16, 512, 128]", view_1233: "bf16[2048, 1024]", mm_586: "bf16[2048, 1024]", convert_element_type_11: "f32[4, 512, 8, 128]", rsqrt_2: "f32[4, 512, 8, 1]", permute_1200: "bf16[4, 512, 8, 128]", view_1237: "bf16[2048, 1024]", mm_588: "bf16[2048, 1024]", convert_element_type_7: "f32[4, 512, 16, 128]", rsqrt_1: "f32[4, 512, 16, 1]", permute_1205: "bf16[4, 512, 16, 128]", view_1241: "bf16[2048, 2048]", mm_590: "bf16[2048, 1024]", primals_4: "bf16[1024]", embedding: "bf16[4, 512, 1024]", rsqrt: "f32[4, 512, 1]", add_667: "bf16[4, 512, 1024]", primals_1: "i64[4, 512]", full_default_58: "f32[]", mm_197: "bf16[151936, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_675, rsqrt_112);  convert_element_type_675 = rsqrt_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_570, convert_element_type_default);  view_570 = convert_element_type_default = None
        sum_dim_int_list: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1024]);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_572, [1, 0]);  view_572 = None
        permute_default_1: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_574, [1, 0]);  view_574 = None
        permute_default_2: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_576, [1, 0]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_665, rsqrt_111);  convert_element_type_665 = rsqrt_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, convert_element_type_default_1);  add_260 = convert_element_type_default_1 = None
        sum_dim_int_list_1: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [1024]);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_3: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_579, [1, 0]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_4: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_243, [0, 2, 1, 3]);  getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_4, [4, 512, -1]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_3: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_2, [2048, 2048]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_585, [1, 0]);  view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_659, rsqrt_110);  convert_element_type_659 = rsqrt_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_2: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_336, convert_element_type_default_2);  permute_336 = convert_element_type_default_2 = None
        sum_dim_int_list_2: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2], True);  mul_tensor_5 = None
        reshape_default_4: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [128]);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_6: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_589, [1, 0]);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_655, rsqrt_109);  convert_element_type_655 = rsqrt_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_3: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        mul_tensor_7: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_341, convert_element_type_default_3);  permute_341 = convert_element_type_default_3 = None
        sum_dim_int_list_3: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1, 2], True);  mul_tensor_7 = None
        reshape_default_5: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [128]);  sum_dim_int_list_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_7: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_593, [1, 0]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_651, rsqrt_108);  convert_element_type_651 = rsqrt_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_4: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        mul_tensor_9: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_270, convert_element_type_default_4);  add_270 = convert_element_type_default_4 = None
        sum_dim_int_list_4: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_6: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [1024]);  sum_dim_int_list_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_8: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_596, [1, 0]);  view_596 = None
        permute_default_9: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_598, [1, 0]);  view_598 = None
        permute_default_10: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_600, [1, 0]);  view_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_641, rsqrt_107);  convert_element_type_641 = rsqrt_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_5: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        mul_tensor_11: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_275, convert_element_type_default_5);  add_275 = convert_element_type_default_5 = None
        sum_dim_int_list_5: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_7: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [1024]);  sum_dim_int_list_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_11: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_603, [1, 0]);  view_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_12: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_234, [0, 2, 1, 3]);  getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_8: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_12, [4, 512, -1]);  permute_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_9: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_8, [2048, 2048]);  reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_13: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_609, [1, 0]);  view_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_635, rsqrt_106);  convert_element_type_635 = rsqrt_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_6: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_368, convert_element_type_default_6);  permute_368 = convert_element_type_default_6 = None
        sum_dim_int_list_6: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1, 2], True);  mul_tensor_13 = None
        reshape_default_10: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [128]);  sum_dim_int_list_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_14: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_613, [1, 0]);  view_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_631, rsqrt_105);  convert_element_type_631 = rsqrt_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_7: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.bfloat16);  mul_tensor_14 = None
        mul_tensor_15: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_373, convert_element_type_default_7);  permute_373 = convert_element_type_default_7 = None
        sum_dim_int_list_7: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1, 2], True);  mul_tensor_15 = None
        reshape_default_11: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [128]);  sum_dim_int_list_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_15: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_617, [1, 0]);  view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_627, rsqrt_104);  convert_element_type_627 = rsqrt_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_8: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        mul_tensor_17: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_285, convert_element_type_default_8);  add_285 = convert_element_type_default_8 = None
        sum_dim_int_list_8: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_12: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [1024]);  sum_dim_int_list_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_16: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_620, [1, 0]);  view_620 = None
        permute_default_17: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_622, [1, 0]);  view_622 = None
        permute_default_18: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_624, [1, 0]);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_617, rsqrt_103);  convert_element_type_617 = rsqrt_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_9: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.bfloat16);  mul_tensor_18 = None
        mul_tensor_19: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, convert_element_type_default_9);  add_290 = convert_element_type_default_9 = None
        sum_dim_int_list_9: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_13: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [1024]);  sum_dim_int_list_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_19: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_627, [1, 0]);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_20: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_14: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_20, [4, 512, -1]);  permute_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_15: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_14, [2048, 2048]);  reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_21: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_633, [1, 0]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_611, rsqrt_102);  convert_element_type_611 = rsqrt_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_10: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.bfloat16);  mul_tensor_20 = None
        mul_tensor_21: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_400, convert_element_type_default_10);  permute_400 = convert_element_type_default_10 = None
        sum_dim_int_list_10: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1, 2], True);  mul_tensor_21 = None
        reshape_default_16: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [128]);  sum_dim_int_list_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_22: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_637, [1, 0]);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_607, rsqrt_101);  convert_element_type_607 = rsqrt_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_11: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_22, torch.bfloat16);  mul_tensor_22 = None
        mul_tensor_23: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_405, convert_element_type_default_11);  permute_405 = convert_element_type_default_11 = None
        sum_dim_int_list_11: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1, 2], True);  mul_tensor_23 = None
        reshape_default_17: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [128]);  sum_dim_int_list_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_23: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_641, [1, 0]);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_603, rsqrt_100);  convert_element_type_603 = rsqrt_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_12: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.bfloat16);  mul_tensor_24 = None
        mul_tensor_25: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_300, convert_element_type_default_12);  add_300 = convert_element_type_default_12 = None
        sum_dim_int_list_12: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_18: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [1024]);  sum_dim_int_list_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_24: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_644, [1, 0]);  view_644 = None
        permute_default_25: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_646, [1, 0]);  view_646 = None
        permute_default_26: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_648, [1, 0]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_593, rsqrt_99);  convert_element_type_593 = rsqrt_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_13: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_26, torch.bfloat16);  mul_tensor_26 = None
        mul_tensor_27: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_305, convert_element_type_default_13);  add_305 = convert_element_type_default_13 = None
        sum_dim_int_list_13: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_19: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, [1024]);  sum_dim_int_list_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_27: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_651, [1, 0]);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_28: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_20: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_28, [4, 512, -1]);  permute_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_21: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_20, [2048, 2048]);  reshape_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_29: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_657, [1, 0]);  view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_587, rsqrt_98);  convert_element_type_587 = rsqrt_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_14: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.bfloat16);  mul_tensor_28 = None
        mul_tensor_29: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_432, convert_element_type_default_14);  permute_432 = convert_element_type_default_14 = None
        sum_dim_int_list_14: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1, 2], True);  mul_tensor_29 = None
        reshape_default_22: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, [128]);  sum_dim_int_list_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_30: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_661, [1, 0]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_583, rsqrt_97);  convert_element_type_583 = rsqrt_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_15: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_30, torch.bfloat16);  mul_tensor_30 = None
        mul_tensor_31: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_437, convert_element_type_default_15);  permute_437 = convert_element_type_default_15 = None
        sum_dim_int_list_15: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1, 2], True);  mul_tensor_31 = None
        reshape_default_23: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, [128]);  sum_dim_int_list_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_31: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_665, [1, 0]);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_32: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_579, rsqrt_96);  convert_element_type_579 = rsqrt_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_16: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.bfloat16);  mul_tensor_32 = None
        mul_tensor_33: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_315, convert_element_type_default_16);  add_315 = convert_element_type_default_16 = None
        sum_dim_int_list_16: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_24: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [1024]);  sum_dim_int_list_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_32: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_668, [1, 0]);  view_668 = None
        permute_default_33: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_670, [1, 0]);  view_670 = None
        permute_default_34: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_672, [1, 0]);  view_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_34: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_569, rsqrt_95);  convert_element_type_569 = rsqrt_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_17: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_34, torch.bfloat16);  mul_tensor_34 = None
        mul_tensor_35: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, convert_element_type_default_17);  add_320 = convert_element_type_default_17 = None
        sum_dim_int_list_17: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_25: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, [1024]);  sum_dim_int_list_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_35: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_675, [1, 0]);  view_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_36: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_26: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_36, [4, 512, -1]);  permute_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_27: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_26, [2048, 2048]);  reshape_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_37: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_681, [1, 0]);  view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_36: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_563, rsqrt_94);  convert_element_type_563 = rsqrt_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_18: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_36, torch.bfloat16);  mul_tensor_36 = None
        mul_tensor_37: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_464, convert_element_type_default_18);  permute_464 = convert_element_type_default_18 = None
        sum_dim_int_list_18: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1, 2], True);  mul_tensor_37 = None
        reshape_default_28: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, [128]);  sum_dim_int_list_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_38: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_685, [1, 0]);  view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_38: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_559, rsqrt_93);  convert_element_type_559 = rsqrt_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_19: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_38, torch.bfloat16);  mul_tensor_38 = None
        mul_tensor_39: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_469, convert_element_type_default_19);  permute_469 = convert_element_type_default_19 = None
        sum_dim_int_list_19: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1, 2], True);  mul_tensor_39 = None
        reshape_default_29: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, [128]);  sum_dim_int_list_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_39: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_689, [1, 0]);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_40: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_555, rsqrt_92);  convert_element_type_555 = rsqrt_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_20: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_40, torch.bfloat16);  mul_tensor_40 = None
        mul_tensor_41: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_330, convert_element_type_default_20);  add_330 = convert_element_type_default_20 = None
        sum_dim_int_list_20: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_30: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, [1024]);  sum_dim_int_list_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_40: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_692, [1, 0]);  view_692 = None
        permute_default_41: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_694, [1, 0]);  view_694 = None
        permute_default_42: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_696, [1, 0]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_42: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_545, rsqrt_91);  convert_element_type_545 = rsqrt_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_21: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_42, torch.bfloat16);  mul_tensor_42 = None
        mul_tensor_43: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_335, convert_element_type_default_21);  add_335 = convert_element_type_default_21 = None
        sum_dim_int_list_21: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_31: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, [1024]);  sum_dim_int_list_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_43: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_699, [1, 0]);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_44: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_32: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_44, [4, 512, -1]);  permute_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_33: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_32, [2048, 2048]);  reshape_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_45: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_705, [1, 0]);  view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_44: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_539, rsqrt_90);  convert_element_type_539 = rsqrt_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_22: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_44, torch.bfloat16);  mul_tensor_44 = None
        mul_tensor_45: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_496, convert_element_type_default_22);  permute_496 = convert_element_type_default_22 = None
        sum_dim_int_list_22: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1, 2], True);  mul_tensor_45 = None
        reshape_default_34: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, [128]);  sum_dim_int_list_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_46: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_709, [1, 0]);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_46: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_535, rsqrt_89);  convert_element_type_535 = rsqrt_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_23: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_46, torch.bfloat16);  mul_tensor_46 = None
        mul_tensor_47: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_501, convert_element_type_default_23);  permute_501 = convert_element_type_default_23 = None
        sum_dim_int_list_23: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1, 2], True);  mul_tensor_47 = None
        reshape_default_35: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [128]);  sum_dim_int_list_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_47: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_713, [1, 0]);  view_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_48: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_531, rsqrt_88);  convert_element_type_531 = rsqrt_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_24: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_48, torch.bfloat16);  mul_tensor_48 = None
        mul_tensor_49: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_345, convert_element_type_default_24);  add_345 = convert_element_type_default_24 = None
        sum_dim_int_list_24: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_49, [0, 1], True);  mul_tensor_49 = None
        reshape_default_36: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, [1024]);  sum_dim_int_list_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_48: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_716, [1, 0]);  view_716 = None
        permute_default_49: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_718, [1, 0]);  view_718 = None
        permute_default_50: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_720, [1, 0]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_50: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_521, rsqrt_87);  convert_element_type_521 = rsqrt_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_25: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_50, torch.bfloat16);  mul_tensor_50 = None
        mul_tensor_51: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_350, convert_element_type_default_25);  add_350 = convert_element_type_default_25 = None
        sum_dim_int_list_25: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [0, 1], True);  mul_tensor_51 = None
        reshape_default_37: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, [1024]);  sum_dim_int_list_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_51: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_723, [1, 0]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_52: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_38: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_52, [4, 512, -1]);  permute_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_39: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_38, [2048, 2048]);  reshape_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_53: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_729, [1, 0]);  view_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_52: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_515, rsqrt_86);  convert_element_type_515 = rsqrt_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_26: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_52, torch.bfloat16);  mul_tensor_52 = None
        mul_tensor_53: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_528, convert_element_type_default_26);  permute_528 = convert_element_type_default_26 = None
        sum_dim_int_list_26: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [0, 1, 2], True);  mul_tensor_53 = None
        reshape_default_40: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, [128]);  sum_dim_int_list_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_54: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_733, [1, 0]);  view_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_54: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_511, rsqrt_85);  convert_element_type_511 = rsqrt_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_27: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_54, torch.bfloat16);  mul_tensor_54 = None
        mul_tensor_55: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_533, convert_element_type_default_27);  permute_533 = convert_element_type_default_27 = None
        sum_dim_int_list_27: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_55, [0, 1, 2], True);  mul_tensor_55 = None
        reshape_default_41: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, [128]);  sum_dim_int_list_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_55: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_737, [1, 0]);  view_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_56: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_507, rsqrt_84);  convert_element_type_507 = rsqrt_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_28: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_56, torch.bfloat16);  mul_tensor_56 = None
        mul_tensor_57: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_360, convert_element_type_default_28);  add_360 = convert_element_type_default_28 = None
        sum_dim_int_list_28: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_57, [0, 1], True);  mul_tensor_57 = None
        reshape_default_42: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, [1024]);  sum_dim_int_list_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_56: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_740, [1, 0]);  view_740 = None
        permute_default_57: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_742, [1, 0]);  view_742 = None
        permute_default_58: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_744, [1, 0]);  view_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_58: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_497, rsqrt_83);  convert_element_type_497 = rsqrt_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_29: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_58, torch.bfloat16);  mul_tensor_58 = None
        mul_tensor_59: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_365, convert_element_type_default_29);  add_365 = convert_element_type_default_29 = None
        sum_dim_int_list_29: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_59, [0, 1], True);  mul_tensor_59 = None
        reshape_default_43: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, [1024]);  sum_dim_int_list_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_59: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_747, [1, 0]);  view_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_60: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_44: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_60, [4, 512, -1]);  permute_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_45: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_44, [2048, 2048]);  reshape_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_61: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_753, [1, 0]);  view_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_60: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_491, rsqrt_82);  convert_element_type_491 = rsqrt_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_30: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_60, torch.bfloat16);  mul_tensor_60 = None
        mul_tensor_61: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_560, convert_element_type_default_30);  permute_560 = convert_element_type_default_30 = None
        sum_dim_int_list_30: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_61, [0, 1, 2], True);  mul_tensor_61 = None
        reshape_default_46: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, [128]);  sum_dim_int_list_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_62: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_757, [1, 0]);  view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_62: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_487, rsqrt_81);  convert_element_type_487 = rsqrt_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_31: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_62, torch.bfloat16);  mul_tensor_62 = None
        mul_tensor_63: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_565, convert_element_type_default_31);  permute_565 = convert_element_type_default_31 = None
        sum_dim_int_list_31: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_63, [0, 1, 2], True);  mul_tensor_63 = None
        reshape_default_47: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, [128]);  sum_dim_int_list_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_63: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_761, [1, 0]);  view_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_64: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_483, rsqrt_80);  convert_element_type_483 = rsqrt_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_32: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_64, torch.bfloat16);  mul_tensor_64 = None
        mul_tensor_65: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_375, convert_element_type_default_32);  add_375 = convert_element_type_default_32 = None
        sum_dim_int_list_32: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_65, [0, 1], True);  mul_tensor_65 = None
        reshape_default_48: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, [1024]);  sum_dim_int_list_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_64: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_764, [1, 0]);  view_764 = None
        permute_default_65: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_766, [1, 0]);  view_766 = None
        permute_default_66: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_768, [1, 0]);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_66: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_473, rsqrt_79);  convert_element_type_473 = rsqrt_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_33: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_66, torch.bfloat16);  mul_tensor_66 = None
        mul_tensor_67: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_380, convert_element_type_default_33);  add_380 = convert_element_type_default_33 = None
        sum_dim_int_list_33: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_67, [0, 1], True);  mul_tensor_67 = None
        reshape_default_49: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, [1024]);  sum_dim_int_list_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_67: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_771, [1, 0]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_68: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_50: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_68, [4, 512, -1]);  permute_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_51: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_50, [2048, 2048]);  reshape_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_69: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_777, [1, 0]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_68: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_467, rsqrt_78);  convert_element_type_467 = rsqrt_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_34: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_68, torch.bfloat16);  mul_tensor_68 = None
        mul_tensor_69: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_592, convert_element_type_default_34);  permute_592 = convert_element_type_default_34 = None
        sum_dim_int_list_34: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_69, [0, 1, 2], True);  mul_tensor_69 = None
        reshape_default_52: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, [128]);  sum_dim_int_list_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_70: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_781, [1, 0]);  view_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_70: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_463, rsqrt_77);  convert_element_type_463 = rsqrt_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_35: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_70, torch.bfloat16);  mul_tensor_70 = None
        mul_tensor_71: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_597, convert_element_type_default_35);  permute_597 = convert_element_type_default_35 = None
        sum_dim_int_list_35: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_71, [0, 1, 2], True);  mul_tensor_71 = None
        reshape_default_53: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, [128]);  sum_dim_int_list_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_71: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_785, [1, 0]);  view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_72: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_459, rsqrt_76);  convert_element_type_459 = rsqrt_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_36: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_72, torch.bfloat16);  mul_tensor_72 = None
        mul_tensor_73: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_390, convert_element_type_default_36);  add_390 = convert_element_type_default_36 = None
        sum_dim_int_list_36: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_73, [0, 1], True);  mul_tensor_73 = None
        reshape_default_54: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, [1024]);  sum_dim_int_list_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_72: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_788, [1, 0]);  view_788 = None
        permute_default_73: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_790, [1, 0]);  view_790 = None
        permute_default_74: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_792, [1, 0]);  view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_74: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_449, rsqrt_75);  convert_element_type_449 = rsqrt_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_37: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_74, torch.bfloat16);  mul_tensor_74 = None
        mul_tensor_75: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_395, convert_element_type_default_37);  add_395 = convert_element_type_default_37 = None
        sum_dim_int_list_37: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_75, [0, 1], True);  mul_tensor_75 = None
        reshape_default_55: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, [1024]);  sum_dim_int_list_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_75: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_795, [1, 0]);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_76: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_56: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_76, [4, 512, -1]);  permute_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_57: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_56, [2048, 2048]);  reshape_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_77: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_801, [1, 0]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_76: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_443, rsqrt_74);  convert_element_type_443 = rsqrt_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_38: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_76, torch.bfloat16);  mul_tensor_76 = None
        mul_tensor_77: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_624, convert_element_type_default_38);  permute_624 = convert_element_type_default_38 = None
        sum_dim_int_list_38: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_77, [0, 1, 2], True);  mul_tensor_77 = None
        reshape_default_58: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, [128]);  sum_dim_int_list_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_78: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_805, [1, 0]);  view_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_78: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_439, rsqrt_73);  convert_element_type_439 = rsqrt_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_39: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_78, torch.bfloat16);  mul_tensor_78 = None
        mul_tensor_79: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_629, convert_element_type_default_39);  permute_629 = convert_element_type_default_39 = None
        sum_dim_int_list_39: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_79, [0, 1, 2], True);  mul_tensor_79 = None
        reshape_default_59: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, [128]);  sum_dim_int_list_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_79: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_809, [1, 0]);  view_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_80: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_435, rsqrt_72);  convert_element_type_435 = rsqrt_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_40: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_80, torch.bfloat16);  mul_tensor_80 = None
        mul_tensor_81: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_405, convert_element_type_default_40);  add_405 = convert_element_type_default_40 = None
        sum_dim_int_list_40: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_81, [0, 1], True);  mul_tensor_81 = None
        reshape_default_60: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, [1024]);  sum_dim_int_list_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_80: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_812, [1, 0]);  view_812 = None
        permute_default_81: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_814, [1, 0]);  view_814 = None
        permute_default_82: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_816, [1, 0]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_82: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_425, rsqrt_71);  convert_element_type_425 = rsqrt_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_41: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_82, torch.bfloat16);  mul_tensor_82 = None
        mul_tensor_83: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_410, convert_element_type_default_41);  add_410 = convert_element_type_default_41 = None
        sum_dim_int_list_41: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_83, [0, 1], True);  mul_tensor_83 = None
        reshape_default_61: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, [1024]);  sum_dim_int_list_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_83: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_819, [1, 0]);  view_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_84: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_62: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_84, [4, 512, -1]);  permute_default_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_63: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_62, [2048, 2048]);  reshape_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_85: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_825, [1, 0]);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_84: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_419, rsqrt_70);  convert_element_type_419 = rsqrt_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_42: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_84, torch.bfloat16);  mul_tensor_84 = None
        mul_tensor_85: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_656, convert_element_type_default_42);  permute_656 = convert_element_type_default_42 = None
        sum_dim_int_list_42: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_85, [0, 1, 2], True);  mul_tensor_85 = None
        reshape_default_64: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, [128]);  sum_dim_int_list_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_86: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_829, [1, 0]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_86: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_415, rsqrt_69);  convert_element_type_415 = rsqrt_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_43: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_86, torch.bfloat16);  mul_tensor_86 = None
        mul_tensor_87: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_661, convert_element_type_default_43);  permute_661 = convert_element_type_default_43 = None
        sum_dim_int_list_43: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_87, [0, 1, 2], True);  mul_tensor_87 = None
        reshape_default_65: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, [128]);  sum_dim_int_list_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_87: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_833, [1, 0]);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_88: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_411, rsqrt_68);  convert_element_type_411 = rsqrt_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_44: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_88, torch.bfloat16);  mul_tensor_88 = None
        mul_tensor_89: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_420, convert_element_type_default_44);  add_420 = convert_element_type_default_44 = None
        sum_dim_int_list_44: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_89, [0, 1], True);  mul_tensor_89 = None
        reshape_default_66: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, [1024]);  sum_dim_int_list_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_88: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_836, [1, 0]);  view_836 = None
        permute_default_89: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_838, [1, 0]);  view_838 = None
        permute_default_90: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_840, [1, 0]);  view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_90: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_401, rsqrt_67);  convert_element_type_401 = rsqrt_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_45: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_90, torch.bfloat16);  mul_tensor_90 = None
        mul_tensor_91: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_425, convert_element_type_default_45);  add_425 = convert_element_type_default_45 = None
        sum_dim_int_list_45: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_91, [0, 1], True);  mul_tensor_91 = None
        reshape_default_67: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, [1024]);  sum_dim_int_list_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_91: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_843, [1, 0]);  view_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_92: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_68: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_92, [4, 512, -1]);  permute_default_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_69: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_68, [2048, 2048]);  reshape_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_93: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_849, [1, 0]);  view_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_92: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_395, rsqrt_66);  convert_element_type_395 = rsqrt_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_46: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_92, torch.bfloat16);  mul_tensor_92 = None
        mul_tensor_93: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_688, convert_element_type_default_46);  permute_688 = convert_element_type_default_46 = None
        sum_dim_int_list_46: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_93, [0, 1, 2], True);  mul_tensor_93 = None
        reshape_default_70: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, [128]);  sum_dim_int_list_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_94: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_853, [1, 0]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_94: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_391, rsqrt_65);  convert_element_type_391 = rsqrt_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_47: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_94, torch.bfloat16);  mul_tensor_94 = None
        mul_tensor_95: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_693, convert_element_type_default_47);  permute_693 = convert_element_type_default_47 = None
        sum_dim_int_list_47: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_95, [0, 1, 2], True);  mul_tensor_95 = None
        reshape_default_71: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, [128]);  sum_dim_int_list_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_95: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_857, [1, 0]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_96: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_387, rsqrt_64);  convert_element_type_387 = rsqrt_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_48: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_96, torch.bfloat16);  mul_tensor_96 = None
        mul_tensor_97: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_435, convert_element_type_default_48);  add_435 = convert_element_type_default_48 = None
        sum_dim_int_list_48: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_97, [0, 1], True);  mul_tensor_97 = None
        reshape_default_72: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, [1024]);  sum_dim_int_list_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_96: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_860, [1, 0]);  view_860 = None
        permute_default_97: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_862, [1, 0]);  view_862 = None
        permute_default_98: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_864, [1, 0]);  view_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_98: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_377, rsqrt_63);  convert_element_type_377 = rsqrt_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_49: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_98, torch.bfloat16);  mul_tensor_98 = None
        mul_tensor_99: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_440, convert_element_type_default_49);  add_440 = convert_element_type_default_49 = None
        sum_dim_int_list_49: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_99, [0, 1], True);  mul_tensor_99 = None
        reshape_default_73: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, [1024]);  sum_dim_int_list_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_99: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_867, [1, 0]);  view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_100: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_74: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_100, [4, 512, -1]);  permute_default_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_75: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_74, [2048, 2048]);  reshape_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_101: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_873, [1, 0]);  view_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_100: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_371, rsqrt_62);  convert_element_type_371 = rsqrt_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_50: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_100, torch.bfloat16);  mul_tensor_100 = None
        mul_tensor_101: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_720, convert_element_type_default_50);  permute_720 = convert_element_type_default_50 = None
        sum_dim_int_list_50: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_101, [0, 1, 2], True);  mul_tensor_101 = None
        reshape_default_76: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, [128]);  sum_dim_int_list_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_102: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_877, [1, 0]);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_102: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_367, rsqrt_61);  convert_element_type_367 = rsqrt_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_51: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_102, torch.bfloat16);  mul_tensor_102 = None
        mul_tensor_103: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_725, convert_element_type_default_51);  permute_725 = convert_element_type_default_51 = None
        sum_dim_int_list_51: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_103, [0, 1, 2], True);  mul_tensor_103 = None
        reshape_default_77: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, [128]);  sum_dim_int_list_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_103: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_881, [1, 0]);  view_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_104: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_363, rsqrt_60);  convert_element_type_363 = rsqrt_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_52: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_104, torch.bfloat16);  mul_tensor_104 = None
        mul_tensor_105: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_450, convert_element_type_default_52);  add_450 = convert_element_type_default_52 = None
        sum_dim_int_list_52: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_105, [0, 1], True);  mul_tensor_105 = None
        reshape_default_78: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, [1024]);  sum_dim_int_list_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_104: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_884, [1, 0]);  view_884 = None
        permute_default_105: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_886, [1, 0]);  view_886 = None
        permute_default_106: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_888, [1, 0]);  view_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_106: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_353, rsqrt_59);  convert_element_type_353 = rsqrt_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_53: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_106, torch.bfloat16);  mul_tensor_106 = None
        mul_tensor_107: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_455, convert_element_type_default_53);  add_455 = convert_element_type_default_53 = None
        sum_dim_int_list_53: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_107, [0, 1], True);  mul_tensor_107 = None
        reshape_default_79: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, [1024]);  sum_dim_int_list_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_107: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_891, [1, 0]);  view_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_108: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_80: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_108, [4, 512, -1]);  permute_default_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_81: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_80, [2048, 2048]);  reshape_default_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_109: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_897, [1, 0]);  view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_108: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_347, rsqrt_58);  convert_element_type_347 = rsqrt_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_54: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_108, torch.bfloat16);  mul_tensor_108 = None
        mul_tensor_109: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_752, convert_element_type_default_54);  permute_752 = convert_element_type_default_54 = None
        sum_dim_int_list_54: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_109, [0, 1, 2], True);  mul_tensor_109 = None
        reshape_default_82: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, [128]);  sum_dim_int_list_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_110: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_901, [1, 0]);  view_901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_110: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_343, rsqrt_57);  convert_element_type_343 = rsqrt_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_55: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_110, torch.bfloat16);  mul_tensor_110 = None
        mul_tensor_111: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_757, convert_element_type_default_55);  permute_757 = convert_element_type_default_55 = None
        sum_dim_int_list_55: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_111, [0, 1, 2], True);  mul_tensor_111 = None
        reshape_default_83: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, [128]);  sum_dim_int_list_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_111: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_905, [1, 0]);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_112: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_339, rsqrt_56);  convert_element_type_339 = rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_56: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_112, torch.bfloat16);  mul_tensor_112 = None
        mul_tensor_113: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_465, convert_element_type_default_56);  add_465 = convert_element_type_default_56 = None
        sum_dim_int_list_56: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_113, [0, 1], True);  mul_tensor_113 = None
        reshape_default_84: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, [1024]);  sum_dim_int_list_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_112: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_908, [1, 0]);  view_908 = None
        permute_default_113: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_910, [1, 0]);  view_910 = None
        permute_default_114: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_912, [1, 0]);  view_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_114: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_329, rsqrt_55);  convert_element_type_329 = rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_57: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_114, torch.bfloat16);  mul_tensor_114 = None
        mul_tensor_115: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_470, convert_element_type_default_57);  add_470 = convert_element_type_default_57 = None
        sum_dim_int_list_57: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_115, [0, 1], True);  mul_tensor_115 = None
        reshape_default_85: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, [1024]);  sum_dim_int_list_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_115: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_915, [1, 0]);  view_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_116: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_86: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_116, [4, 512, -1]);  permute_default_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_87: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_86, [2048, 2048]);  reshape_default_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_117: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_921, [1, 0]);  view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_116: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_323, rsqrt_54);  convert_element_type_323 = rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_58: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_116, torch.bfloat16);  mul_tensor_116 = None
        mul_tensor_117: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_784, convert_element_type_default_58);  permute_784 = convert_element_type_default_58 = None
        sum_dim_int_list_58: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_117, [0, 1, 2], True);  mul_tensor_117 = None
        reshape_default_88: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, [128]);  sum_dim_int_list_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_118: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_925, [1, 0]);  view_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_118: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_319, rsqrt_53);  convert_element_type_319 = rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_59: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_118, torch.bfloat16);  mul_tensor_118 = None
        mul_tensor_119: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_789, convert_element_type_default_59);  permute_789 = convert_element_type_default_59 = None
        sum_dim_int_list_59: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_119, [0, 1, 2], True);  mul_tensor_119 = None
        reshape_default_89: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, [128]);  sum_dim_int_list_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_119: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_929, [1, 0]);  view_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_120: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_315, rsqrt_52);  convert_element_type_315 = rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_60: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_120, torch.bfloat16);  mul_tensor_120 = None
        mul_tensor_121: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_480, convert_element_type_default_60);  add_480 = convert_element_type_default_60 = None
        sum_dim_int_list_60: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_121, [0, 1], True);  mul_tensor_121 = None
        reshape_default_90: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, [1024]);  sum_dim_int_list_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_120: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_932, [1, 0]);  view_932 = None
        permute_default_121: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_934, [1, 0]);  view_934 = None
        permute_default_122: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_936, [1, 0]);  view_936 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_122: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_305, rsqrt_51);  convert_element_type_305 = rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_61: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_122, torch.bfloat16);  mul_tensor_122 = None
        mul_tensor_123: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_485, convert_element_type_default_61);  add_485 = convert_element_type_default_61 = None
        sum_dim_int_list_61: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_123, [0, 1], True);  mul_tensor_123 = None
        reshape_default_91: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, [1024]);  sum_dim_int_list_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_123: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_939, [1, 0]);  view_939 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_124: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_92: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_124, [4, 512, -1]);  permute_default_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_93: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_92, [2048, 2048]);  reshape_default_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_125: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_945, [1, 0]);  view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_124: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_50);  convert_element_type_299 = rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_62: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_124, torch.bfloat16);  mul_tensor_124 = None
        mul_tensor_125: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_816, convert_element_type_default_62);  permute_816 = convert_element_type_default_62 = None
        sum_dim_int_list_62: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_125, [0, 1, 2], True);  mul_tensor_125 = None
        reshape_default_94: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, [128]);  sum_dim_int_list_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_126: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_949, [1, 0]);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_126: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_295, rsqrt_49);  convert_element_type_295 = rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_63: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_126, torch.bfloat16);  mul_tensor_126 = None
        mul_tensor_127: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_821, convert_element_type_default_63);  permute_821 = convert_element_type_default_63 = None
        sum_dim_int_list_63: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_127, [0, 1, 2], True);  mul_tensor_127 = None
        reshape_default_95: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, [128]);  sum_dim_int_list_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_127: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_953, [1, 0]);  view_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_128: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_291, rsqrt_48);  convert_element_type_291 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_64: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_128, torch.bfloat16);  mul_tensor_128 = None
        mul_tensor_129: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_495, convert_element_type_default_64);  add_495 = convert_element_type_default_64 = None
        sum_dim_int_list_64: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_129, [0, 1], True);  mul_tensor_129 = None
        reshape_default_96: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, [1024]);  sum_dim_int_list_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_128: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_956, [1, 0]);  view_956 = None
        permute_default_129: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_958, [1, 0]);  view_958 = None
        permute_default_130: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_960, [1, 0]);  view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_130: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_281, rsqrt_47);  convert_element_type_281 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_65: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_130, torch.bfloat16);  mul_tensor_130 = None
        mul_tensor_131: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_500, convert_element_type_default_65);  add_500 = convert_element_type_default_65 = None
        sum_dim_int_list_65: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_131, [0, 1], True);  mul_tensor_131 = None
        reshape_default_97: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, [1024]);  sum_dim_int_list_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_131: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_963, [1, 0]);  view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_132: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_98: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_132, [4, 512, -1]);  permute_default_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_99: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_98, [2048, 2048]);  reshape_default_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_133: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_969, [1, 0]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_132: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_275, rsqrt_46);  convert_element_type_275 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_66: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_132, torch.bfloat16);  mul_tensor_132 = None
        mul_tensor_133: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_848, convert_element_type_default_66);  permute_848 = convert_element_type_default_66 = None
        sum_dim_int_list_66: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_133, [0, 1, 2], True);  mul_tensor_133 = None
        reshape_default_100: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, [128]);  sum_dim_int_list_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_134: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_973, [1, 0]);  view_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_134: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_271, rsqrt_45);  convert_element_type_271 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_67: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_134, torch.bfloat16);  mul_tensor_134 = None
        mul_tensor_135: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_853, convert_element_type_default_67);  permute_853 = convert_element_type_default_67 = None
        sum_dim_int_list_67: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_135, [0, 1, 2], True);  mul_tensor_135 = None
        reshape_default_101: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, [128]);  sum_dim_int_list_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_135: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_977, [1, 0]);  view_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_136: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_267, rsqrt_44);  convert_element_type_267 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_68: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_136, torch.bfloat16);  mul_tensor_136 = None
        mul_tensor_137: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_510, convert_element_type_default_68);  add_510 = convert_element_type_default_68 = None
        sum_dim_int_list_68: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_137, [0, 1], True);  mul_tensor_137 = None
        reshape_default_102: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, [1024]);  sum_dim_int_list_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_136: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_980, [1, 0]);  view_980 = None
        permute_default_137: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_982, [1, 0]);  view_982 = None
        permute_default_138: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_984, [1, 0]);  view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_138: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_257, rsqrt_43);  convert_element_type_257 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_69: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_138, torch.bfloat16);  mul_tensor_138 = None
        mul_tensor_139: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_515, convert_element_type_default_69);  add_515 = convert_element_type_default_69 = None
        sum_dim_int_list_69: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_139, [0, 1], True);  mul_tensor_139 = None
        reshape_default_103: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, [1024]);  sum_dim_int_list_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_139: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_987, [1, 0]);  view_987 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_140: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_104: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_140, [4, 512, -1]);  permute_default_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_105: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_104, [2048, 2048]);  reshape_default_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_141: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_993, [1, 0]);  view_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_140: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_251, rsqrt_42);  convert_element_type_251 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_70: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_140, torch.bfloat16);  mul_tensor_140 = None
        mul_tensor_141: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_880, convert_element_type_default_70);  permute_880 = convert_element_type_default_70 = None
        sum_dim_int_list_70: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_141, [0, 1, 2], True);  mul_tensor_141 = None
        reshape_default_106: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, [128]);  sum_dim_int_list_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_142: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_997, [1, 0]);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_142: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_247, rsqrt_41);  convert_element_type_247 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_71: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_142, torch.bfloat16);  mul_tensor_142 = None
        mul_tensor_143: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_885, convert_element_type_default_71);  permute_885 = convert_element_type_default_71 = None
        sum_dim_int_list_71: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_143, [0, 1, 2], True);  mul_tensor_143 = None
        reshape_default_107: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, [128]);  sum_dim_int_list_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_143: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1001, [1, 0]);  view_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_144: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_243, rsqrt_40);  convert_element_type_243 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_72: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_144, torch.bfloat16);  mul_tensor_144 = None
        mul_tensor_145: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_525, convert_element_type_default_72);  add_525 = convert_element_type_default_72 = None
        sum_dim_int_list_72: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_145, [0, 1], True);  mul_tensor_145 = None
        reshape_default_108: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, [1024]);  sum_dim_int_list_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_144: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1004, [1, 0]);  view_1004 = None
        permute_default_145: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1006, [1, 0]);  view_1006 = None
        permute_default_146: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1008, [1, 0]);  view_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_146: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_39);  convert_element_type_233 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_73: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_146, torch.bfloat16);  mul_tensor_146 = None
        mul_tensor_147: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_530, convert_element_type_default_73);  add_530 = convert_element_type_default_73 = None
        sum_dim_int_list_73: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_147, [0, 1], True);  mul_tensor_147 = None
        reshape_default_109: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, [1024]);  sum_dim_int_list_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_147: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1011, [1, 0]);  view_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_148: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_110: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_148, [4, 512, -1]);  permute_default_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_111: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_110, [2048, 2048]);  reshape_default_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_149: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1017, [1, 0]);  view_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_148: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_227, rsqrt_38);  convert_element_type_227 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_74: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_148, torch.bfloat16);  mul_tensor_148 = None
        mul_tensor_149: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_912, convert_element_type_default_74);  permute_912 = convert_element_type_default_74 = None
        sum_dim_int_list_74: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_149, [0, 1, 2], True);  mul_tensor_149 = None
        reshape_default_112: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, [128]);  sum_dim_int_list_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_150: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1021, [1, 0]);  view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_150: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_223, rsqrt_37);  convert_element_type_223 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_75: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_150, torch.bfloat16);  mul_tensor_150 = None
        mul_tensor_151: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_917, convert_element_type_default_75);  permute_917 = convert_element_type_default_75 = None
        sum_dim_int_list_75: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_151, [0, 1, 2], True);  mul_tensor_151 = None
        reshape_default_113: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, [128]);  sum_dim_int_list_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_151: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1025, [1, 0]);  view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_152: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_219, rsqrt_36);  convert_element_type_219 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_76: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_152, torch.bfloat16);  mul_tensor_152 = None
        mul_tensor_153: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_540, convert_element_type_default_76);  add_540 = convert_element_type_default_76 = None
        sum_dim_int_list_76: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_153, [0, 1], True);  mul_tensor_153 = None
        reshape_default_114: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, [1024]);  sum_dim_int_list_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_152: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1028, [1, 0]);  view_1028 = None
        permute_default_153: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1030, [1, 0]);  view_1030 = None
        permute_default_154: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1032, [1, 0]);  view_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_154: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_209, rsqrt_35);  convert_element_type_209 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_77: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_154, torch.bfloat16);  mul_tensor_154 = None
        mul_tensor_155: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_545, convert_element_type_default_77);  add_545 = convert_element_type_default_77 = None
        sum_dim_int_list_77: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_155, [0, 1], True);  mul_tensor_155 = None
        reshape_default_115: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, [1024]);  sum_dim_int_list_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_155: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1035, [1, 0]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_156: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_116: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_156, [4, 512, -1]);  permute_default_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_117: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_116, [2048, 2048]);  reshape_default_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_157: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1041, [1, 0]);  view_1041 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_156: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_203, rsqrt_34);  convert_element_type_203 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_78: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_156, torch.bfloat16);  mul_tensor_156 = None
        mul_tensor_157: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_944, convert_element_type_default_78);  permute_944 = convert_element_type_default_78 = None
        sum_dim_int_list_78: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_157, [0, 1, 2], True);  mul_tensor_157 = None
        reshape_default_118: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, [128]);  sum_dim_int_list_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_158: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1045, [1, 0]);  view_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_158: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_199, rsqrt_33);  convert_element_type_199 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_79: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_158, torch.bfloat16);  mul_tensor_158 = None
        mul_tensor_159: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_949, convert_element_type_default_79);  permute_949 = convert_element_type_default_79 = None
        sum_dim_int_list_79: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_159, [0, 1, 2], True);  mul_tensor_159 = None
        reshape_default_119: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, [128]);  sum_dim_int_list_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_159: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1049, [1, 0]);  view_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_160: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_195, rsqrt_32);  convert_element_type_195 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_80: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_160, torch.bfloat16);  mul_tensor_160 = None
        mul_tensor_161: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_555, convert_element_type_default_80);  add_555 = convert_element_type_default_80 = None
        sum_dim_int_list_80: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_161, [0, 1], True);  mul_tensor_161 = None
        reshape_default_120: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, [1024]);  sum_dim_int_list_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_160: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1052, [1, 0]);  view_1052 = None
        permute_default_161: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1054, [1, 0]);  view_1054 = None
        permute_default_162: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1056, [1, 0]);  view_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_162: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_185, rsqrt_31);  convert_element_type_185 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_81: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_162, torch.bfloat16);  mul_tensor_162 = None
        mul_tensor_163: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_560, convert_element_type_default_81);  add_560 = convert_element_type_default_81 = None
        sum_dim_int_list_81: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_163, [0, 1], True);  mul_tensor_163 = None
        reshape_default_121: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, [1024]);  sum_dim_int_list_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_163: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1059, [1, 0]);  view_1059 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_164: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_122: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_164, [4, 512, -1]);  permute_default_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_123: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_122, [2048, 2048]);  reshape_default_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_165: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1065, [1, 0]);  view_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_164: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_179, rsqrt_30);  convert_element_type_179 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_82: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_164, torch.bfloat16);  mul_tensor_164 = None
        mul_tensor_165: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_976, convert_element_type_default_82);  permute_976 = convert_element_type_default_82 = None
        sum_dim_int_list_82: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_165, [0, 1, 2], True);  mul_tensor_165 = None
        reshape_default_124: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, [128]);  sum_dim_int_list_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_166: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1069, [1, 0]);  view_1069 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_166: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_175, rsqrt_29);  convert_element_type_175 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_83: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_166, torch.bfloat16);  mul_tensor_166 = None
        mul_tensor_167: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_981, convert_element_type_default_83);  permute_981 = convert_element_type_default_83 = None
        sum_dim_int_list_83: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_167, [0, 1, 2], True);  mul_tensor_167 = None
        reshape_default_125: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, [128]);  sum_dim_int_list_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_167: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1073, [1, 0]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_168: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_171, rsqrt_28);  convert_element_type_171 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_84: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_168, torch.bfloat16);  mul_tensor_168 = None
        mul_tensor_169: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_570, convert_element_type_default_84);  add_570 = convert_element_type_default_84 = None
        sum_dim_int_list_84: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_169, [0, 1], True);  mul_tensor_169 = None
        reshape_default_126: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, [1024]);  sum_dim_int_list_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_168: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1076, [1, 0]);  view_1076 = None
        permute_default_169: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1078, [1, 0]);  view_1078 = None
        permute_default_170: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1080, [1, 0]);  view_1080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_170: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_161, rsqrt_27);  convert_element_type_161 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_85: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_170, torch.bfloat16);  mul_tensor_170 = None
        mul_tensor_171: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_575, convert_element_type_default_85);  add_575 = convert_element_type_default_85 = None
        sum_dim_int_list_85: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_171, [0, 1], True);  mul_tensor_171 = None
        reshape_default_127: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_85, [1024]);  sum_dim_int_list_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_171: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1083, [1, 0]);  view_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_172: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_128: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_172, [4, 512, -1]);  permute_default_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_129: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_128, [2048, 2048]);  reshape_default_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_173: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1089, [1, 0]);  view_1089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_172: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_155, rsqrt_26);  convert_element_type_155 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_86: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_172, torch.bfloat16);  mul_tensor_172 = None
        mul_tensor_173: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_1008, convert_element_type_default_86);  permute_1008 = convert_element_type_default_86 = None
        sum_dim_int_list_86: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_173, [0, 1, 2], True);  mul_tensor_173 = None
        reshape_default_130: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, [128]);  sum_dim_int_list_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_174: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1093, [1, 0]);  view_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_174: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_151, rsqrt_25);  convert_element_type_151 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_87: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_174, torch.bfloat16);  mul_tensor_174 = None
        mul_tensor_175: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_1013, convert_element_type_default_87);  permute_1013 = convert_element_type_default_87 = None
        sum_dim_int_list_87: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_175, [0, 1, 2], True);  mul_tensor_175 = None
        reshape_default_131: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, [128]);  sum_dim_int_list_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_175: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1097, [1, 0]);  view_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_176: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_147, rsqrt_24);  convert_element_type_147 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_88: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_176, torch.bfloat16);  mul_tensor_176 = None
        mul_tensor_177: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_585, convert_element_type_default_88);  add_585 = convert_element_type_default_88 = None
        sum_dim_int_list_88: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_177, [0, 1], True);  mul_tensor_177 = None
        reshape_default_132: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, [1024]);  sum_dim_int_list_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_176: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1100, [1, 0]);  view_1100 = None
        permute_default_177: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1102, [1, 0]);  view_1102 = None
        permute_default_178: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1104, [1, 0]);  view_1104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_178: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_137, rsqrt_23);  convert_element_type_137 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_89: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_178, torch.bfloat16);  mul_tensor_178 = None
        mul_tensor_179: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_590, convert_element_type_default_89);  add_590 = convert_element_type_default_89 = None
        sum_dim_int_list_89: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_179, [0, 1], True);  mul_tensor_179 = None
        reshape_default_133: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, [1024]);  sum_dim_int_list_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_179: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1107, [1, 0]);  view_1107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_180: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_134: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_180, [4, 512, -1]);  permute_default_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_135: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_134, [2048, 2048]);  reshape_default_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_181: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1113, [1, 0]);  view_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_180: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_131, rsqrt_22);  convert_element_type_131 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_90: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_180, torch.bfloat16);  mul_tensor_180 = None
        mul_tensor_181: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_1040, convert_element_type_default_90);  permute_1040 = convert_element_type_default_90 = None
        sum_dim_int_list_90: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_181, [0, 1, 2], True);  mul_tensor_181 = None
        reshape_default_136: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, [128]);  sum_dim_int_list_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_182: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1117, [1, 0]);  view_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_182: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, rsqrt_21);  convert_element_type_127 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_91: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_182, torch.bfloat16);  mul_tensor_182 = None
        mul_tensor_183: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_1045, convert_element_type_default_91);  permute_1045 = convert_element_type_default_91 = None
        sum_dim_int_list_91: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_183, [0, 1, 2], True);  mul_tensor_183 = None
        reshape_default_137: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, [128]);  sum_dim_int_list_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_183: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1121, [1, 0]);  view_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_184: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_20);  convert_element_type_123 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_92: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_184, torch.bfloat16);  mul_tensor_184 = None
        mul_tensor_185: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_600, convert_element_type_default_92);  add_600 = convert_element_type_default_92 = None
        sum_dim_int_list_92: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_185, [0, 1], True);  mul_tensor_185 = None
        reshape_default_138: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, [1024]);  sum_dim_int_list_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_184: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1124, [1, 0]);  view_1124 = None
        permute_default_185: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1126, [1, 0]);  view_1126 = None
        permute_default_186: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1128, [1, 0]);  view_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_186: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_19);  convert_element_type_113 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_93: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_186, torch.bfloat16);  mul_tensor_186 = None
        mul_tensor_187: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_605, convert_element_type_default_93);  add_605 = convert_element_type_default_93 = None
        sum_dim_int_list_93: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_187, [0, 1], True);  mul_tensor_187 = None
        reshape_default_139: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, [1024]);  sum_dim_int_list_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_187: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1131, [1, 0]);  view_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_188: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_140: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_188, [4, 512, -1]);  permute_default_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_141: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_140, [2048, 2048]);  reshape_default_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_189: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1137, [1, 0]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_188: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_107, rsqrt_18);  convert_element_type_107 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_94: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_188, torch.bfloat16);  mul_tensor_188 = None
        mul_tensor_189: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_1072, convert_element_type_default_94);  permute_1072 = convert_element_type_default_94 = None
        sum_dim_int_list_94: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_189, [0, 1, 2], True);  mul_tensor_189 = None
        reshape_default_142: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, [128]);  sum_dim_int_list_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_190: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1141, [1, 0]);  view_1141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_190: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_17);  convert_element_type_103 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_95: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_190, torch.bfloat16);  mul_tensor_190 = None
        mul_tensor_191: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_1077, convert_element_type_default_95);  permute_1077 = convert_element_type_default_95 = None
        sum_dim_int_list_95: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_191, [0, 1, 2], True);  mul_tensor_191 = None
        reshape_default_143: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, [128]);  sum_dim_int_list_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_191: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1145, [1, 0]);  view_1145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_192: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_99, rsqrt_16);  convert_element_type_99 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_96: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_192, torch.bfloat16);  mul_tensor_192 = None
        mul_tensor_193: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_615, convert_element_type_default_96);  add_615 = convert_element_type_default_96 = None
        sum_dim_int_list_96: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_193, [0, 1], True);  mul_tensor_193 = None
        reshape_default_144: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, [1024]);  sum_dim_int_list_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_192: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1148, [1, 0]);  view_1148 = None
        permute_default_193: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1150, [1, 0]);  view_1150 = None
        permute_default_194: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1152, [1, 0]);  view_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_194: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_89, rsqrt_15);  convert_element_type_89 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_97: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_194, torch.bfloat16);  mul_tensor_194 = None
        mul_tensor_195: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_620, convert_element_type_default_97);  add_620 = convert_element_type_default_97 = None
        sum_dim_int_list_97: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_195, [0, 1], True);  mul_tensor_195 = None
        reshape_default_145: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, [1024]);  sum_dim_int_list_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_195: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1155, [1, 0]);  view_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_196: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_146: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_196, [4, 512, -1]);  permute_default_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_147: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_146, [2048, 2048]);  reshape_default_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_197: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1161, [1, 0]);  view_1161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_196: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_14);  convert_element_type_83 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_98: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_196, torch.bfloat16);  mul_tensor_196 = None
        mul_tensor_197: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_1104, convert_element_type_default_98);  permute_1104 = convert_element_type_default_98 = None
        sum_dim_int_list_98: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_197, [0, 1, 2], True);  mul_tensor_197 = None
        reshape_default_148: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, [128]);  sum_dim_int_list_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_198: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1165, [1, 0]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_198: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_79, rsqrt_13);  convert_element_type_79 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_99: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_198, torch.bfloat16);  mul_tensor_198 = None
        mul_tensor_199: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_1109, convert_element_type_default_99);  permute_1109 = convert_element_type_default_99 = None
        sum_dim_int_list_99: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_199, [0, 1, 2], True);  mul_tensor_199 = None
        reshape_default_149: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, [128]);  sum_dim_int_list_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_199: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1169, [1, 0]);  view_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_200: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_75, rsqrt_12);  convert_element_type_75 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_100: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_200, torch.bfloat16);  mul_tensor_200 = None
        mul_tensor_201: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_630, convert_element_type_default_100);  add_630 = convert_element_type_default_100 = None
        sum_dim_int_list_100: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_201, [0, 1], True);  mul_tensor_201 = None
        reshape_default_150: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, [1024]);  sum_dim_int_list_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_200: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1172, [1, 0]);  view_1172 = None
        permute_default_201: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1174, [1, 0]);  view_1174 = None
        permute_default_202: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1176, [1, 0]);  view_1176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_202: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_65, rsqrt_11);  convert_element_type_65 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_101: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_202, torch.bfloat16);  mul_tensor_202 = None
        mul_tensor_203: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_635, convert_element_type_default_101);  add_635 = convert_element_type_default_101 = None
        sum_dim_int_list_101: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_203, [0, 1], True);  mul_tensor_203 = None
        reshape_default_151: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, [1024]);  sum_dim_int_list_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_203: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1179, [1, 0]);  view_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_204: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_152: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_204, [4, 512, -1]);  permute_default_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_153: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_152, [2048, 2048]);  reshape_default_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_205: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1185, [1, 0]);  view_1185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_204: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_59, rsqrt_10);  convert_element_type_59 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_102: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_204, torch.bfloat16);  mul_tensor_204 = None
        mul_tensor_205: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_1136, convert_element_type_default_102);  permute_1136 = convert_element_type_default_102 = None
        sum_dim_int_list_102: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_205, [0, 1, 2], True);  mul_tensor_205 = None
        reshape_default_154: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, [128]);  sum_dim_int_list_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_206: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1189, [1, 0]);  view_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_206: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_55, rsqrt_9);  convert_element_type_55 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_103: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_206, torch.bfloat16);  mul_tensor_206 = None
        mul_tensor_207: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_1141, convert_element_type_default_103);  permute_1141 = convert_element_type_default_103 = None
        sum_dim_int_list_103: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_207, [0, 1, 2], True);  mul_tensor_207 = None
        reshape_default_155: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, [128]);  sum_dim_int_list_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_207: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1193, [1, 0]);  view_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_208: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_51, rsqrt_8);  convert_element_type_51 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_104: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_208, torch.bfloat16);  mul_tensor_208 = None
        mul_tensor_209: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_645, convert_element_type_default_104);  add_645 = convert_element_type_default_104 = None
        sum_dim_int_list_104: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_209, [0, 1], True);  mul_tensor_209 = None
        reshape_default_156: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, [1024]);  sum_dim_int_list_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_208: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1196, [1, 0]);  view_1196 = None
        permute_default_209: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1198, [1, 0]);  view_1198 = None
        permute_default_210: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1200, [1, 0]);  view_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_210: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_41, rsqrt_7);  convert_element_type_41 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_105: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_210, torch.bfloat16);  mul_tensor_210 = None
        mul_tensor_211: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_650, convert_element_type_default_105);  add_650 = convert_element_type_default_105 = None
        sum_dim_int_list_105: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_211, [0, 1], True);  mul_tensor_211 = None
        reshape_default_157: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_105, [1024]);  sum_dim_int_list_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_211: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1203, [1, 0]);  view_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_212: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_158: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_212, [4, 512, -1]);  permute_default_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_159: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_158, [2048, 2048]);  reshape_default_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_213: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1209, [1, 0]);  view_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_212: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_35, rsqrt_6);  convert_element_type_35 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_106: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_212, torch.bfloat16);  mul_tensor_212 = None
        mul_tensor_213: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_1168, convert_element_type_default_106);  permute_1168 = convert_element_type_default_106 = None
        sum_dim_int_list_106: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_213, [0, 1, 2], True);  mul_tensor_213 = None
        reshape_default_160: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, [128]);  sum_dim_int_list_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_214: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1213, [1, 0]);  view_1213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_214: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_31, rsqrt_5);  convert_element_type_31 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_107: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_214, torch.bfloat16);  mul_tensor_214 = None
        mul_tensor_215: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_1173, convert_element_type_default_107);  permute_1173 = convert_element_type_default_107 = None
        sum_dim_int_list_107: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_215, [0, 1, 2], True);  mul_tensor_215 = None
        reshape_default_161: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, [128]);  sum_dim_int_list_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_215: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1217, [1, 0]);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_216: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, rsqrt_4);  convert_element_type_27 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_108: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_216, torch.bfloat16);  mul_tensor_216 = None
        mul_tensor_217: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_660, convert_element_type_default_108);  add_660 = convert_element_type_default_108 = None
        sum_dim_int_list_108: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_217, [0, 1], True);  mul_tensor_217 = None
        reshape_default_162: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, [1024]);  sum_dim_int_list_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_216: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1220, [1, 0]);  view_1220 = None
        permute_default_217: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1222, [1, 0]);  view_1222 = None
        permute_default_218: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_1224, [1, 0]);  view_1224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_218: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_3);  convert_element_type_17 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_109: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_218, torch.bfloat16);  mul_tensor_218 = None
        mul_tensor_219: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_665, convert_element_type_default_109);  add_665 = convert_element_type_default_109 = None
        sum_dim_int_list_109: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_219, [0, 1], True);  mul_tensor_219 = None
        reshape_default_163: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_109, [1024]);  sum_dim_int_list_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_219: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1227, [1, 0]);  view_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_220: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_164: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_default_220, [4, 512, -1]);  permute_default_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_165: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_164, [2048, 2048]);  reshape_default_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_221: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1233, [1, 0]);  view_1233 = None
        reshape_default_166: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_586, [4, 512, 1024]);  mm_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_220: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_11, rsqrt_2);  convert_element_type_11 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_110: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_220, torch.bfloat16);  mul_tensor_220 = None
        mul_tensor_221: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_1200, convert_element_type_default_110);  permute_1200 = convert_element_type_default_110 = None
        sum_dim_int_list_110: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_221, [0, 1, 2], True);  mul_tensor_221 = None
        reshape_default_167: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, [128]);  sum_dim_int_list_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_222: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_1237, [1, 0]);  view_1237 = None
        reshape_default_168: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_588, [4, 512, 1024]);  mm_588 = None
        add_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(reshape_default_166, reshape_default_168);  reshape_default_166 = reshape_default_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_222: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, rsqrt_1);  convert_element_type_7 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_111: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_222, torch.bfloat16);  mul_tensor_222 = None
        mul_tensor_223: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_1205, convert_element_type_default_111);  permute_1205 = convert_element_type_default_111 = None
        sum_dim_int_list_111: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_223, [0, 1, 2], True);  mul_tensor_223 = None
        reshape_default_169: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, [128]);  sum_dim_int_list_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_223: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_1241, [1, 0]);  view_1241 = None
        reshape_default_170: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_590, [4, 512, 1024]);  mm_590 = None
        add_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_170);  add_tensor = reshape_default_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_224: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_112: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_225: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_112, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_113: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_225, torch.bfloat16);  mul_tensor_225 = None
        mul_tensor_226: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, convert_element_type_default_113);  add_tensor_1 = convert_element_type_default_113 = None
        sum_dim_int_list_112: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_226, [0, 1], True);  mul_tensor_226 = None
        reshape_default_171: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, [1024]);  sum_dim_int_list_112 = None
        convert_element_type_default_114: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_224, torch.float32);  mul_tensor_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_227: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_114, convert_element_type_default_112)
        mul_tensor_228: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_114, rsqrt);  convert_element_type_default_114 = None
        sum_dim_int_list_113: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_227, [2], True);  mul_tensor_227 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_113, -0.5);  sum_dim_int_list_113 = None
        mul_tensor_229: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 1024]" = torch.ops.aten.expand.default(mul_tensor_229, [4, 512, 1024]);  mul_tensor_229 = None
        div_scalar: "f32[4, 512, 1024]" = torch.ops.aten.div.Scalar(expand_default, 1024);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_112, 1.0);  convert_element_type_default_112 = None
        mul_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_230: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_230);  mul_tensor_228 = mul_tensor_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_115: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_667, convert_element_type_default_115);  add_667 = convert_element_type_default_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:392 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        convert_element_type_default_116: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default_58, convert_element_type_default_116);  unsqueeze_default = full_default_58 = convert_element_type_default_116 = None
        full_default: "f32[151936, 1024]" = torch.ops.aten.full.default([151936, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[151936, 1024]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        convert_element_type_default_117: "bf16[151936, 1024]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.bfloat16);  index_put_default = None
        add_tensor_4: "bf16[151936, 1024]" = torch.ops.aten.add.Tensor(mm_197, convert_element_type_default_117);  mm_197 = convert_element_type_default_117 = None
        return (reshape_default, permute_default, permute_default_1, permute_default_2, reshape_default_1, permute_default_3, reshape_default_3, permute_default_5, reshape_default_4, permute_default_6, reshape_default_5, permute_default_7, reshape_default_6, permute_default_8, permute_default_9, permute_default_10, reshape_default_7, permute_default_11, reshape_default_9, permute_default_13, reshape_default_10, permute_default_14, reshape_default_11, permute_default_15, reshape_default_12, permute_default_16, permute_default_17, permute_default_18, reshape_default_13, permute_default_19, reshape_default_15, permute_default_21, reshape_default_16, permute_default_22, reshape_default_17, permute_default_23, reshape_default_18, permute_default_24, permute_default_25, permute_default_26, reshape_default_19, permute_default_27, reshape_default_21, permute_default_29, reshape_default_22, permute_default_30, reshape_default_23, permute_default_31, reshape_default_24, permute_default_32, permute_default_33, permute_default_34, reshape_default_25, permute_default_35, reshape_default_27, permute_default_37, reshape_default_28, permute_default_38, reshape_default_29, permute_default_39, reshape_default_30, permute_default_40, permute_default_41, permute_default_42, reshape_default_31, permute_default_43, reshape_default_33, permute_default_45, reshape_default_34, permute_default_46, reshape_default_35, permute_default_47, reshape_default_36, permute_default_48, permute_default_49, permute_default_50, reshape_default_37, permute_default_51, reshape_default_39, permute_default_53, reshape_default_40, permute_default_54, reshape_default_41, permute_default_55, reshape_default_42, permute_default_56, permute_default_57, permute_default_58, reshape_default_43, permute_default_59, reshape_default_45, permute_default_61, reshape_default_46, permute_default_62, reshape_default_47, permute_default_63, reshape_default_48, permute_default_64, permute_default_65, permute_default_66, reshape_default_49, permute_default_67, reshape_default_51, permute_default_69, reshape_default_52, permute_default_70, reshape_default_53, permute_default_71, reshape_default_54, permute_default_72, permute_default_73, permute_default_74, reshape_default_55, permute_default_75, reshape_default_57, permute_default_77, reshape_default_58, permute_default_78, reshape_default_59, permute_default_79, reshape_default_60, permute_default_80, permute_default_81, permute_default_82, reshape_default_61, permute_default_83, reshape_default_63, permute_default_85, reshape_default_64, permute_default_86, reshape_default_65, permute_default_87, reshape_default_66, permute_default_88, permute_default_89, permute_default_90, reshape_default_67, permute_default_91, reshape_default_69, permute_default_93, reshape_default_70, permute_default_94, reshape_default_71, permute_default_95, reshape_default_72, permute_default_96, permute_default_97, permute_default_98, reshape_default_73, permute_default_99, reshape_default_75, permute_default_101, reshape_default_76, permute_default_102, reshape_default_77, permute_default_103, reshape_default_78, permute_default_104, permute_default_105, permute_default_106, reshape_default_79, permute_default_107, reshape_default_81, permute_default_109, reshape_default_82, permute_default_110, reshape_default_83, permute_default_111, reshape_default_84, permute_default_112, permute_default_113, permute_default_114, reshape_default_85, permute_default_115, reshape_default_87, permute_default_117, reshape_default_88, permute_default_118, reshape_default_89, permute_default_119, reshape_default_90, permute_default_120, permute_default_121, permute_default_122, reshape_default_91, permute_default_123, reshape_default_93, permute_default_125, reshape_default_94, permute_default_126, reshape_default_95, permute_default_127, reshape_default_96, permute_default_128, permute_default_129, permute_default_130, reshape_default_97, permute_default_131, reshape_default_99, permute_default_133, reshape_default_100, permute_default_134, reshape_default_101, permute_default_135, reshape_default_102, permute_default_136, permute_default_137, permute_default_138, reshape_default_103, permute_default_139, reshape_default_105, permute_default_141, reshape_default_106, permute_default_142, reshape_default_107, permute_default_143, reshape_default_108, permute_default_144, permute_default_145, permute_default_146, reshape_default_109, permute_default_147, reshape_default_111, permute_default_149, reshape_default_112, permute_default_150, reshape_default_113, permute_default_151, reshape_default_114, permute_default_152, permute_default_153, permute_default_154, reshape_default_115, permute_default_155, reshape_default_117, permute_default_157, reshape_default_118, permute_default_158, reshape_default_119, permute_default_159, reshape_default_120, permute_default_160, permute_default_161, permute_default_162, reshape_default_121, permute_default_163, reshape_default_123, permute_default_165, reshape_default_124, permute_default_166, reshape_default_125, permute_default_167, reshape_default_126, permute_default_168, permute_default_169, permute_default_170, reshape_default_127, permute_default_171, reshape_default_129, permute_default_173, reshape_default_130, permute_default_174, reshape_default_131, permute_default_175, reshape_default_132, permute_default_176, permute_default_177, permute_default_178, reshape_default_133, permute_default_179, reshape_default_135, permute_default_181, reshape_default_136, permute_default_182, reshape_default_137, permute_default_183, reshape_default_138, permute_default_184, permute_default_185, permute_default_186, reshape_default_139, permute_default_187, reshape_default_141, permute_default_189, reshape_default_142, permute_default_190, reshape_default_143, permute_default_191, reshape_default_144, permute_default_192, permute_default_193, permute_default_194, reshape_default_145, permute_default_195, reshape_default_147, permute_default_197, reshape_default_148, permute_default_198, reshape_default_149, permute_default_199, reshape_default_150, permute_default_200, permute_default_201, permute_default_202, reshape_default_151, permute_default_203, reshape_default_153, permute_default_205, reshape_default_154, permute_default_206, reshape_default_155, permute_default_207, reshape_default_156, permute_default_208, permute_default_209, permute_default_210, reshape_default_157, permute_default_211, reshape_default_159, permute_default_213, reshape_default_160, permute_default_214, reshape_default_161, permute_default_215, reshape_default_162, permute_default_216, permute_default_217, permute_default_218, reshape_default_163, permute_default_219, reshape_default_165, permute_default_221, reshape_default_167, permute_default_222, reshape_default_169, permute_default_223, reshape_default_171, add_tensor_4)


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
