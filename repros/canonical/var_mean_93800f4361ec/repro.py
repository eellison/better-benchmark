"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: 93800f4361ec
Shape hash: 9a06cf2a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[30522, 768]", arg0_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg2_1: "i64[1, 512]", arg4_1: "f32[1024, 768]", arg5_1: "f32[1024, 768]", arg6_1: "f32[1024, 768]", arg7_1: "f32[1024, 768]", arg8_1: "f32[2, 768]"):
        # No stacktrace found for following nodes
        full_default: "i64[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "i64[1, 512, 4]" = torch.ops.aten.full.default([1, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        embedding_default: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None
        select_int: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 0)
        embedding_default_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_int)
        select_int_1: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 1)
        embedding_default_3: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_int_1)
        select_int_2: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 2)
        embedding_default_4: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_int_2);  arg4_1 = None
        select_int_3: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 3);  full_default_1 = None
        embedding_default_5: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_int_3);  arg5_1 = None
        sub_tensor: "i64[1, 512]" = torch.ops.aten.sub.Tensor(select_int_3, select_int_1);  select_int_3 = select_int_1 = None
        embedding_default_6: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, sub_tensor);  arg6_1 = sub_tensor = None
        sub_tensor_1: "i64[1, 512]" = torch.ops.aten.sub.Tensor(select_int_2, select_int);  select_int_2 = select_int = None
        embedding_default_7: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg7_1, sub_tensor_1);  arg7_1 = sub_tensor_1 = None
        embedding_default_8: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg8_1, full_default);  arg8_1 = full_default = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_3);  add_tensor_1 = embedding_default_3 = None
        add_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, embedding_default_4);  add_tensor_2 = embedding_default_4 = None
        add_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, embedding_default_5);  add_tensor_3 = embedding_default_5 = None
        add_tensor_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, embedding_default_6);  add_tensor_4 = embedding_default_6 = None
        add_tensor_6: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, embedding_default_7);  add_tensor_5 = embedding_default_7 = None
        add_tensor_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_6, embedding_default_8);  add_tensor_6 = embedding_default_8 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_7, [2], correction = 0, keepdim = True);  add_tensor_7 = None
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 30522, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
