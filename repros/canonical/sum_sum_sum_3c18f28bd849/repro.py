"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: 3c18f28bd849
Shape hash: 1508ffcb
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 8, 371372], f32), T([64, 128, 92842], f32), T([64, 64, 92844], f32), T([64, 64, 92844], f32), T([64, 256, 23210], f32), T([64, 128, 23212], f32), T([64, 128, 23212], f32), T([64, 512, 5802], f32), T([64, 256, 5804], f32), T([64, 256, 5804], f32), T([64, 1024, 1450], f32), T([64, 512, 1452], f32), T([64, 512, 1452], f32), T([64, 2048, 362], f32), T([64, 1024, 364], f32), T([64, 1024, 364], f32), T([64, 4096, 90], f32))"

class Repro(torch.nn.Module):
    def forward(self, view_1: "f32[64, 8, 371372]", cat: "f32[64, 128, 92842]", getitem_3: "f32[64, 64, 92844]", where: "f32[64, 64, 92844]", cat_1: "f32[64, 256, 23210]", getitem_9: "f32[64, 128, 23212]", where_1: "f32[64, 128, 23212]", cat_2: "f32[64, 512, 5802]", getitem_15: "f32[64, 256, 5804]", where_2: "f32[64, 256, 5804]", cat_3: "f32[64, 1024, 1450]", getitem_21: "f32[64, 512, 1452]", where_3: "f32[64, 512, 1452]", cat_4: "f32[64, 2048, 362]", getitem_27: "f32[64, 1024, 364]", where_4: "f32[64, 1024, 364]", cat_5: "f32[64, 4096, 90]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_dim_int_list: "f32[8]" = torch.ops.aten.sum.dim_IntList(view_1, [0, 2]);  view_1 = None
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(cat, [0, 2]);  cat = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default: "f32[64, 64, 95696]" = torch.ops.aten.full.default([64, 64, 95696], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[64, 64, 95696]" = torch.ops.aten.slice_scatter.default(full_default, getitem_3, 2, 1426, -1426);  full_default = getitem_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_dim_int_list_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2]);  where = None
        sum_dim_int_list_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(cat_1, [0, 2]);  cat_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_1: "f32[64, 128, 23923]" = torch.ops.aten.full.default([64, 128, 23923], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_1: "f32[64, 128, 23923]" = torch.ops.aten.slice_scatter.default(full_default_1, getitem_9, 2, 355, -356);  full_default_1 = getitem_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_dim_int_list_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2]);  where_1 = None
        sum_dim_int_list_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(cat_2, [0, 2]);  cat_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_2: "f32[64, 256, 5979]" = torch.ops.aten.full.default([64, 256, 5979], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_2: "f32[64, 256, 5979]" = torch.ops.aten.slice_scatter.default(full_default_2, getitem_15, 2, 87, -88);  full_default_2 = getitem_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_dim_int_list_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2]);  where_2 = None
        sum_dim_int_list_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(cat_3, [0, 2]);  cat_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_3: "f32[64, 512, 1493]" = torch.ops.aten.full.default([64, 512, 1493], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_3: "f32[64, 512, 1493]" = torch.ops.aten.slice_scatter.default(full_default_3, getitem_21, 2, 20, -21);  full_default_3 = getitem_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_dim_int_list_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2]);  where_3 = None
        sum_dim_int_list_9: "f32[2048]" = torch.ops.aten.sum.dim_IntList(cat_4, [0, 2]);  cat_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_4: "f32[64, 1024, 372]" = torch.ops.aten.full.default([64, 1024, 372], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_4: "f32[64, 1024, 372]" = torch.ops.aten.slice_scatter.default(full_default_4, getitem_27, 2, 4, -4);  full_default_4 = getitem_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_dim_int_list_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2]);  where_4 = None
        sum_dim_int_list_11: "f32[4096]" = torch.ops.aten.sum.dim_IntList(cat_5, [0, 2]);  cat_5 = None
        return (sum_dim_int_list, sum_dim_int_list_1, slice_scatter_default, sum_dim_int_list_2, sum_dim_int_list_3, slice_scatter_default_1, sum_dim_int_list_4, sum_dim_int_list_5, slice_scatter_default_2, sum_dim_int_list_6, sum_dim_int_list_7, slice_scatter_default_3, sum_dim_int_list_8, sum_dim_int_list_9, slice_scatter_default_4, sum_dim_int_list_10, sum_dim_int_list_11)



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
