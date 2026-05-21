"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: 62006d12db51
Shape hash: 14f4aaaf
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
_shapes_config = "(T([128, 32000], f32), T([32, 128], f32), T([1], i64))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[128, 32000]", arg0_1: "f32[32, 128]", arg2_1: "i64[1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:72 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue[:, ptr : ptr + batch_size] = keys.T
        slice_tensor: "f32[128, 32]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 32)
        permute_default: "f32[128, 32]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        copy_default: "f32[128, 32]" = torch.ops.aten.copy.default(slice_tensor, permute_default);  slice_tensor = permute_default = None
        slice_scatter_default: "f32[128, 32000]" = torch.ops.aten.slice_scatter.default(arg1_1, copy_default, 1, 0, 32);  copy_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:75 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue_ptr[0] = ptr
        select_int: "i64[]" = torch.ops.aten.select.int(arg2_1, 0, 0)
        full_default: "i64[]" = torch.ops.aten.full.default([], 32, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_1: "i64[]" = torch.ops.aten.copy.default(select_int, full_default);  select_int = full_default = None
        select_scatter_default: "i64[1]" = torch.ops.aten.select_scatter.default(arg2_1, copy_default_1, 0, 0);  copy_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:72 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue[:, ptr : ptr + batch_size] = keys.T
        copy__default: "f32[128, 32000]" = torch.ops.aten.copy_.default(arg1_1, slice_scatter_default);  arg1_1 = slice_scatter_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:75 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue_ptr[0] = ptr
        copy__default_1: "i64[1]" = torch.ops.aten.copy_.default(arg2_1, select_scatter_default);  arg2_1 = select_scatter_default = None
        return (copy__default, copy__default_1)



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
