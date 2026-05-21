"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: b8fa3afdfbed
Shape hash: b857f3e4
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
_shapes_config = "(T([32], i64, gen=Index(32)), T([32, 3, 224, 224], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[32]", arg1_1: "f32[32, 3, 224, 224]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:97 in torch_dynamo_resume_in__batch_shuffle_ddp_at_94, code: idx_unshuffle = torch.argsort(idx_shuffle)
        sort_default = torch.ops.aten.sort.default(arg0_1)
        getitem: "i64[32]" = sort_default[1];  sort_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:101 in torch_dynamo_resume_in__batch_shuffle_ddp_at_94, code: idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        reshape_default: "i64[1, 32]" = torch.ops.aten.reshape.default(arg0_1, [1, -1]);  arg0_1 = None
        select_int: "i64[32]" = torch.ops.aten.select.int(reshape_default, 0, 0);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:103 in torch_dynamo_resume_in__batch_shuffle_ddp_at_94, code: return x_gather[idx_this], idx_unshuffle
        index_tensor: "f32[32, 3, 224, 224]" = torch.ops.aten.index.Tensor(arg1_1, [select_int]);  arg1_1 = select_int = None
        return (getitem, index_tensor)



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
