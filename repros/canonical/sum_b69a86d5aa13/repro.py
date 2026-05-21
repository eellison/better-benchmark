"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: b69a86d5aa13
Shape hash: c6889d40
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
_shapes_config = "(T([64, 5, 2, 426888], f32), S([-1, -1, -1, 426888]), S([-1, -1, 2, -1]), S([16, 4, 4, 2, 382788]), S([-1, -1, -1, 2, 382788]), S([64, 4, 2, 382788]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 5, 2, 426888]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[4]" = torch.ops.prims.inductor_seeds.default(4, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:105 in forward, code: th.rand(groups, group_size, streams, 1, 1, device=device), dim=1
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default: "f32[16, 4, 4, 1, 1]" = torch.ops.prims.inductor_random.default([16, 4, 4, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:104 in forward, code: permutations = th.argsort(
        sort_default = torch.ops.aten.sort.default(inductor_random_default, 1);  inductor_random_default = None
        getitem: "i64[16, 4, 4, 1, 1]" = sort_default[1];  sort_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/__init__.py:29 in forward, code: sources = streams[:, 1:]
        slice_tensor: "f32[64, 4, 2, 426888]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807);  arg0_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:65 in forward, code: signs = th.randint(
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_randint_default: "i64[64, 4, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 2, [64, 4, 1, 1], inductor_lookup_seed_default_1);  inductor_lookup_seed_default_1 = None
        convert_element_type_default: "f32[64, 4, 1, 1]" = torch.ops.prims.convert_element_type.default(inductor_randint_default, torch.float32);  inductor_randint_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:68 in forward, code: wav = wav * (2 * signs - 1)
        mul_tensor: "f32[64, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2);  convert_element_type_default = None
        sub_tensor: "f32[64, 4, 1, 1]" = torch.ops.aten.sub.Tensor(mul_tensor, 1);  mul_tensor = None
        mul_tensor_1: "f32[64, 4, 2, 426888]" = torch.ops.aten.mul.Tensor(slice_tensor, sub_tensor);  slice_tensor = sub_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:48 in forward, code: left = th.randint(
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_randint_default_1: "i64[64, 4, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 2, [64, 4, 1, 1], inductor_lookup_seed_default_2);  inductor_lookup_seed_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:51 in forward, code: left = left.expand(-1, -1, -1, time)
        expand_default: "i64[64, 4, 1, 426888]" = torch.ops.aten.expand.default(inductor_randint_default_1, _shape_param_0);  inductor_randint_default_1 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:53 in forward, code: wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        gather_default: "f32[64, 4, 1, 426888]" = torch.ops.aten.gather.default(mul_tensor_1, 2, expand_default)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:52 in forward, code: right = 1 - left
        sub_tensor_1: "i64[64, 4, 1, 426888]" = torch.ops.aten.sub.Tensor(1, expand_default);  expand_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:53 in forward, code: wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        gather_default_1: "f32[64, 4, 1, 426888]" = torch.ops.aten.gather.default(mul_tensor_1, 2, sub_tensor_1);  mul_tensor_1 = sub_tensor_1 = None
        cat_default: "f32[64, 4, 2, 426888]" = torch.ops.aten.cat.default([gather_default, gather_default_1], 2);  gather_default = gather_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:35 in forward, code: indexes = th.arange(length, device=wav.device)
        iota_default: "i64[382788]" = torch.ops.prims.iota.default(382788, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:28 in forward, code: offsets = th.randint(
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_randint_default_2: "i64[64, 4, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 44100, [64, 4, 1, 1], inductor_lookup_seed_default_3);  inductor_lookup_seed_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:34 in forward, code: offsets = offsets.expand(-1, -1, channels, -1)
        expand_default_1: "i64[64, 4, 2, 1]" = torch.ops.aten.expand.default(inductor_randint_default_2, _shape_param_1);  inductor_randint_default_2 = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:36 in forward, code: wav = wav.gather(3, indexes + offsets)
        add_tensor: "i64[64, 4, 2, 382788]" = torch.ops.aten.add.Tensor(iota_default, expand_default_1);  iota_default = expand_default_1 = None
        gather_default_2: "f32[64, 4, 2, 382788]" = torch.ops.aten.gather.default(cat_default, 3, add_tensor);  cat_default = add_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:103 in forward, code: wav = wav.view(groups, group_size, streams, channels, time)
        reshape_default: "f32[16, 4, 4, 2, 382788]" = torch.ops.aten.reshape.default(gather_default_2, _shape_param_2);  gather_default_2 = _shape_param_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:107 in forward, code: wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
        expand_default_2: "i64[16, 4, 4, 2, 382788]" = torch.ops.aten.expand.default(getitem, _shape_param_3);  getitem = _shape_param_3 = None
        gather_default_3: "f32[16, 4, 4, 2, 382788]" = torch.ops.aten.gather.default(reshape_default, 1, expand_default_2);  reshape_default = expand_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:108 in forward, code: wav = wav.view(batch, streams, channels, time)
        reshape_default_1: "f32[64, 4, 2, 382788]" = torch.ops.aten.reshape.default(gather_default_3, _shape_param_4);  gather_default_3 = _shape_param_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/__init__.py:31 in forward, code: mix = sources.sum(dim=1)
        sum_dim_int_list: "f32[64, 2, 382788]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [1]);  reshape_default_1 = None
        return sum_dim_int_list



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
