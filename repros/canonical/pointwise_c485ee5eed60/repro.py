"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_training
Pattern hash: c485ee5eed60
Shape hash: 29a6843d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_94: "f32[512, 8, 1]", add_261: "f32[512, 8, 1024]", getitem_95: "f32[512, 8, 1]", primals_361: "f32[1024]", primals_362: "f32[1024]", inductor_seeds_default: "i64[99]", _shape_param_0, primals_2: "f32[32000, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_tensor: "f32[512, 8, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_default: "f32[512, 8, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 8, 1024]" = torch.ops.aten.sub.Tensor(add_261, getitem_95);  add_261 = getitem_95 = None
        mul_tensor: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_361);  mul_tensor = primals_361 = None
        add_tensor_1: "f32[512, 8, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_362);  mul_tensor_1 = primals_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1177 in forward, code: output = self.dropout(output_g if output_g is not None else output_h)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 98);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 8, 1024]" = torch.ops.prims.inductor_random.default([512, 8, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 8, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_3: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1180 in forward, code: output = output.permute(1, 0, 2).contiguous()
        permute_default: "f32[8, 512, 1024]" = torch.ops.aten.permute.default(mul_tensor_3, [1, 0, 2]);  mul_tensor_3 = None
        clone_default: "f32[8, 512, 1024]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        reshape_default: "f32[4096, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default_1: "f32[1024, 32000]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([512, 8, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 8, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 8, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [99], dtype=torch.int64, device='cuda'),
    [4096, 1024],  # _shape_param_0
    torch.randn([32000, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
