"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 1a267d618444
Shape hash: 026baa31
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_3: "f32[4, 768]", _shape_param_0, _shape_param_1, primals_183: "f32[768]", mul_126: "f32[4, 7, 7, 768]", div_35: "f32[4, 7, 7, 1]", lt_21: "b8[4, 1, 1, 1]", _shape_param_2, primals_181: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:612 in forward, code: x = self.flatten(x)
        reshape_default: "f32[4, 768, 1, 1]" = torch.ops.aten.reshape.default(mm_3, _shape_param_0);  mm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:611 in forward, code: x = self.avgpool(x)
        squeeze_dim: "f32[4, 768, 1]" = torch.ops.aten.squeeze.dim(reshape_default, 3);  reshape_default = None
        squeeze_dim_1: "f32[4, 768]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[3072]" = torch.ops.aten.full.default([3072], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[3072]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [4, 768], [768, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[4, 768, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [4, 768, 1, 1], [768, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[4, 768, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[4, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute_default: "f32[4, 7, 7, 768]" = torch.ops.aten.permute.default(div_scalar, [0, 2, 3, 1]);  div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:609 in forward, code: x = self.norm(x)
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(permute_default, primals_183);  permute_default = primals_183 = None
        mul_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_126);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_126, sum_dim_int_list_1);  mul_126 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_tensor_1);  div_35 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_21, torch.float32);  lt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_5: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, div_tensor);  mul_tensor_4 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        clone_default: "f32[4, 7, 7, 768]" = torch.ops.aten.clone.default(mul_tensor_5, memory_format = torch.contiguous_format);  mul_tensor_5 = None
        reshape_default_1: "f32[196, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        permute_default_2: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    [4, 768, 1, 1],  # _shape_param_0
    [4, 768, 7, 7],  # _shape_param_1
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    [196, 768],  # _shape_param_2
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
