"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_infer
Pattern hash: cfb0f315bc5e
Shape hash: 94542781
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_79: "f32[16384, 384]", _shape_param_0, add_135: "f32[32, 512, 768]", _shape_param_1, arg261_1: "f32[384, 768]", _shape_param_2, arg263_1: "f32[384, 768]", _shape_param_3, arg272_1: "f32[384, 768]", convolution_23: "f32[32, 384, 512]", arg267_1: "f32[384, 1]", _shape_param_4, arg270_1: "f32[54, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_79, _shape_param_0);  addmm_79 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_135, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[768, 384]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_135, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[768, 384]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        reshape_default_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_135, _shape_param_3);  add_135 = _shape_param_3 = None
        permute_default_2: "f32[768, 384]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_tensor: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, arg267_1);  convolution_23 = arg267_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_3: "f32[32, 512, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor: "f32[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute_default_3, reshape_default);  permute_default_3 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_default: "f32[32, 512, 384]" = torch.ops.aten.clone.default(mul_tensor, memory_format = torch.contiguous_format);  mul_tensor = None
        reshape_default_4: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        permute_default_4: "f32[384, 54]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2, reshape_default_4, permute_default_4)


def _default_make_inputs():
    return [
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    [32, 512, 384],  # _shape_param_0
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_1
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_2
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_3
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 1], dtype=torch.float32, device='cuda'),
    [16384, 384],  # _shape_param_4
    torch.randn([54, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
