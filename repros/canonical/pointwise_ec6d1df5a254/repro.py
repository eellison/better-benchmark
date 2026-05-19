"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_train
Pattern hash: ec6d1df5a254
Shape hash: 39d01969
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
    def forward(self, view_271: "f32[16384, 30522]", getitem_135: "f32[32, 12, 512, 64]", primals_16: "f32[768, 768]", getitem_134: "f32[32, 12, 512, 64]", primals_14: "f32[768, 768]", getitem_133: "f32[32, 12, 512, 64]", primals_12: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        permute_default: "f32[30522, 16384]" = torch.ops.aten.permute.default(view_271, [1, 0]);  view_271 = None
        constant_pad_nd_default: "f32[30524, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 2]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
        clone_default: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_4: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        reshape_default_2: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_2);  permute_default_4 = _shape_param_2 = None
        clone_default_1: "f32[32, 512, 768]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        permute_default_5: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        clone_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_4: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_4);  clone_default_2 = _shape_param_4 = None
        reshape_default_5: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_default_9: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (constant_pad_nd_default, reshape_default_1, permute_default_3, reshape_default_3, permute_default_6, reshape_default_5, permute_default_9)


def _default_make_inputs():
    return [
    torch.randn([16384, 30522], dtype=torch.float32, device='cuda'),
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([32, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_135
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([32, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_134
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([32, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_133
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [32, 512, 768],  # _shape_param_0
    [16384, 768],  # _shape_param_1
    [32, 512, 768],  # _shape_param_2
    [16384, 768],  # _shape_param_3
    [32, 512, 768],  # _shape_param_4
    [16384, 768],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
