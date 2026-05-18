"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: f158b8504153
Shape hash: fcf93ca2
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
    def forward(self, mm_default_2: "f32[16384, 50260]", _shape_param_0, rsqrt_12: "f32[32, 512, 1]", view_71: "f32[16384, 3072]", view_69: "f32[16384, 768]", rsqrt_11: "f32[32, 512, 1]", view_61: "f32[16384, 768]", rsqrt_10: "f32[32, 512, 1]", view_59: "f32[16384, 3072]", view_57: "f32[16384, 768]", rsqrt_9: "f32[32, 512, 1]", view_49: "f32[16384, 768]", rsqrt_8: "f32[32, 512, 1]", view_47: "f32[16384, 3072]", view_45: "f32[16384, 768]", rsqrt_7: "f32[32, 512, 1]", view_37: "f32[16384, 768]", rsqrt_6: "f32[32, 512, 1]", view_35: "f32[16384, 3072]", view_33: "f32[16384, 768]", rsqrt_5: "f32[32, 512, 1]", view_25: "f32[16384, 768]", rsqrt_4: "f32[32, 512, 1]", view_23: "f32[16384, 3072]", view_21: "f32[16384, 768]", rsqrt_3: "f32[32, 512, 1]", view_13: "f32[16384, 768]", rsqrt_2: "f32[32, 512, 1]", view_11: "f32[16384, 3072]", view_9: "f32[16384, 768]", rsqrt_1: "f32[32, 512, 1]", view_1: "f32[16384, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        slice_tensor: "f32[16384, 50257]" = torch.ops.aten.slice.Tensor(mm_default_2, 1, 0, -3);  mm_default_2 = None
        reshape_default: "f32[32, 512, 50257]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_tensor: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_71, [1, 0]);  view_71 = None
        permute_default_1: "f32[768, 16384]" = torch.ops.aten.permute.default(view_69, [1, 0]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_2: "f32[768, 16384]" = torch.ops.aten.permute.default(view_61, [1, 0]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_3: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_59, [1, 0]);  view_59 = None
        permute_default_4: "f32[768, 16384]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_3: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_5: "f32[768, 16384]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_4: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_6: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_47, [1, 0]);  view_47 = None
        permute_default_7: "f32[768, 16384]" = torch.ops.aten.permute.default(view_45, [1, 0]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_5: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_8: "f32[768, 16384]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_6: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_9: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_35, [1, 0]);  view_35 = None
        permute_default_10: "f32[768, 16384]" = torch.ops.aten.permute.default(view_33, [1, 0]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_7: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_11: "f32[768, 16384]" = torch.ops.aten.permute.default(view_25, [1, 0]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_8: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_12: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_23, [1, 0]);  view_23 = None
        permute_default_13: "f32[768, 16384]" = torch.ops.aten.permute.default(view_21, [1, 0]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_9: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_14: "f32[768, 16384]" = torch.ops.aten.permute.default(view_13, [1, 0]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_10: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_15: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_11, [1, 0]);  view_11 = None
        permute_default_16: "f32[768, 16384]" = torch.ops.aten.permute.default(view_9, [1, 0]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_11: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_17: "f32[768, 16384]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        return (reshape_default, div_tensor, permute_default, permute_default_1, div_tensor_1, permute_default_2, div_tensor_2, permute_default_3, permute_default_4, div_tensor_3, permute_default_5, div_tensor_4, permute_default_6, permute_default_7, div_tensor_5, permute_default_8, div_tensor_6, permute_default_9, permute_default_10, div_tensor_7, permute_default_11, div_tensor_8, permute_default_12, permute_default_13, div_tensor_9, permute_default_14, div_tensor_10, permute_default_15, permute_default_16, div_tensor_11, permute_default_17)


def _default_make_inputs():
    return [
    torch.randn([16384, 50260], dtype=torch.float32, device='cuda'),
    [32, 512, 50257],  # _shape_param_0
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
