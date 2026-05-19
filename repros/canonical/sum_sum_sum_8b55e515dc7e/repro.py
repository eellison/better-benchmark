"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_train
Pattern hash: 8b55e515dc7e
Shape hash: d798b5b2
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
    def forward(self, tangents_1: "f32[64, 256, 1024]", mul_10: "f32[64, 256, 1024]", view_22: "f32[16384, 1024]", view_25: "f32[16384, 4096]", add_10: "f32[64, 256, 1024]", mul_3: "f32[64, 256, 1024]", view_28: "f32[16384, 1024]", view_41: "f32[16384, 1024]", primals_1: "f32[64, 256, 1024]", mm_6: "f32[16384, 1024]", mul_33: "f32[64, 256, 1024]", view_44: "f32[16384, 1024]", mm_8: "f32[16384, 1024]", view_47: "f32[16384, 1024]", mm_10: "f32[16384, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_10);  mul_10 = None
        sum_dim_int_list: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[1024, 16384]" = torch.ops.aten.permute.default(view_22, [1, 0])
        sum_dim_int_list_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        reshape_default: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_default_1: "f32[4096, 16384]" = torch.ops.aten.permute.default(view_25, [1, 0])
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_tensor_1: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(add_10, mul_3);  mul_3 = None
        sum_dim_int_list_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_10, [0, 1]);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_2: "f32[1024, 16384]" = torch.ops.aten.permute.default(view_28, [1, 0])
        sum_dim_int_list_6: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        reshape_default_2: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_default_3: "f32[1024, 16384]" = torch.ops.aten.permute.default(view_41, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default_3: "f32[16384, 1024]" = torch.ops.aten.reshape.default(primals_1, _shape_param_3);  primals_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        sum_dim_int_list_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
        reshape_default_4: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_4);  sum_dim_int_list_7 = _shape_param_4 = None
        reshape_default_5: "f32[64, 256, 1024]" = torch.ops.aten.reshape.default(mm_6, _shape_param_5);  mm_6 = _shape_param_5 = None
        add_tensor: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(mul_33, reshape_default_5);  mul_33 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        permute_default_4: "f32[1024, 16384]" = torch.ops.aten.permute.default(view_44, [1, 0])
        sum_dim_int_list_8: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
        reshape_default_6: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_6);  sum_dim_int_list_8 = _shape_param_6 = None
        reshape_default_7: "f32[64, 256, 1024]" = torch.ops.aten.reshape.default(mm_8, _shape_param_7);  mm_8 = _shape_param_7 = None
        add_tensor_1: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_7);  add_tensor = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        permute_default_5: "f32[1024, 16384]" = torch.ops.aten.permute.default(view_47, [1, 0])
        sum_dim_int_list_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
        reshape_default_8: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_8);  sum_dim_int_list_9 = _shape_param_8 = None
        reshape_default_9: "f32[64, 256, 1024]" = torch.ops.aten.reshape.default(mm_10, _shape_param_9);  mm_10 = _shape_param_9 = None
        add_tensor_2: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_9);  add_tensor_1 = reshape_default_9 = None
        return (sum_dim_int_list, sum_dim_int_list_1, permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_2, reshape_default_2, permute_default_3, reshape_default_3, reshape_default_4, permute_default_4, reshape_default_6, permute_default_5, reshape_default_8, add_tensor_2)


def _default_make_inputs():
    return [
    torch.randn([64, 256, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 256, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([64, 256, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 256, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 256, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 256, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_0
    [4096],  # _shape_param_1
    [1024],  # _shape_param_2
    [16384, 1024],  # _shape_param_3
    [1024],  # _shape_param_4
    [64, 256, 1024],  # _shape_param_5
    [1024],  # _shape_param_6
    [64, 256, 1024],  # _shape_param_7
    [1024],  # _shape_param_8
    [64, 256, 1024],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
