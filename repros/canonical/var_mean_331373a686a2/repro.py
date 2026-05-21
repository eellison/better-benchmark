"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_infer
Pattern hash: 331373a686a2
Shape hash: f911c416
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
_shapes_config = "(T([50257, 768], f16), T([1, 1024], i64, gen=Index(50257)), T([1024, 768], f16), T([768], f16), T([768], f16), S([-1, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[50257, 768]", arg0_1: "i64[1, 1024]", arg2_1: "f16[1024, 768]", arg3_1: "f16[768]", arg4_1: "f16[768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding_default: "f16[1, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_default_1: "f16[1, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze_default);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_tensor_1: "f16[1, 1024, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_default: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_2: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg3_1);  mul_tensor = arg3_1 = None
        add_tensor_3: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        convert_element_type_default_1: "f16[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default: "f16[1024, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_0);  convert_element_type_default_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_tensor: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze_default, 1, 0, 1)
        sub_tensor_1: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat_default: "i64[1, 1025]" = torch.ops.aten.cat.default([sub_tensor_1, unsqueeze_default], -1);  sub_tensor_1 = unsqueeze_default = None
        slice_tensor_1: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 1025)
        slice_tensor_2: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 1024);  cat_default = None
        sub_tensor_2: "i64[1, 1024]" = torch.ops.aten.sub.Tensor(slice_tensor_1, slice_tensor_2);  slice_tensor_1 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne_scalar: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(sub_tensor_2, 1);  sub_tensor_2 = None
        return (ne_scalar, reshape_default)



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
