"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train
Pattern hash: 59083955eeac
Shape hash: c488c227
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
    def forward(self, tangents_1: "f32[]", convert_element_type: "f32[]", constant_pad_nd: "i64[64, 513]", _shape_param_0, _shape_param_1, view_269: "f32[64, 512, 30522]", _shape_param_2, amax_12: "f32[32768, 1]", log: "f32[32768, 1]", _shape_param_3, tangents_2: "f32[64, 512, 30522]", _shape_param_4, primals_5: "f32[30522, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_tensor: "i64[64, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_default: "i64[64, 512]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        reshape_default: "i64[32768]" = torch.ops.aten.reshape.default(clone_default, [-1]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_default: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, 1);  reshape_default = None
        ne_scalar: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32768, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[32768, 30522]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[32768, 30522]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self_1: "f32[32768, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self_2: "f32[32768, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        reshape_default_2: "f32[32768, 30522]" = torch.ops.aten.reshape.default(view_269, _shape_param_2);  view_269 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_tensor: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_12);  reshape_default_2 = amax_12 = None
        sub_tensor_1: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log);  sub_tensor = log = None
        exp_default: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        reshape_default_3: "f32[64, 512, 30522]" = torch.ops.aten.reshape.default(sub_tensor_2, _shape_param_3);  sub_tensor_2 = _shape_param_3 = None
        add_tensor: "f32[64, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, reshape_default_3);  tangents_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        reshape_default_4: "f32[32768, 30522]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        permute_default: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_default_1: "f32[30522, 128]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        constant_pad_nd_default: "f32[32768, 30524]" = torch.ops.aten.constant_pad_nd.default(reshape_default_4, [0, 2, 0, 0]);  reshape_default_4 = None
        constant_pad_nd_default_1: "f32[30524, 128]" = torch.ops.aten.constant_pad_nd.default(permute_default_1, [0, 0, 0, 2]);  permute_default_1 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [64, 513], dtype=torch.int64, device='cuda'),
    [1, 30522],  # _shape_param_0
    [32768, 30522],  # _shape_param_1
    torch.randn(1000210430, dtype=torch.float32, device='cuda').as_strided([64, 512, 30522], [15628288, 30524, 1]),  # view_269
    [-1, 30522],  # _shape_param_2
    torch.randn([32768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 1], dtype=torch.float32, device='cuda'),
    [64, 512, 30522],  # _shape_param_3
    torch.randn([64, 512, 30522], dtype=torch.float32, device='cuda'),
    [32768, 30522],  # _shape_param_4
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
