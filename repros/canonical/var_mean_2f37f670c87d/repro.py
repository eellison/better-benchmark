"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=var_mean, ranges=[], reduction_ranges=[]
#   origins: ['aten.var_mean.correction']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[50272, 768]", arg0_1: "i64[4, 512]", cumsum: "f32[4, 512]", full: "f32[4, 512]", arg2_1: "f16[2050, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:591 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f16[4, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:614 in forward, code: position_ids = (position_ids * attention_mask - 1).long()
        mul_tensor: "f32[4, 512]" = torch.ops.aten.mul.Tensor(cumsum, full);  cumsum = full = None
        sub_tensor: "f32[4, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, 1);  mul_tensor = None
        convert_element_type_default: "i64[4, 512]" = torch.ops.prims.convert_element_type.default(sub_tensor, torch.int64);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:76 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[4, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, 2);  convert_element_type_default = None
        embedding_default_1: "f16[4, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:623 in forward, code: hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)
        add_tensor_1: "f16[4, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:252 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_default_1: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True);  convert_element_type_default_1 = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([50272, 768], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    torch.randn([4, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2050, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
