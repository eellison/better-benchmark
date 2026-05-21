"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer
Pattern hash: 4c5a39635a32
Shape hash: 0f05ac4a
"""
_shapes_config = "(T([30522, 128], f32), T([256, 128], i64), T([512, 384], f32), S([32768, 384]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[30522, 128]", arg0_1: "i64[256, 128]", arg3_1: "f32[512, 384]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:113 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[256, 128, 128]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:125 in forward, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
        slice_tensor: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding_default, 1, 1, 9223372036854775807)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 0, 0, 1, 0, 0], 0.0);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:127 in forward, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
        slice_tensor_1: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding_default, 1, 0, -1)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_1: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_1, [0, 0, 1, 0, 0, 0], 0.0);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:123 in forward, code: inputs_embeds = torch.cat(
        cat_default: "f32[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd_default, embedding_default, constant_pad_nd_default_1], 2);  constant_pad_nd_default = embedding_default = constant_pad_nd_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default: "f32[32768, 384]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None
        permute_default: "f32[384, 512]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 30522, [256, 128], dtype=torch.int64, device='cuda'),
    torch.randn([512, 384], dtype=torch.float32, device='cuda'),
    [32768, 384],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
