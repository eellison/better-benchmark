"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer
Pattern hash: 763aa63c6b58
Shape hash: 906ff581
"""
_shapes_config = "(T([192, 512, 64], f32), T([1536, 1536], f32), S([-1, 24, 512, 64]), S([8, 512, -1]), S([4096, 1536]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_47: "f32[192, 512, 64]", arg380_1: "f32[1536, 1536]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        reshape_default: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, _shape_param_0);  bmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_default: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_default: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        reshape_default_1: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[4096, 1536]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    [-1, 24, 512, 64],  # _shape_param_0
    [8, 512, -1],  # _shape_param_1
    [4096, 1536],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
