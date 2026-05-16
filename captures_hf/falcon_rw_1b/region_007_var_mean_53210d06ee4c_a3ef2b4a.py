"""
Standalone repro captured via capture_hook.
Label: falcon_rw_1b
Pattern hash: 53210d06ee4c
Shape hash: a3ef2b4a
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[50304, 2048]", arg0_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:744 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f16[4, 512, 2048]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:598 in forward, code: attention_layernorm_out = self.input_layernorm(hidden_states)
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32);  embedding_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([50304, 2048], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
