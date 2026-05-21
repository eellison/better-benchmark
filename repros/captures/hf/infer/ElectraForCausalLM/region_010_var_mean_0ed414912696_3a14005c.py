"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_infer
Pattern hash: 0ed414912696
Shape hash: 3a14005c
"""
_shapes_config = "(T([30522, 128], f32), T([64, 512], i64), T([1, 512], i64), T([1, 512], i64), T([2, 128], f32), T([512, 128], f32), T([128], f32), T([128], f32), T([256, 128], f32), S([64, 512]), S([32768, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg4_1: "f32[30522, 128]", arg1_1: "i64[64, 512]", arg3_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg5_1: "f32[2, 128]", arg6_1: "f32[512, 128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[256, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:108 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, arg1_1, 0);  arg4_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:101 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(arg3_1, [1, -1]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:102 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather_default: "i64[1, 512]" = torch.ops.aten.gather.default(expand_default, 1, arg2_1);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:103 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default_1: "i64[64, 512]" = torch.ops.aten.expand.default(gather_default, _shape_param_0);  gather_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:109 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, expand_default_1);  arg5_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:110 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_tensor: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:112 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg6_1, arg2_1);  arg6_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:113 in forward, code: embeddings = embeddings + position_embeddings
        add_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg7_1);  mul_tensor = arg7_1 = None
        add_tensor_3: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg8_1);  mul_tensor_1 = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        reshape_default: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        permute_default: "f32[128, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 30522, [64, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([2, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128], dtype=torch.float32, device='cuda'),
    [64, 512],  # _shape_param_0
    [32768, 128],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
