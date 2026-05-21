"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 6bd4b7f8de4c
Shape hash: 056c6e6e
"""
_shapes_config = "(T([8192, 512], f32), T([64, 1024, 1024], f32), T([8, 8, 1024, 1024], f32, stride=(8388608, 1, 8192, 8)), T([64], i64), T([8, 1024, 512], f32), T([512], f32), T([512, 512], f32), T([512, 512], f32), S([8, 1024, 512]), S([8, 1024, -1, 64]), S([8, 8, 1024, 1024]), S([8, 8, 1024, 1024]), S([64, 1024, 1024]), S([8, 8, 1024, 64]), S([64, 1024, 64]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_32: "f32[8192, 512]", bmm_10: "f32[64, 1024, 1024]", add_7: "f32[8, 8, 1024, 1024]", inductor_seeds_default: "i64[64]", embedding: "f32[8, 1024, 512]", primals_54: "f32[512]", primals_55: "f32[512, 512]", primals_56: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_32, _shape_param_0);  mm_32 = _shape_param_0 = None
        reshape_default_1: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, _shape_param_2);  bmm_10 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_2, add_7);  reshape_default_2 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_default: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor);  gt_scalar = div_tensor = None
        mul_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        reshape_default_3: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26);  inductor_seeds_default = None
        inductor_random_default_1: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar_1, embedding);  gt_scalar_1 = embedding = None
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_3, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_3, rsqrt_default);  mul_tensor_3 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_54, mul_tensor_4);  primals_54 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        reshape_default_5: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_7);  mul_tensor_5 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        return (reshape_default_3, reshape_default_4, permute_default_1, reshape_default_5, permute_default_2)



def make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn(67108864, dtype=torch.float32, device='cuda').as_strided([8, 8, 1024, 1024], [8388608, 1, 8192, 8]),  # add_7
    torch.randint(0, 64, [64], dtype=torch.int64, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_0
    [8, 1024, -1, 64],  # _shape_param_1
    [8, 8, 1024, 1024],  # _shape_param_2
    [8, 8, 1024, 1024],  # _shape_param_3
    [64, 1024, 1024],  # _shape_param_4
    [8, 8, 1024, 64],  # _shape_param_5
    [64, 1024, 64],  # _shape_param_6
    [8192, 512],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
