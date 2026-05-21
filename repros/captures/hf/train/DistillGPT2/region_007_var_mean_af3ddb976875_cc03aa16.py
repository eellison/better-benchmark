"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: af3ddb976875
Shape hash: cc03aa16
"""
_shapes_config = "(T([50257, 768], f32), T([32, 512], i64), T([1024, 768], f32), T([768], f32), T([768], f32), S([32, -1]), S([-1, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[50257, 768]", primals_1: "i64[32, 512]", primals_3: "f32[1024, 768]", primals_4: "f32[768]", primals_5: "f32[768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, unsqueeze_default);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_tensor: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand_default, 1, 0, 1)
        sub_tensor: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat_default: "i64[32, 513]" = torch.ops.aten.cat.default([sub_tensor, expand_default], -1);  sub_tensor = expand_default = None
        slice_tensor_1: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 512)
        slice_tensor_2: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 513);  cat_default = None
        sub_tensor_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(slice_tensor_2, slice_tensor_1);  slice_tensor_2 = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne_scalar: "b8[32, 512]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[13]" = torch.ops.prims.inductor_seeds.default(13, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, getitem_1);  mul_tensor_1 = getitem_1 = None
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_4);  mul_tensor_2 = primals_4 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_5);  mul_tensor_3 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        return (ne_scalar, reshape_default)



def make_inputs():
    return [
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50257, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [32, -1],  # _shape_param_0
    [-1, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
