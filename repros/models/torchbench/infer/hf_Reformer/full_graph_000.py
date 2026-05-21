class GraphModule(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:167 in _get_least_common_mult_chunk_len, code: return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
        _tensor_constant0: "i64[]" = self._tensor_constant0;  _tensor_constant0 = None
        _tensor_constant1: "i64[]" = self._tensor_constant1;  _tensor_constant1 = None
        full_default: "i64[]" = torch.ops.aten.full.default([], 64, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 64, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        gcd: "i64[]" = torch.ops.aten.gcd.default(full_default, full_default_1)
        eq: "b8[]" = torch.ops.aten.eq.Scalar(gcd, 0);  gcd = eq = None
        scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'));  scalar_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:2013 in forward, code: input_shape[-1] % least_common_mult_chunk_length != 0
        _tensor_constant2: "i64[]" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:167 in _get_least_common_mult_chunk_len, code: return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 64, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "i64[]" = torch.ops.prims.div.default(full_default, full_default_2);  full_default = full_default_2 = None
        mul: "i64[]" = torch.ops.aten.mul.Tensor(div, full_default_1);  div = full_default_1 = None
        abs_1: "i64[]" = torch.ops.aten.abs.default(mul);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:2013 in forward, code: input_shape[-1] % least_common_mult_chunk_length != 0
        remainder: "i64[]" = torch.ops.aten.remainder.Scalar_Tensor(4096, abs_1)
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        eq_1: "b8[]" = torch.ops.aten.eq.Tensor(remainder, full_default_3);  remainder = full_default_3 = None
        bitwise_not: "b8[]" = torch.ops.aten.bitwise_not.default(eq_1);  eq_1 = None
        return (bitwise_not, abs_1)
