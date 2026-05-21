class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 4096]", arg1_1: "f16[320, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:324 in forward, code: position_ids = torch.arange(
        iota: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:327 in forward, code: position_ids = position_ids.unsqueeze(0).expand(input_shape)
        unsqueeze: "i64[1, 4096]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        expand: "i64[1, 4096]" = torch.ops.aten.expand.default(unsqueeze, [1, 4096]);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:330 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f16[1, 4096, 256]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        return (expand, embedding)
