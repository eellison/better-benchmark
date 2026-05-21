class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 4096, 256]", arg1_1: "f16[1, 4096, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:343 in torch_dynamo_resume_in_forward_at_342, code: embeddings = embeddings + position_embeddings
        add: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (add,)
