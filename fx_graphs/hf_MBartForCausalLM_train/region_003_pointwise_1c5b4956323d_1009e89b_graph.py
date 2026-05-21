class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1024]", primals_2: "f32[1026, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:107 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:109 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(primals_2, add_tensor);  primals_2 = add_tensor = None
        return embedding_default
