class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1191 in forward, code: is_index_global_attn = attention_mask > 0
        gt: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(arg0_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1194 in forward, code: is_global_attn = is_index_global_attn.flatten().any().item()
        view: "b8[8192][1]cuda:0" = torch.ops.aten.reshape.default(gt, [8192])
        any_1: "b8[][]cuda:0" = torch.ops.aten.any.default(view);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1190 in forward, code: is_index_masked = attention_mask < 0
        lt: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(arg0_1, 0);  arg0_1 = None
        return (any_1, lt, gt)
