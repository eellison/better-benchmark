class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 128][128, 1]cuda:0", arg1_1: "f32[2050, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:96 in forward, code: position_ids = position_ids + self.offset
        add: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:102 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        view: "i64[128][1]cuda:0" = torch.ops.aten.reshape.default(add, [-1]);  add = None
        index: "f32[128, 1024][1024, 1]cuda:0" = torch.ops.aten.index.Tensor(arg1_1, [view]);  arg1_1 = view = None
        view_1: "f32[1, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(index, [1, 128, 1024]);  index = None
        return (view_1,)
