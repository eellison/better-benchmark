class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 5, 1, 28, 28]", arg1_1: "i64[32, 5]", arg2_1: "f32[32, 75, 1, 28, 28]", arg3_1: "i64[32, 75]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/meta.py:61 in forward, code: return self.finetunning(x_spt[0], y_spt[0], x_qry[0], y_qry[0])
        select: "f32[5, 1, 28, 28]" = torch.ops.aten.select.int(arg0_1, 0, 0);  arg0_1 = None
        select_1: "i64[5]" = torch.ops.aten.select.int(arg1_1, 0, 0);  arg1_1 = None
        select_2: "f32[75, 1, 28, 28]" = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        select_3: "i64[75]" = torch.ops.aten.select.int(arg3_1, 0, 0);  arg3_1 = None
        return (select, select_1, select_2, select_3)
