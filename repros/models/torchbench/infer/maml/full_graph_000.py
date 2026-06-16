class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32, 5, 1, 28, 28][3920, 784, 784, 28, 1]cuda:0", arg1_1: "i64[32, 5][5, 1]cuda:0", arg2_1: "bf16[32, 75, 1, 28, 28][58800, 784, 784, 28, 1]cuda:0", arg3_1: "i64[32, 75][75, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/meta.py:61 in forward, code: return self.finetunning(x_spt[0], y_spt[0], x_qry[0], y_qry[0])
        select: "bf16[5, 1, 28, 28][784, 784, 28, 1]cuda:0" = torch.ops.aten.select.int(arg0_1, 0, 0);  arg0_1 = None
        select_1: "i64[5][1]cuda:0" = torch.ops.aten.select.int(arg1_1, 0, 0);  arg1_1 = None
        select_2: "bf16[75, 1, 28, 28][784, 784, 28, 1]cuda:0" = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        select_3: "i64[75][1]cuda:0" = torch.ops.aten.select.int(arg3_1, 0, 0);  arg3_1 = None
        return (select, select_1, select_2, select_3)
