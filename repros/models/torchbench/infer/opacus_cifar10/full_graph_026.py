class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[10, 512][512, 1]cuda:0", arg1_1: "bf16[10][1]cuda:0", arg2_1: "bf16[64, 512][512, 1]cuda:0"):
        # No stacktrace found for following nodes
        permute: "bf16[512, 10][1, 512]cuda:0" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "bf16[64, 10][10, 1]cuda:0" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        return (addmm,)
