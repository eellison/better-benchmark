class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", arg1_1: "Sym(s16)", arg2_1: "Sym(s82)", arg3_1: "bf16[64, 64, s16, s82][64*s16*s82, s16*s82, s82, 1]cuda:0"):
        # No stacktrace found for following nodes
        convolution: "bf16[64, 128, (((s16 - 1)//2)) + 1, (((s82 - 1)//2)) + 1][128*((((s16 - 1)//2)) + 1)*((((s82 - 1)//2)) + 1), ((((s16 - 1)//2)) + 1)*((((s82 - 1)//2)) + 1), (((s82 - 1)//2)) + 1, 1]cuda:0" = torch.ops.aten.convolution.default(arg3_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg3_1 = arg0_1 = None
        return (convolution,)
