class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0", arg1_1: "bf16[64, 3, 32, 32][3072, 1024, 32, 1]cuda:0"):
        # No stacktrace found for following nodes
        convolution: "bf16[64, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        return (convolution,)
