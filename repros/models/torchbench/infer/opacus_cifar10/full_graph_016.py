class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[256, 256, 3, 3]", arg1_1: "Sym(s16)", arg2_1: "Sym(s82)", arg3_1: "f32[64, 256, s16, s82]"):
        # No stacktrace found for following nodes
        convolution: "f32[64, 256, s16, s82]" = torch.ops.aten.convolution.default(arg3_1, arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg3_1 = arg0_1 = None
        return (convolution,)
