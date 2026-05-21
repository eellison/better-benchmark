class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[64, 3, 32, 32]"):
        # No stacktrace found for following nodes
        convolution: "f32[64, 64, 16, 16]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        return (convolution,)
