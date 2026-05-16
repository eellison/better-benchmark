Goal: improve generic heuristics and especially per-hardware , as well as improve productivity, and benchmarking artifacts. 

Sub goal 1:

faster iteration on kernels , from the kernels in triton bench, the kernels in the genai benchmark in ptyorch, and the kernels from huggingface, timm, and torchbench

Here, the idea would be to serialize, all of these kernels, particular for reducitons, as fx graph runnable, stand alone repros. you might want to disable split reductions, or , identify, what the logical grouping of reductions is.  and save them as stand alone repros that can be executed. wed want to deduplicate same kernels. Its possible we can do this by just, collecting the origins, of our unique fused reductions scheduler nodes, and then , looking at fx graph to reconstruct. 

Sub goal 2: 

To see, for all of these kernels, how far away, inductor is from speed of light, as measured by a memcopy kernel of the same size. both with default heuristics, and with coordinate descent tuning. 

Sub goal 3 :

Once we have a good iteration loop, improve heuristics.

You probably want to first work on a workflow of small model -> fuse -> see fused reduction scheduler nodes -> saving fx grpah -> . then run hf models. then run all of our benchmarks in tritonbench. 

If youre having difficulty you can ask me