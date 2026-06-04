# sum_sum_6f518556b3ea
## Compile: 40.0us, Oracle: 25.5us, Gap: 1.57x
## Diagnosis: scatter_reduce (structured pool-upsample backward)
## Root cause: Inductor materializes the zero-fill as_strided_scatter -> as_strided -> expand -> div adaptive-average-pool backward gradient tensor as ordinary tensor work and schedules the hard-ReLU6 mask, two sibling channel reductions, and dependent full BN-backward epilogue as separate generic consumers, instead of recognizing the structured scatter/expand as a gated pool-gradient source.
## Fix path: Add a structured average-pool-backward scatter/expand lowering that maps pooled-gradient source elements directly into gated channel reductions and emits the dependent BN-backward output tensor in one fused template.
## Status: design_todo
