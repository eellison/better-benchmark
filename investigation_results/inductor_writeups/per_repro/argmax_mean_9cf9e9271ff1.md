# argmax_mean_9cf9e9271ff1

## Summary
- **Model**: torchbench_hf_Reformer (LSH routing)
- **Pattern**: Reformer LSH routing with argmax(cat([x,-x])), sort, gather, RMS normalization
- **Classification**: NEW_PATTERN (oracle broken)
- **Status**: Oracle has implementation bug - cannot benchmark

## Oracle Error
```
TypeError: oracle_reformer_lsh_routing() takes 3 positional arguments but 20 were given
```

The oracle's `oracle_forward` function signature does not match the number of inputs produced by `make_inputs()` for this particular shape configuration. The oracle was likely written for a different shape variant (argmax_mean_180a59791f51 has 3 inputs) but this repro has 20 inputs.

## Root Cause (from sibling repro argmax_mean_180a59791f51)

This is the same Reformer LSH routing pattern as argmax_mean_180a59791f51:
1. Compute argmax(cat([x, -x])) to get 0..127 bucket assignment
2. Sort by bucket*4096 + position
3. Gather from sorted indices, compute RMS scaling, cyclic concatenation

The main Inductor inefficiency is materializing the cat([x, -x]) buffer when only the argmax is needed. The oracle computes signed-abs argmax without the allocation.

## Conclusion
Oracle is broken for this specific repro's input count. The pattern and root cause are identical to argmax_mean_180a59791f51 (see that writeup for full analysis). The oracle needs to be regenerated to handle this repro's 20-input signature.
