# Scope-Mismatch Oracle Audit

These oracle files are diagnosis/prototype artifacts, not valid floor measurements,
unless they are rewritten so `--bench` covers the same computation scope as the
compiled `repro.py` region. Keep their CSV `oracle_path` blank until then.

| Repro | Oracle file | Issue | Classification | Queue action |
| --- | --- | --- | --- | --- |
| `amax_sum_55ae6a130879` | `oracle_online_softmax.py` | Measures softmax/dropout prototype, not the full fused dropout plus permute repro scope. | `SCHEDULER_FUSION` | Rewrite full-scope oracle. |
| `amax_sum_4c524f75213e` | `oracle_online_softmax.py` | Softmax is only a small part of the full Longformer attention assembly. | `NEW_PATTERN` | Do not wire as a floor. |
| `amax_sum_68fe981b18dd` | `oracle_online_softmax.py` | Same Longformer scope mismatch. | `NEW_PATTERN` | Do not wire as a floor. |
| `amax_sum_87e1fb077f24` | `oracle_online_softmax.py` | Same Longformer scope mismatch. | `NEW_PATTERN` | Do not wire as a floor. |
| `amax_sum_amax_2a81770def44` | `oracle_online_softmax.py` | Misses dropout and position-bias full repro scope. | `SCHEDULER_FUSION` | Rewrite full-scope oracle. |
| `amax_sum_amax_68fa105ccaf0` | `oracle_online_softmax.py` | Misses dropout and position-bias full repro scope. | `SCHEDULER_FUSION` | Rewrite full-scope oracle. |
| `sum_617cd87647d6` | `oracle_multi_output_reduction.py` | Measures only the sum/FMA core, not the full graph. | `SCATTER_REDUCE` | Rewrite full-scope oracle. |
| `sum_sum_0930bd38b7d1` | `oracle_multi_output_reduction.py` | Measures a subset; current compiled repro is already faster. | `SCATTER_REDUCE` | Rewrite full-scope oracle or leave actionable. |
| `sum_e00c7291b6ee` | `oracle_multi_output_reduction.py` | Oracle is slower than compile. | `COOPERATIVE_SPLIT_K` | Rewrite before counting as a floor. |
| `sum_b691b8dad90a` | `oracle_multi_output_reduction.py` | Oracle is slower than compile. | `COOPERATIVE_SPLIT_K` | Rewrite before counting as a floor. |
