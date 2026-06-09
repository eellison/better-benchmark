# pointwise_27183a793fcd

## Classification: AT_FLOOR

## Current Result

- Family: `stencil_scatter`
- Oracle path: `repros/canonical/pointwise_27183a793fcd/oracle_stencil_scatter.py`
- Correctness: PASS
- Oracle: `36.58 us`
- `torch.compile coordinate_descent_tuning=True`: `37.66 us`
- Ratio: 1.03
- Status: `at_floor`

## Diagnosis

The gap is within noise (3%). Inductor handles this stencil scatter pattern at essentially the same speed as the oracle. The pattern involves [384, 30522] and [30522, 128] output tensors with stencil indexing.

## Config exploration results
- Ratio 1.03 is within measurement noise -- no improvement needed.
