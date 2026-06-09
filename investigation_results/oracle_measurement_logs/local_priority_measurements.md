# Local priority oracle measurements

Timestamp: 2026-06-03T11:18:54+00:00

## sum_18262b26934c
```bash
python repros/canonical/sum_18262b26934c/oracle_maxpool_direct_reduce.py --warmup 3 --rep 10 --no-append 
```
```text
oracle_us=579.456 impl=triton shape=N512,C64,OH55,OW55,IH111,IW111 block=1024 warps=4
```
exit_code=0

## sum_sum_sum_f90d684d32cb
```bash
python repros/canonical/sum_sum_sum_f90d684d32cb/oracle_structured_upsample_reduce.py --warmup 3 --rep 10 --no-append 
```
```text
oracle_us=93505.686 impl=torch-direct device=cuda warmup=3 rep=10
```
exit_code=0

## sum_sum_sum_70d71fcb0d68
```bash
python repros/canonical/sum_sum_sum_70d71fcb0d68/oracle_multi_output_reduction.py --warmup 3 --rep 10 --no-append 
```
```text
oracle_us=6579.374 impl=torch-direct device=cuda warmup=3 rep=10
```
exit_code=0

