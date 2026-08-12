#!/bin/bash
# Watchdog: keeps the mover_fillin driver alive (idempotent resume), logs progress.
cd /tmp/scratch_space/better_benchmark
LOG=results/pytorch_landing/A1_commit_walk/_mover_watchdog.log
DRIVERLOG=results/pytorch_landing/A1_commit_walk/mover_fillin_driver.log
EXPECT=daa79cd25ca9a80bfd65799394cf4255d6be75a6
done_marker=results/pytorch_landing/A1_commit_walk/_mover_DONE
for iter in $(seq 1 200); do
  # count completed mover states
  n=0
  while read sha; do
    [ -z "$sha" ] && continue
    [ -f "results/pytorch_landing/A1_commit_walk/states/$sha.json" ] && n=$((n+1))
  done < results/pytorch_landing/A1_commit_walk/mover_fillin.txt
  ts=$(date '+%H:%M:%S')
  # isolation check
  iso=$(git -C /tmp/pytorch-work rev-parse HEAD 2>/dev/null)
  [ "$iso" = "$EXPECT" ] && isook="ISO-OK" || isook="ISO-BROKEN($iso)"
  # driver alive?
  if pgrep -f "walk_driver.py --from-file results/pytorch_landing/A1_commit_walk/mover_fillin.txt" >/dev/null; then
    alive="alive"
  else
    alive="DEAD"
  fi
  echo "[$ts] iter=$iter done=$n/25 driver=$alive $isook" >> $LOG
  if [ "$n" -ge 25 ]; then
    echo "[$ts] ALL 25 COMPLETE" >> $LOG
    touch $done_marker
    break
  fi
  if [ "$alive" = "DEAD" ]; then
    echo "[$ts] driver dead, relaunching (idempotent resume)" >> $LOG
    nohup python3 results/pytorch_landing/A1_commit_walk/walk_driver.py --from-file results/pytorch_landing/A1_commit_walk/mover_fillin.txt >> $DRIVERLOG 2>&1 &
    sleep 5
  fi
  sleep 120
done
