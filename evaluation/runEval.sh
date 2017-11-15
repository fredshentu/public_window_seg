#!/bin/bash
for i in $(seq -f "%g" 0 4)
do
  python metrics.py -mp savedOutputs/501_msk_thr0.50_head_"$i".npy
  python metrics.py -mp savedOutputs/503_msk_thr0.50_head_"$i".npy
done
for i in $(seq -f "%g" 0 4)
do
  python metrics.py -mp savedOutputs/501_msk_thr0.60_head_"$i".npy
  python metrics.py -mp savedOutputs/503_msk_thr0.60_head_"$i".npy
done
for i in $(seq -f "%g" 0 4)
do
  python metrics.py -mp savedOutputs/501_msk_thr0.70_head_"$i".npy
  python metrics.py -mp savedOutputs/503_msk_thr0.70_head_"$i".npy
done
