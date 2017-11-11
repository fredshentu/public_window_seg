#!/bin/bash
for i in $(seq -f "%03g" 1 23)
do
  python metrics.py -mp savedOutputs/$i.npy
done
python metrics.py -mp savedOutputs/deepmask_result_NMS.npy
