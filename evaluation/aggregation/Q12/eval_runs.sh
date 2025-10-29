#!/bin/bash

sizes=(1000 10000 30000)

for size in "${sizes[@]}"; do
    echo "Evaluating with -s $size and -m $model"
    python evaluation/aggregation/Q12/eval_scripts/q12_eval.py -s $size
done