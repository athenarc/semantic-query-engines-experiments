#!/bin/bash

sizes=(1000 10000 50000)

for size in "${sizes[@]}"; do
    echo "Evaluating with -s $size and -m $model"
    python evaluation/aggregation/Q11/eval_scripts/q11_eval.py -s $size
done