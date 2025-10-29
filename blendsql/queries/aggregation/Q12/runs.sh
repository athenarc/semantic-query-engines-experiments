#!/bin/bash
sizes=(1000 10000 30000)
models_ollama=("gemma3-32k:latest" "llama8-32k:latest")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
  for model in "${models_ollama[@]}"; do
    echo "Running with -s $size and -m ollama/$model"
    python blendsql/queries/aggregation/Q12/q12.py  --wandb -s $size -m $model -p ollama
  done

  for model in "${models_vllm[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/aggregation/Q12/q12.py  --wandb -s $size -m $model -p vllm
  done
done