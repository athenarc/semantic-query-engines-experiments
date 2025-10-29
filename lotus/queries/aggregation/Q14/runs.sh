#!/bin/bash
sizes=(10000 30000)
models_ollama=("gemma3:12b" "llama3.1:8b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
  for model in "${models_ollama[@]}"; do
    echo "Running with -s $size and -m ollama/$model"
    python lotus/queries/aggregation/Q14/q14.py  --wandb -s $size -m $model -p ollama
  done

  for model in "${models_vllm[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/aggregation/Q14/q14.py  --wandb -s $size -m $model -p vllm
  done
done