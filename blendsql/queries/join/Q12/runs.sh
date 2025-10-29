#!/bin/bash
sizes=(5 10 20)
models_ollama=("gemma3-32k:latest")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
  for model in "${models_ollama[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/join/Q12/q12.py  --wandb -s $size -m $model -p ollama
  done

  for model in "${models_vllm[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/join/Q12/q12.py  --wandb -s $size -m $model -p vllm
  done
done