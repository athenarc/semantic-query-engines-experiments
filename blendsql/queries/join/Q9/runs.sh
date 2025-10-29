#!/bin/bash
sizes=(20 50 100)
model_ollama=("llama3.1:8b" "gemma3:12b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
  for model in "${models_ollama[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/join/Q9/q9.py  --wandb -s $size -m $model -p ollama
  done

  for model in "${models_vllm[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/join/Q9/q9.py  --wandb -s $size -m $model -p vllm
  done
done