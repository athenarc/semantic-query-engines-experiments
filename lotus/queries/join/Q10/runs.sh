#!/bin/bash
sizes=(5 10 20)
models_ollama=("gemma3:12b" "llama3.1:8b")
# models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
  for model in "${models_ollama[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/join/Q10/default.py  --wandb -s $size -m $model -p ollama
  done
done

  for model in "${models_vllm[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/join/Q10/default.py  --wandb -s $size -m $model -p vllm
  done
done