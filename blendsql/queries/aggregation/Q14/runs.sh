#!/bin/bash
models_ollama=("gemma3:12b" "llama3.1:8b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for model in "${models_ollama[@]}"; do
    echo "Running with -s $size and -m ollama/$model"
    python blendsql/queries/aggregation/Q14/q14.py  --wandb -m $model -p ollama
done

for model in "${models_vllm[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/aggregation/Q14/q14.py  --wandb -m $model -p vllm
done
