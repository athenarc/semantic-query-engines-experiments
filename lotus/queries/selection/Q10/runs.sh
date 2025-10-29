#!/bin/bash
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for model in "${models_ollama[@]}"; do
    echo "Running Q10 default with -m $model"
    python lotus/queries/selection/Q10/default.py --wandb -m $model -p ollama
done

for model in "${models_vllm[@]}"; do
    echo "Running Q10 default with -m $model"
    python lotus/queries/selection/Q10/default.py --wandb -m $model -p vllm
done

for model in "${models_ollama[@]}"; do
    echo "Running Q10 cascades with -m $model"
    python lotus/queries/selection/Q10/cascades.py --wandb -m $model -p ollama
done

for model in "${models_vllm[@]}"; do
    echo "Running Q10 cascades with -m $model"
    python lotus/queries/selection/Q10/cascades.py --wandb -m $model -p vllm
done