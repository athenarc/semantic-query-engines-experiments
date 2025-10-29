#!/bin/bash
sizes=(100 1000 10000)
models_ollama=("llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Running Q5 with -s $size and m $model"
        python palimpzest/queries/selection/Q5/q5.py --wandb -s $size -m $model -p ollama
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_vllm[@]}"; do
        echo "Running Q5 with -s $size and m $model"
        python palimpzest/queries/selection/Q5/q5.py --wandb -s $size -m $model -p vllm
    done
done