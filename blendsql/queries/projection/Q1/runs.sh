#!/bin/bash
sizes=(10 20 50)
models_ollama=("llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
transformers=("llama3.1:8b")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Running Q1 with -s $size and m $model"
        python blendsql/queries/projection/Q1/q1.py --wandb -s $size -m $model -p ollama
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_vllm[@]}"; do
        echo "Running Q1 with -s $size and m $model"
        python blendsql/queries/projection/Q1/q1.py --wandb -s $size -m $model -p vllm
    done
done

for size in "${sizes[@]}"; do
    for model in "${transformers[@]}"; do
        echo "Running Q1 with -s $size and m $model"
        python blendsql/queries/projection/Q1/q1.py --wandb -s $size -m $model -p transformers
    done
done