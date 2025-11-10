#!/bin/bash
sizes=(50 100 200)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
transformers=("llama3.1:8b")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Running Q3 with -s $size and m $model"
        python blendsql/queries/derivation/Q3/q3.py --wandb -s $size -m $model -p ollama
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_vllm[@]}"; do
        echo "Running Q3 with -s $size and m $model"
        python blendsql/queries/derivation/Q3/q3.py --wandb -s $size -m $model -p vllm
    done
done

for size in "${sizes[@]}"; do
    for model in "${transformers[@]}"; do
        echo "Running Q3 with -s $size and m $model"
        python blendsql/queries/derivation/Q3/q3.py --wandb -s $size -m $model -p transformers
    done
done