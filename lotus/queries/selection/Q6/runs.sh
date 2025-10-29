a#!/bin/bash
sizes=(8 14 30) 
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Running Q5 default with -s $size and m $model"
        python lotus/queries/selection/Q6/default.py  --wandb -s $size -m $model -p ollama
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_vllm[@]}"; do
        echo "Running Q5 default with -s $size and m $model"
        python lotus/queries/selection/Q6/default.py  --wandb -s $size -m $model -p vllm
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Running Q5 cascades with -s $size and -m $model"
        python lotus/queries/selection/Q6/cascades.py --wandb -s $size -m $model -p ollama
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_vllm[@]}"; do
        echo "Running Q5 cascades with -s $size and -m $model"
        python lotus/queries/selection/Q5/cascades.py --wandb -s $size -m $model -p vllm
    done
done