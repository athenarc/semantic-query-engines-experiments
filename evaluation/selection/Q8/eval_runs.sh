#!/bin/bash
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for model in "${models_ollama[@]}"; do
    echo "Evaluating Lotus with -m $model"
    python evaluation/selection/Q8/eval_scripts/lotus_q8_eval.py -m $model -p ollama
    echo ""
    
    echo "Evaluating Palimpzest with -m $model"
    python evaluation/selection/Q8/eval_scripts/pz_q8_eval.py -m $model -p ollama
    echo ""
done

for model in "${models_vllm[@]}"; do
    echo "Evaluating Lotus with -m $model"
    python evaluation/selection/Q8/eval_scripts/lotus_q8_eval.py -m $model -p vllm
    echo ""

    echo "Evaluating Palimpzest with -m $model"
    python evaluation/selection/Q8/eval_scripts/pz_q8_eval.py -m $model -p vllm
    echo ""
done