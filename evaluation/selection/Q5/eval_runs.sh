#!/bin/bash
sizes=(100 1000 10000)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Evaluating Lotus with -s $size and -m $model"
        python evaluation/selection/Q5/eval_scripts/lotus_q5_eval.py -s $size -m $model -p ollama
        echo "Evaluating Palimpzest with -s $size and -m $model"
        python evaluation/selection/Q5/eval_scripts/pz_q5_eval.py -s $size -m $model -p ollama
        echo ""
    done

    for model in "${models_vllm[@]}"; do
        echo "Evaluating Lotus with -s $size and -m $model"
        python evaluation/selection/Q5/eval_scripts/lotus_q5_eval.py -s $size -m $model -p vllm
        echo ""
        echo -n "Evaluating Palimpzest with -s $size -m $model "
        python evaluation/selection/Q5/eval_scripts/pz_q5_eval.py -s $size -m $model -p vllm
        echo ""
    done
done