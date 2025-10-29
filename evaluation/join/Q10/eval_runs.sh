#!/bin/bash
sizes=(5 10 20)
models_ollama=("gemma3:12b" "llama3.1:8b")
models=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        python evaluation/join/Q10/eval_scripts/lotus_q10_eval.py -s $size -m $model -p ollama
        python evaluation/join/Q10/eval_scripts/blendsql_q10_eval.py -s $size -m $model -p ollama
    done

    for model in "${models_vllm[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        python evaluation/join/Q10/eval_scripts/lotus_q10_eval.py -s $size -m $model -p vllm
        python evaluation/join/Q10/eval_scripts/blendsql_q10_eval.py -s $size -m $model -p vllm
    done
done