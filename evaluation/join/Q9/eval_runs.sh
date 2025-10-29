#!/bin/bash
sizes=(20 50 100)
models_ollama=("gema3:12b" "llama3.1:8b")
models=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        echo -n "Lotus "
        python evaluation/join/Q9/eval_scripts/lotus_q9_eval.py -s $size -m $model -p ollama
        echo -n "BlendSQL "
        python evaluation/join/Q9/eval_scripts/blendsql_q9_eval.py -s $size -m $model -p ollama
        echo ""
    done

    for model in "${models_vllm[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        echo -n "Lotus "
        python evaluation/join/Q9/eval_scripts/lotus_q9_eval.py -s $size -m $model -p vllm
        echo -n "BlendSQL "
        python evaluation/join/Q9/eval_scripts/blendsql_q9_eval.py -s $size -m $model -p vllm
        echo ""
    done
done