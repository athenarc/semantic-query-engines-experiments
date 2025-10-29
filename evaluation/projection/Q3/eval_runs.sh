#!/bin/bash
sizes=(50 100 200)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
models_transformers=("llama3.1:8b")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Evaluating Lotus with -s $size and -m $model"
        python evaluation/projection/Q3/eval_scripts/lotus_q3_eval.py -s $size -m $model -p ollama
        echo ""
        echo "Evaluating BlendSQL with -s $size and -m $model"
        python evaluation/projection/Q3/eval_scripts/blendsql_q3_eval.py -s $size -m $model -p ollama
        echo ""
    done

    for model in "${models_vllm[@]}"; do
        echo "Evaluating Lotus with -s $size and -m $model"
        python evaluation/projection/Q3/eval_scripts/lotus_q3_eval.py -s $size -m $model -p vllm
        echo ""
        echo  "Evaluating BlendSQL with -s $size -m $model "
        python evaluation/projection/Q3/eval_scripts/blendsql_q3_eval.py -s $size -m $model -p vllm
        echo ""
    done

    for model in "${models_transformers[@]}"; do
        echo "Evaluating BlendSQL -s $size and -m $model (Transformers)"
        python evaluation/projection/Q3/eval_scripts/blendsql_q3_eval.py -s $size -m $model -p transformers
        echo ""
    done
done