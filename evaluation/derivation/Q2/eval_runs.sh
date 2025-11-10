#!/bin/bash
sizes=(10 20 50)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
transformers=("llama3.1:8b")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Evaluating Lotus with Ollama, -s $size and -m $model"
        python evaluation/derivation/Q2/eval_scripts/lotus_q2_eval.py -s $size -m $model -p ollama
        echo "Evaluating Palimpzest with Ollama, -s $size and -m $model"
        python evaluation/derivation/Q2/eval_scripts/pz_q2_eval.py -s $size -m $model -p ollama
        echo ""
        echo "Evaluating BlendSQL with Ollama, -s $size and -m $model"
        python evaluation/derivation/Q2/eval_scripts/blendsql_q2_eval.py -s $size -m $model -p ollama
        echo ""
    done

    for model in "${models_vllm[@]}"; do
        echo "Evaluating Lotus with vLLM, -s $size and -m $model"
        python evaluation/derivation/Q2/eval_scripts/lotus_q2_eval.py -s $size -m $model -p vllm
        echo ""
        echo "Evaluating Palimpzest with vLLM, -s $size -m $model "
        python evaluation/derivation/Q2/eval_scripts/pz_q2_eval.py -s $size -m $model -p vllm
        echo ""
        echo "Evaluating BlendSQL with vLLM, -s $size and -m $model"
        python evaluation/derivation/Q2/eval_scripts/blendsql_q2_eval.py -s $size -m $model -p vllm
        echo ""
    done

    for model in "${transformers[@]}"; do
        echo "Evaluating BlendSQL with Transformers, -s $size and -m $model"
        python evaluation/derivation/Q2/eval_scripts/blendsql_q2_eval.py -s $size -m $model -p transformers
        echo ""
    done
done