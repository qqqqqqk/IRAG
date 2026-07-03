export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/hf_cache"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/hf_cache"
export HUGGING_FACE_TOKEN="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"
python ircot_evaluation.py \
        --model_path /root/autodl-tmp/va_embedding-4-11-_epoch_0_step_799__root_autodl-tmp_hf_cache_Qwen3-0.6B_bs_1024_lambda_0.0/global_step799 \
        --dataset_name musique \
        --split validation \
        --tokenizer_path Qwen/Qwen3-0.6B \
        --index_path ./results/musique_document_index_new.pt \
        --output_dir ./results