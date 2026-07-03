export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/hf_cache"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/hf_cache"
export HUGGING_FACE_TOKEN="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"
# export HF_HOME="/path/to/hf_cache"
# export HUGGINGFACE_HUB_CACHE="/path/to/hf_cache"
# export HUGGING_FACE_TOKEN="your_token_here"
# python ./download_model.py
deepspeed --num_gpus 2 multihop_contrastive_train.py --train_batch_size 512 --model_name Qwen/Qwen3-0.6B