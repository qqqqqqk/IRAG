from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(
#             "Qwen/Qwen3-0.6B",
#             # use_auth_token=use_auth_token
#         )

from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="Qwen/Qwen3-0.6B",
  cache_dir="/root/autodl-tmp/hf_cache",
  max_workers=1,
  resume_download=True,
)