from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, LlamaConfig, AutoModelForCausalLM
import torch
from models import InforNCE_and_Generative_Hops_Eval
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", add_eos_token=True)
special_tokens = ["[STOP_SEARCH]", "[SUFFICIENT_EVIDENCE]", "[ANSWER_READY]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
tokenizer.padding_side = "left"
model = InforNCE_and_Generative_Hops_Eval(0, "Qwen/Qwen3-0.6B") # InforNCE_and_Eigenvalue_Eval(training_args.local_rank, model_args.model_name)
model.model.encoder.base_model.model.resize_token_embeddings(len(tokenizer))
checkpoint = torch.load("/root/autodl-tmp/multihop_reasoning_embedding_epoch_0_step_399_Qwen_Qwen3-0.6B_bs_512_lambda_0.0/global_step399/mp_rank_00_model_states.pt")
model.load_state_dict(checkpoint['module'], strict=True)
model.cuda()
model.eval()
model.to(torch.bfloat16)

sentence = ["[STOP_SEARCH] [SUFFICIENT_EVIDENCE] [ANSWER_READY]"]
sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
sentence = sentence.to("cuda")
output = model(**sentence)
import pdb
pdb.set_trace()
print()