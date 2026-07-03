from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class DataTrainingArguments:
    path: Optional[str] = field(
        default='/data/ablation_study/data'
    )
    max_length: int = field(default=512)
    eval_path: Optional[str] = field(
        default='/root/mydata/value_aggregation/eval_dataset'
    )
    
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="/root/autodl-tmp/hf_cache/Qwen3-1.7B",
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    save_dir: Optional[str] = field(
        default="/root/autodl-tmp/multihop_reasoning_embedding",
        metadata={
            "help": "The place to save the trained model."
        },
    )

@dataclass
class TrainingArguments:
    weight_decay: float = field(default=1e-4)
    lr: float = field(default=1e-4)
    deepspeed:Optional[str] = field(
        default="./va_deepspeed_stage_new.json",
        metadata={
            "help": "Deepspeed file path."
        }
    )
    local_rank: Optional[int] = field(default=0)
    train_epoch: Optional[int] = field(default=1)
    train_batch_size: Optional[int] = field(default=128)
    dropout_rate: float = field(default=0.0)