# IRAG

## Data Preparation

### 1. Evaluation Dataset

We use the evaluation data from the [AceSearcher](https://github.com/ritaranx/AceSearcher/) repo.

Place evaluation data under `datasets/{dataset_name}/`:

```
datasets/
├── musique/
│   └── test_subsampled.jsonl
└── 2wiki/
│   └── test_subsampled.jsonl
└── hotpot/
    └── test_subsampled.jsonl
```

Each line in the JSONL file should contain:

### 2. Corpus & Embeddings

Corpus data can be obtained using scripts from the [IRCOT](https://github.com/StonyBrookNLP/ircot) repo.

Place the passage corpus and pre-built embeddings under `datacorpus/{dataset_name}/`

## Configuration

All parameters can be set in `multihop_config.yaml` and overridden via command-line arguments. 

## Usage

### Run Inference

Edit `run.sh` to set your models and parameters, then:

```bash
bash run.sh
```


You can also run inference directly:

```bash
python batch_multihop_inference.py \
  --config multihop_config.yaml \
  --model_name "Qwen/Qwen2.5-32B-Instruct" \
  --compress_model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset_path "datasets/2wiki/test_subsampled.jsonl" \
  --passage_path "datacorpus/2wiki/corpus.tsv" \
  --embedding_path "datacorpus/2wiki/e5" \
  --heads_json "./compressor/top_heads_supporting.json" \
  --max_hops "5" \
  --topk "10,20,40,80" \
  --compress_threshold "0.2"
```

Results are saved to `results_multihop/{dataset_name}/{model_name}/` as JSONL files. Logs are saved to `logs/{dataset_name}/`.

### Evaluate Results

```bash
python eval_multihop.py --file results_multihop/2wiki/Qwen_Qwen2.5-32B-Instruct/multihop_hops5_topk10_compress0.2.jsonl
```
