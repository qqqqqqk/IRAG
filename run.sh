#!/bin/bash
# bash run.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
# model list
MODELS=(
  # "Qwen/Qwen2.5-32B-Instruct",
  # "mistralai/Ministral-3-8B-Instruct-2512-BF16",
  # "/data/lzb/models/Qwen2.5-32B-Instruct",
  # "/data/lzb/models/Qwen3-8B",
  "deepseek-v4-flash"
)

DATASET_NAME="lifebench"

# compress model path
# COMPRESS_MODEL="Qwen/Qwen3-4B-Instruct-2507"
# COMPRESS_MODEL="/data/lzb/models/Qwen3-4B-Instruct-2507"
RERANK_MODEL="/data/lzb/models/Qwen3-Reranker-4B"

# heads json path
HEADS_JSON="./compressor/top_heads_supporting.json"

# dataset path
DATASET_PATH="datasets/${DATASET_NAME}/test_subsampled.jsonl"

# passage path
# embedding path
PASSAGE_PATH="datacorpus/${DATASET_NAME}/corpus.jsonl"
EMBEDDING_PATH="datacorpus/${DATASET_NAME}/e5"  

# multihop config
MAX_HOPS="5"
TOPK="10,20,40,80"
RERANK_THRESHOLD="0.2"

# config file
BASE_CONFIG="multihop_config.yaml"

# log dir
LOG_DIR="logs/${DATASET_NAME}"
mkdir -p ${LOG_DIR}

echo "========================================="
echo "model num:${#MODELS[@]}"
echo "dataset path:${DATASET_PATH}"
echo "rerank model:${RERANK_MODEL}"
echo "heads json:${HEADS_JSON}"
echo "max_hops: ${MAX_HOPS}"
echo "topk: ${TOPK}"
echo "rerank_threshold: ${RERANK_THRESHOLD}"
echo "log dir:${LOG_DIR}"
echo "========================================="

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(echo "${MODEL}" | sed 's/\//_/g')

  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${TIMESTAMP}.log"
  
  echo ""
  echo "========================================="
  echo "start process model: ${MODEL}"
  echo "log file:${LOG_FILE}"
  echo "========================================="
  
  python batch_multihop_inference.py \
    --config ${BASE_CONFIG} \
    --model_name "${MODEL}" \
    --reranker_model_name "${RERANK_MODEL}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_path "${DATASET_PATH}" \
    --passage_path "${PASSAGE_PATH}" \
    --embedding_path "${EMBEDDING_PATH}" \
    --heads_json "${HEADS_JSON}" \
    --max_hops "${MAX_HOPS}" \
    --topk "${TOPK}" \
    --rerank_threshold "${RERANK_THRESHOLD}" \
    --retriever_model_type Qwen3-Embedding-4B \
    --retriever_model_path /data/lzb/models/Qwen3-Embedding-4B \
    2>&1 | tee "${LOG_FILE}"
    
done
