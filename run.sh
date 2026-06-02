#!/bin/bash
# bash run.sh

# model list
MODELS=(
  "Qwen/Qwen2.5-32B-Instruct"
  # "mistralai/Ministral-3-8B-Instruct-2512-BF16"
)

DATASET_NAME="2wiki"

# compress model path
COMPRESS_MODEL="Qwen/Qwen3-4B-Instruct-2507"

# heads json path
HEADS_JSON="./compressor/top_heads_supporting.json"

# dataset path
DATASET_PATH="datasets/${DATASET_NAME}/test_subsampled.jsonl"

# passage path
# embedding path
PASSAGE_PATH="datacorpus/${DATASET_NAME}/corpus.tsv"
EMBEDDING_PATH="datacorpus/${DATASET_NAME}/e5"  

# multihop config
MAX_HOPS="5"
TOPK="10,20,40,80"
COMPRESS_THRESHOLD="0.2"

# config file
BASE_CONFIG="multihop_config.yaml"

# log dir
LOG_DIR="logs/${DATASET_NAME}"
mkdir -p ${LOG_DIR}

echo "========================================="
echo "model num:${#MODELS[@]}"
echo "dataset path:${DATASET_PATH}"
echo "compress model:${COMPRESS_MODEL}"
echo "heads json:${HEADS_JSON}"
echo "max_hops: ${MAX_HOPS}"
echo "topk: ${TOPK}"
echo "compress_threshold: ${COMPRESS_THRESHOLD}"
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
    --compress_model_name "${COMPRESS_MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --passage_path "${PASSAGE_PATH}" \
    --embedding_path "${EMBEDDING_PATH}" \
    --heads_json "${HEADS_JSON}" \
    --max_hops "${MAX_HOPS}" \
    --topk "${TOPK}" \
    --compress_threshold "${COMPRESS_THRESHOLD}" \
    2>&1 | tee "${LOG_FILE}"
    
done
