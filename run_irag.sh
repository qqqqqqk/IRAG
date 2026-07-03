#!/bin/bash
# bash run.sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
# model list
MODELS=(
  "deepseek-v4-flash"
)

DATASET_NAME="hotpot"

RETRIEVER_MODEL_TYPE="qwen3"
RETRIEVER_MODEL_PATH="/data/lzb/models/Qwen3-Embedding-4B"
RERANK_MODEL="/data/lzb/models/Qwen3-Reranker-4B"

# dataset path
DATASET_PATH="/data/agentic-rag/datasets/${DATASET_NAME}/test_subsampled.jsonl"

# passage path
# embedding path
PASSAGE_PATH="/data/agentic-rag/datacorpus/${DATASET_NAME}/corpus.jsonl"
EMBEDDING_PATH="/data/agentic-rag/datacorpus/${DATASET_NAME}/${RETRIEVER_MODEL_TYPE}"  

# config file
BASE_CONFIG="multihop_config.yaml"

# log dir
LOG_DIR="logs/${DATASET_NAME}"
mkdir -p ${LOG_DIR}

echo "========================================="
echo "model num:${#MODELS[@]}"
echo "dataset path:${DATASET_PATH}"
echo "rerank model:${RERANK_MODEL}"
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
  
  python denoiseirag.py \
    --config ${BASE_CONFIG} \
    --model_name "${MODEL}" \
    --reranker_model_name "${RERANK_MODEL}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_path "${DATASET_PATH}" \
    --passage_path "${PASSAGE_PATH}" \
    --embedding_path "${EMBEDDING_PATH}" \
    --retriever_model_type "${RETRIEVER_MODEL_TYPE}" \
    --retriever_model_path "${RETRIEVER_MODEL_PATH}" \
    2>&1 | tee "${LOG_FILE}"
    
done
