DATASET_NAME="musique"
EMBEDDING_MODEL_TYPE="qwen3"
EMBEDDING_MODEL_PATH="/data/lzb/models/Qwen3-Embedding-4B"

PASSAGE_PATH="/data/agentic-rag/datacorpus/${DATASET_NAME}/corpus.tsv"
OUTPUT_PATH="/data/agentic-rag/datacorpus/${DATASET_NAME}/${EMBEDDING_MODEL_TYPE}"

python ./retrievers/passage_embedder.py \
    --passages "${PASSAGE_PATH}" \
    --output_dir "${OUTPUT_PATH}" \
    --model_name_or_path "${EMBEDDING_MODEL_PATH}" \
    --test_mode \
    --chunk_size 10000 \
    --batch_size 32