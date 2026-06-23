python ./retrievers/passage_embedder.py \
    --passages /home/lizibo/lzb/IRAG/datacorpus/lifebench/corpus.jsonl \
    --output_dir /home/lizibo/lzb/IRAG/datacorpus/lifebench/e5 \
    --model_name_or_path /data/lzb/models/Qwen3-Embedding-4B \
    --test_mode \
    --chunk_size 10000 \
    --batch_size 32