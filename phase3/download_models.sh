#!/bin/bash
mkdir -p phase3/checkpoints/pretrained/hindi
mkdir -p phase3/checkpoints/pretrained/nepali

BASE_URL="https://huggingface.co/manyajain111/lma-phase2/resolve/main"

wget -q $BASE_URL/hindi/model_final.pt -O phase3/checkpoints/pretrained/hindi/model_final.pt
wget -q $BASE_URL/hindi/config.json -O phase3/checkpoints/pretrained/hindi/config.json
wget -q $BASE_URL/hindi/tokenizer.json -O phase3/checkpoints/pretrained/hindi/tokenizer.json

wget -q $BASE_URL/nepali/model_final.pt -O phase3/checkpoints/pretrained/nepali/model_final.pt
wget -q $BASE_URL/nepali/config.json -O phase3/checkpoints/pretrained/nepali/config.json
wget -q $BASE_URL/nepali/tokenizer.json -O phase3/checkpoints/pretrained/nepali/tokenizer.json

echo "Downloaded all files!"
