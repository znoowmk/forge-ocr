#!/usr/bin/env bash
# Downloads the PP-OCRv5 detector and Latin recognizer (PaddleOCR, Apache-2.0; ONNX exports
# published by RapidAI) into a models folder and checks them.
#   scripts/fetch-models.sh [DIR]          (default: models)
set -euo pipefail
dir="${1:-models}"
mkdir -p "$dir"
base="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5"
fetch() { # name url sha256
  if [ ! -f "$dir/$1" ]; then curl -fL --retry 3 -o "$dir/$1.part" "$2" && mv "$dir/$1.part" "$dir/$1"; fi
  echo "$3  $dir/$1" | sha256sum -c -
}
fetch ch_PP-OCRv5_det_mobile.onnx "$base/det/ch_PP-OCRv5_det_mobile.onnx" 4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae
fetch latin_PP-OCRv5_rec_mobile.onnx "$base/rec/latin_PP-OCRv5_rec_mobile.onnx" b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a
echo "Models ready in $dir. Optional FP16 recognizer (faster on CUDA): python3 scripts/fp16.py $dir"
