#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <huggingface-model-id> <output-dir> [quantization]" >&2
  echo "Example: $0 sarvamai/indic-whisper-large-v2-ct2 models/indic-large-ct2" >&2
  exit 1
fi

MODEL_ID="$1"
OUTPUT_DIR="$2"
QUANTIZATION="${3:-float16}"

if ! command -v ct2-transformers-converter >/dev/null 2>&1; then
  echo "ct2-transformers-converter is not installed. Install with: pip install ctranslate2 transformers" >&2
  exit 2
fi

echo "Converting $MODEL_ID -> $OUTPUT_DIR (quantization: $QUANTIZATION)"

ct2-transformers-converter \
  --model "$MODEL_ID" \
  --output_dir "$OUTPUT_DIR" \
  --quantization "$QUANTIZATION" \
  --copy_files tokenizer_config.json preprocessor_config.json config.json special_tokens_map.json || {
    echo "Conversion failed" >&2
    exit 3
}

echo "Done. Point videomaker --model at $OUTPUT_DIR"
