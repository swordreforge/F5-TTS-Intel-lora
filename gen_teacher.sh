#!/bin/bash
# Batch teacher-data generation: QORA-TTS clone 32uwgy voice, fixed seed,
# one wav per sentence + F5-TTS metadata.csv row.
# Clip names are prefixed per source file (no cross-file collisions).
# Input txt must already be one-sentence-per-line (see clean_texts.py).
# Usage: gen_teacher.sh <input.txt> <out_wav_dir> <out_csv>
set -u
QORA_DIR="/home/swordreforge/下载/QORA-TTS-12Hz-1.7B"
BIN="$QORA_DIR/target/release/qora-tts"
REF_AUDIO="$QORA_DIR/32uwgy.wav"
REF_TEXT="现在我的本体处在海的深层这里的端流很危险用于保护研究员的稳定结构正在被持续破坏我必须不断对它进行重构"
VOICE="$QORA_DIR/32uwgy.qvoice"
SEED=1790574344731051634
MAX_CODES=400 # 12.5 codes/s -> 32s cap; dataset.py drops >30s anyway

IN_TXT="$1"; OUT_DIR="$2"; OUT_CSV="$3"
PREFIX=$(basename "$IN_TXT" .txt)
mkdir -p "$OUT_DIR"
[ -f "$OUT_CSV" ] || echo "audio_file|text" > "$OUT_CSV"

i=0
ok=0
while IFS= read -r sent; do
  [ -z "$sent" ] && continue
  i=$((i+1))
  wav="$OUT_DIR/${PREFIX}_$(printf '%04d' $i).wav"
  if [ -f "$wav" ]; then echo "[$PREFIX $i] skip (exists)"; continue; fi
  echo "[$PREFIX $i] synth ${#sent} chars"
  if "$BIN" --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT" \
      --load-voice "$VOICE" --text "$sent" --language chinese \
      --seed "$SEED" --max-codes "$MAX_CODES" --output "$wav" 2>"$wav.log"; then
    echo "$wav|$sent" >> "$OUT_CSV"
    ok=$((ok+1))
    echo "[$PREFIX $i] ok"
  else
    echo "[$PREFIX $i] FAILED, see $wav.log"
  fi
done < "$IN_TXT"
echo "DONE: $PREFIX total=$i new=$ok"
