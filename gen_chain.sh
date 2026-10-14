#!/bin/bash
# Chain: wait for any running qora-tts batch, then synthesize missing files.
set -u
cd /home/swordreforge/F5-TTS
while pgrep -f "target/release/qora-tts" >/dev/null; do
  echo "$(date '+%H:%M:%S') waiting for running batch..."
  sleep 120
done
for f in data/missing/extend.clean.miss.txt data/missing/extend1.clean.miss.txt \
         data/missing/extend2.clean.miss.txt data/missing/extend3.clean.miss.txt; do
  [ -f "$f" ] || continue
  echo "$(date '+%H:%M:%S') starting $f"
  ./gen_teacher.sh "$f" data/teacher_wavs teacher_metadata.csv >>/tmp/teacher_batch3.log 2>&1
done
echo "$(date '+%H:%M:%S') ALL DONE"
