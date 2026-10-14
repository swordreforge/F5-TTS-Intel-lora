# TEACHER32：32uwgy 音色教师蒸馏 + Intel XPU 微调（本地记录）

> 官方代码未动（`git status` 干净，仅新增本目录脚本）。大文件（`data/`、`ckpts/`、`tests/`）走 `.gitignore`，不进仓库。

## 链路

1. **教师数据**：QORA-TTS-12Hz-1.7B（Rust 二进制，`target/release/qora-tts`）克隆 `32uwgy.wav`，
   固定 `seed=1790574344731051634` + 预编码 `32uwgy.qvoice` + ICL ref-text，逐句合成。
   - `gen_teacher.sh <in.txt> <wav_dir> <csv>`：按源文件前缀编号（跨文件不碰撞），`--max-codes 400`。
   - `clean_texts.py`：去 URL/日期/markdown/纯英文行，长英文括号展开删除，`，；：、` 拆分 + 100 字硬封顶（对应 <30s，`dataset.py:58` 会静默丢弃超限条）。
   - `recover_teacher.py`：删截断/回粘坏条，算缺失句。
2. **循环卡死黑名单**（同 seed 复现）：句首英文缩写（CPU/WER/MIUI）、`；。`双标点 → 改写中文表述或换 seed。
3. **prepare**：`prepare_csv_wavs.py teacher_metadata.csv data/TEACHER32_pinyin`（240 条 / 0.53h）。
4. **训练**：`train_teacher32.py`（torchaudio→soundfile 补丁；`num_workers=1`，默认 16 会 OOM；
   batch 1600 frames × grad_accum 2；25 epoch ≈ 1800 update，XPU 约 16s/update）。
   systemd `f5train.service`（`Restart=on-failure`）。OOM 教训：默认 16 worker 直接 OOM（实测峰值 26.9G），先保内存再谈收敛。
5. **推理**：`infer_32uwgy.py`（固定好 ref + faster-whisper ASR 门 + best-of-N）调 `infer_once.py` 子进程。
   - 固定 ref：`data/teacher_wavs/extend3.clean.miss_0047.wav` + 原文（已验干净）。
   - 生 ref（32uwgy.wav，48% 静音 + 5字/s）必复读 ref 尾（base/微调都复读，见 GSV-TTS-Lite/LESSONS.md 的坏 y 同构）。
   - `--speed 0.8` 配快嘴 ref；`--ckpt base|finetuned|<path>`。

## 环境

- venv：`f5tts/`（uv 建，Python 3.12，无 pip，用 `uv pip --python f5tts/bin/python`）。
- torch 2.15.0+xpu（XPU 可用，`Intel(R) Arc(TM) Graphics`）；torchaudio 2.11.1+xpu 的 `load` 写死 torchcodec（CUDA 构建，
  缺 `libnvrtc` 起不来）→ 全管线用 soundfile 兜底。torchcodec 已卸载，别装。
- XPU 单流别碰 fp16（GSV-TTS-Lite 实测更慢，见 GSV-TTS-Lite/INTEL_XPU.md）。
