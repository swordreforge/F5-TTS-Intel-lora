# F5-TTS Intel XPU 微调复现手册（设备级精确版）

目标：在 **Intel 核显一体机**上，无 NVIDIA、无 conda，从零复现 32uwgy 音色微调 + 推理服务。
官方代码零修改（`git status` 仅新增脚本）；数据与 ckpt 不进仓库（`.gitignore` 已覆盖）。

## 0. 设备快照（实测环境）

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 7 155H（Meteor Lake，22 线程） |
| GPU | Intel Arc Graphics（Xe-LPG 核显，`renderD128`，`torch.xpu.get_device_name` 报此名） |
| NPU | Intel AI Boost（PyTorch 用不上，忽略） |
| 内存 | 30G（训练峰值 ~27G，必须关杂项，见 §5） |
| 系统 | Arch Linux（`/opt/cuda` 存在但无 N 卡；torchcodec 即便有它也起不来，见 §2） |
| Python | venv `f5tts/`，**uv 建，Python 3.12**，无 pip（一切 `uv pip --python f5tts/bin/python`） |

## 1. 环境安装（顺序不能反）

```bash
cd /home/swordreforge/F5-TTS
uv venv --python 3.12 f5tts
# 先装 XPU torch（注意：test/xpu 源最高只有 torchaudio 2.9.1，与新 torch 配不对版本是常态）
uv pip install --python f5tts/bin/python torch torchaudio \
  --index-url https://download.pytorch.org/whl/test/xpu
# 再装项目（会把 numpy 顶到 2.x，随它去；torchcodec 会被带进来——必须卸，见 §2）
uv pip install --python f5tts/bin/python -e .
```

版本实测：`torch 2.15.0+xpu` + `torchaudio 2.11.1+xpu`（**大版本对不上是已知状态，不用强求对齐**）。

## 2. 必踩坑：torchaudio / torchcodec（省 2 小时）

- `torchaudio 2.11` 的 `load` 写死走 torchcodec；torchcodec 是 CUDA 构建，
  在 XPU 机上报 `libnvrtc.so` / `libtorchcodec_image.so` 加载失败（即使装了 `/opt/cuda` 也救不回来）。
- 解法：**卸载 torchcodec**，全管线 `torchaudio.load/info` 打到 soundfile：

```python
import soundfile as sf, torch, torchaudio
torchaudio.load = lambda p, *a, **k: (lambda d, sr: (torch.from_numpy(d.T), sr))(*sf.read(p, dtype="float32", always_2d=True))
```

- 本仓库所有入口脚本（`train_*.py`、`infer_*.py`、`serve_*.py`）都已内置该补丁。

## 3. 教师数据（QORA-TTS 蒸馏，CPU 跑，约 13x realtime）

教师：`/home/swordreforge/下载/QORA-TTS-12Hz-1.7B/target/release/qora-tts`（Rust，模型 `model.qora-tts` 1.6G 同目录），
`--check` 先过自检。固定embl：`32uwgy.wav`（12.88s）+ `32uwgy.qvoice`（预编码，省 encoder 时间）
+ ICL ref-text（音频原文）+ `--seed 1790574344731051634` + `--language chinese`。

```bash
./gen_teacher.sh <one-sentence-per-line.txt> data/teacher_wavs teacher_metadata.csv
```

- 文本先过 `clean_texts.py`：删 URL/日期/markdown/纯英文行；超长英文括号展开删除；
  `，；：、` 拆分 + **100 字硬封顶**（>30s 会被 `dataset.py:58` 静默丢弃）。
- **编号必须按源文件前缀**（`gen_teacher.sh` 已做），否则多文件互相 `skip`/覆盖（我们丢过 83 句）。
- **循环卡死黑名单**（同 seed 必现，表现=撞 `--max-codes` 上限被截断）：句首英文缩写（CPU/WER/MIUI）、
  `；。`双标点。修法：改写中文表述（CPU→中央处理器）或换 seed；改完单条试音验时长。
- 收尾必做时长扫查：`>30s / <0.5s / >0.6s每字` 全视为坏对，删 wav + 删 CSV 行。
- `--max-codes 400`（=32s，超了就截断，宁可截断暴露也不要静默长垃圾）。

本次规模：240 条 / 31.7 分钟，`prepare_csv_wavs.py teacher_metadata.csv data/TEACHER32_pinyin`
（注意 CSV 音频列**必须绝对路径**；数据集目录名须是 `<NAME>_pinyin`，`load_dataset` 按此拼接）。

## 4. 训练（XPU，约 16s/update）

```bash
f5tts/bin/python train_teacher32.py   # 或 systemd f5train.service
```

关键参数：`F5TTS_v1_Base --finetune`，batch 1600 frames × grad_accum 2，
lr 1e-5，warmup 200，25 epoch ≈ 1800 update，`save_per 200 / last_per 100 / keep 3`，**不用 `--bnb_optimizer`**（CUDA kernel）。
本次落点：`ckpts/TEACHER32/model_{1400,1600,1800}.pt` + `model_last.pt`。

## 5. OOM 生死线（30G 机器，血泪）

1. `trainer.train(num_workers=16)` 默认 16 常驻 worker ≈ 20G+ → **压到 1**（0 不合法，
   `persistent_workers=True` 写死）。本 wrapper 已 patch。
2. EMA 关不掉（`trainer.py:106` 写死创建），认命留 ~1.2G。
3. 峰值 25~27G：关 GSV 服务（`gsv-api/gsv-webui`）、停 containerd 再训；峰值平台型（到顶不涨），
   单调爬才是泄漏。
4. systemd `Restart=on-failure` + `linger`，63 次 OOM 重启的教训：先看 `oom-kill` 再调参，别空转。
5. 断点续训官方自带（load 最新 ckpt + optimizer + update 计数），放心重启。

## 6. 推理：复读根因与门控（对照 GSV-TTS-Lite/LESSONS.md 的坏 y）

- 现象：输出开头复读 ref 尾（ASR 实锤 `"不断对它进行重构" + 正文`），**base/微调双双复读** → prompt 侧问题。
- 根因：生 ref（32uwgy.wav，48% 静音 + 5字/s）使 `duration = ref_len/ref_text_len`  pacing 失准，
  `generated[:, ref_audio_len:]` 一刀切露出 ref 尾。2x2 矩阵：换干净 ref 即好。
- 工程解（`infer_32uwgy.py`）：**固定好 ref 落盘**（`extend3.clean.miss_0047` 对，已验干净），
  faster-whisper ASR 门（ref 尾复读 ≥6 字则换轨迹重试，best-of-3），`--speed 0.8` 配快嘴 ref。
- 单次推理走 `infer_once.py` 子进程（`infer_cli` 在 import 时解析 argv，不隔离必串参；顺带崩溃隔离）。

## 7. 实时性（核显实测，~5s 音频）

| 档位 | 耗时 | RTF | 结论 |
|---|---|---|---|
| 常驻 + nfe16/cfg2 | 11.6s | 2.3 | 基准 |
| 常驻 + **nfe8/cfg2（默认）** | **4.9s** | **~1.0** | 转写全对，定档 |
| nfe16/cfg0 | 6.2s | 1.2 | 发音稳定性掉（边墙/重够），弃 |
| 流式首包 | 3.3s | — | 体感实时（`POST /tts/stream`，首句切小） |

- 加载 10.6s/次 → 常驻服务摊掉（`serve_32uwgy.py` + `f5api.service`，:8899，启动预热付掉 oneDNN JIT）。
- GSV 已证伪、**别碰**：torch.compile、fp16（XPU 单流更慢）、FlashAttention（Xe-LPG 无 kernel，自动回退即可）、
  midpoint 求解器（默认 euler 就是 1 倍量）。
- 再往下只有两条路：流式（体感，已做）或蒸馏到 2~4 步（要开训）。

```bash
curl -s localhost:8899/health
curl -X POST localhost:8899/tts -H 'Content-Type: application/json' \
  -d '{"text":"要合成的中文。"}' --output out.wav
curl -X POST localhost:8899/tts/stream -H 'Content-Type: application/json' \
  -d '{"text":"长文本……"}' --output out.pcm
ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav
```

## 8. 复现 checklist（按序）

1. `uv venv` + 先 XPU torch 后 `-e .`，卸 torchcodec。
2. QORA `--check`（模型+参考音）→ 单句试音测速（~13x realtime）。
3. 文本 `clean_texts.py` → `gen_teacher.sh`（systemd 跑，`Restart=on-failure`）→ 时长扫查。
4. `prepare_csv_wavs.py`（绝对路径！）→ `data/<NAME>_pinyin`。
5. 关杂项（gsv/containerd）→ `train_teacher32.py`（num_workers=1，batch1600×accum2）。
6. 固定好 ref → `infer_32uwgy.py` 验复读 → 常驻 `f5api.service`。
