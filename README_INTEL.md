# F5-TTS-Intel-lora：无 N 卡集成显卡微调与推理

> **Fork 声明**：本仓库 fork 自 [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)（MIT License），
> 上游官方代码零修改，所有新增均为 Intel XPU 适配脚本与文档。预训练权重遵循原 CC-BY-NC 许可。
> 下文为本 fork 的完整中文说明；上游英文原文见 [`README.md`](README.md)。

## 0. 这是什么

在**只有 Intel 核显、没有 N 卡**的机器上，跑通 F5-TTS 的中文音色微调 + 推理服务。
用 QORA-TTS（CPU/Rust）批量合成教师数据（32uwgy 音色，240 条 / 31.7 分钟），
在 Arc 核显上微调 `F5TTS_v1_Base`，常驻 API 对外服务（稳态 RTF~1.0，流式首包 3.3s）。

## 1. 实测环境

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 7 155H（Meteor Lake，22 线程） |
| GPU | Intel Arc Graphics（Xe-LPG 核显，`torch.xpu` 报此名；NPU 用不上） |
| 内存 | 30G（训练峰值 ~27G，见 §5） |
| 系统 | Arch Linux；Python 3.12 venv（uv 建，无 pip，一律 `uv pip --python f5tts/bin/python`） |
| torch | `torch 2.15.0+xpu` + `torchaudio 2.11.1+xpu`（大版本对不上是已知常态，不用强求） |

## 2. 快速开始

```bash
# 1. 环境（顺序不能反：先 XPU torch，后项目依赖）
uv venv --python 3.12 f5tts
uv pip install --python f5tts/bin/python torch torchaudio \
  --index-url https://download.pytorch.org/whl/test/xpu
uv pip install --python f5tts/bin/python -e .
uv pip uninstall --python f5tts/bin/python torchcodec   # 必卸，见 §3

# 2. 推理服务（常驻，模型只加载一次；训练好的 ckpt 需自行训出放 ckpts/TEACHER32/）
f5tts/bin/python serve_32uwgy.py --port 8899
curl -X POST localhost:8899/tts -H 'Content-Type: application/json' \
  -d '{"text":"要合成的中文。"}' --output out.wav
# 流式（首包快）：POST /tts/stream 得 int16 PCM，再 ffmpeg 转 wav（命令见 §6）
```

## 3. 必踩坑：torchaudio / torchcodec

`torchaudio 2.11` 的 `load` 写死走 torchcodec，而 torchcodec 是 CUDA 构建，
在 XPU 机上报 `libnvrtc.so` / `libtorchcodec_image.so` 加载失败（装 `/opt/cuda` 也救不回来）。
解法：卸载 torchcodec，全管线 `torchaudio.load/info` 打到 soundfile（本仓库所有入口脚本已内置该补丁）。

## 4. 教师数据与微调

- 教师用 QORA-TTS（Rust/CPU，约 13x realtime）克隆 `32uwgy`：固定 seed + 预编码 voice + ICL 原文，
  `gen_teacher.sh` 逐句合成（编号按源文件前缀，防覆盖）；文本先过 `clean_texts.py`
 （去 URL/markdown/纯英文，100 字硬封顶，否则超 30s 会被 `dataset.py:58` 静默丢弃）。
- **循环卡死黑名单**（撞 `--max-codes` 上限的表现）：句首英文缩写（CPU/WER/MIUI）、`；。`双标点。
  修法：改写中文表述或换 seed，单条试音验时长；收尾扫查 `>30s / <0.5s / >0.6s每字` 全删。
- `prepare_csv_wavs.py teacher_metadata.csv data/TEACHER32_pinyin`（CSV 音频列必须绝对路径）。
- 训练：`F5TTS_v1_Base --finetune`，batch 1600 frames × grad_accum 2，lr 1e-5，
  25 epoch ≈ 1800 update（XPU 约 16s/update），`save_per 200 / last_per 100 / keep 3`，
  不用 `--bnb_optimizer`（CUDA kernel）。systemd `f5train.service`（`Restart=on-failure`）。

## 5. OOM 生死线（30G 机器）

1. `trainer.train(num_workers=16)` 默认 16 常驻 worker ≈ 20G+ → 本 wrapper 已压到 1
   （0 不合法，`persistent_workers=True` 写死；EMA 也关不掉，认命）。
2. 峰值 25~27G 且到顶不涨（平台型）；训练前关 GSV 服务、停 containerd。
3. 断点续训官方自带（最新 ckpt + optimizer + update 计数），放心重启。

## 6. 推理质量与实时性（核显实测，~5s 音频）

- **复读问题**：生参考音（48% 静音 + 5字/s）会让输出开头复读 ref 尾（base/微调都复读，属 prompt 侧，
  对照 GSV-TTS-Lite 的坏 y）。工程解：固定教师合成的干净 ref 落盘 + faster-whisper ASR 门
 （复读 ≥6 字则换轨迹重试，`infer_32uwgy.py` best-of-3），`--speed 0.8` 配快嘴 ref。

| 档位 | 耗时 | RTF | 结论 |
|---|---|---|---|
| 常驻 + nfe16/cfg2 | 11.6s | 2.3 | 基准 |
| 常驻 + **nfe8/cfg2（默认）** | **4.9s** | **~1.0** | 转写全对，定档 |
| nfe16/cfg0 | 6.2s | 1.2 | 发音稳定性掉，弃 |
| 流式首包 | 3.3s | — | 体感实时（`POST /tts/stream`） |

```bash
curl -s localhost:8899/health
curl -X POST localhost:8899/tts/stream -H 'Content-Type: application/json' \
  -d '{"text":"长文本……"}' --output out.pcm
ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav
```

别碰（已证伪）：torch.compile、fp16（XPU 单流更慢）、FlashAttention（Xe-LPG 无 kernel 自动回退即可）。
再往下只有：流式（已做）或蒸馏到 2~4 步（要开训）。

## 7. 与上游的差异

| 上游 | 本 fork |
|---|---|
| CUDA / ROCm / MPS | Intel Arc 核显全流程（推理、微调、API） |
| 通用音色 | 32uwgy 中文音色教师蒸馏微调（ckpt 本地持有，不进库） |
| Docker CUDA 镜像发布 | 已删除，仅保留 pre-commit（文档改动跳过 CI） |

## 8. 文档索引

- `FINETUNE_XPU.md`：设备级复现手册（本文件的详细版，含复现 checklist）
- `TEACHER32.md`：教师数据管线记录
- 上游英文 README：`README.md`（本 fork 仅顶部加了导流横幅）
