# F5-TTS-Intel-lora：无 N 卡集成显卡微调与推理

> **Fork 声明**：本仓库 fork 自 [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)（MIT License），
> 上游官方代码零修改，所有新增均为 Intel XPU 适配脚本与文档。预训练权重遵循原 CC-BY-NC 许可。

## 与上游的差异

| 上游 | 本 fork |
|---|---|
| CUDA / ROCm / MPS 推理训练 | Intel Arc 核显（`torch.xpu`）全流程：推理、微调、API 服务 |
| 英文参考 + 通用音色 | 32uwgy 中文音色教师蒸馏微调（`ckpts/TEACHER32/`，本地持有，不进库） |
| Docker CUDA 镜像发布 | 已删除（`.github/workflows/publish-docker-image.yaml`），仅保留 pre-commit |
| CI 全量触发 | 文档改动跳过 CI（`paths-ignore`） |

## 快速开始（Intel 核显）

```bash
# 1. 环境（uv + Python 3.12，先 XPU torch 后项目依赖；torchcodec 必须卸载，见 FINETUNE_XPU.md §2）
uv venv --python 3.12 f5tts
uv pip install --python f5tts/bin/python torch torchaudio \
  --index-url https://download.pytorch.org/whl/test/xpu
uv pip install --python f5tts/bin/python -e .
uv pip uninstall --python f5tts/bin/python torchcodec

# 2. 推理服务（常驻，模型只加载一次）
f5tts/bin/python serve_32uwgy.py --port 8899
curl -X POST localhost:8899/tts -H 'Content-Type: application/json' \
  -d '{"text":"要合成的中文。"}' --output out.wav
```

systemd 常驻见 `FINETUNE_XPU.md`；单次推理/门控见 `infer_32uwgy.py`。

## 管线（详见 FINETUNE_XPU.md）

1. QORA-TTS（Rust/CPU）克隆 32uwgy 批量合成教师数据 → `gen_teacher.sh` + `clean_texts.py`
2. `prepare_csv_wavs.py` → `data/TEACHER32_pinyin`（240 条 / 0.53h）
3. `train_teacher32.py` 微调（XPU，25 epoch）
4. `serve_32uwgy.py` 对外服务（`/tts` 整句、`/tts/stream` 流式首包 3.3s）

实测档位：nfe8/cfg2 → RTF~1.0；复读问题用固定好 ref + ASR 门解决（详见 TEACHER32.md）。

## 文档索引

- `FINETUNE_XPU.md`：设备级复现手册（环境、OOM 线、RTF 档位）
- `TEACHER32.md`：教师数据管线记录
- 上游英文 README 保持原样（`README.md`），本文件为 fork 中文说明。
