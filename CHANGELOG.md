# Changelog

## [1.0.0] — 2026-03-14

### New Features

- **Multi-model support**: Added support for Ming-omni-tts-0.5B (MLX-native, ~2.8 GB) alongside Fish Audio S2 Pro BF16
- **VoiceDesign / `instruct` field**: New `instruct` parameter in `TTSRequest` for voice style description, enabling Qwen3-TTS VoiceDesign models; auto-applies default style for `qwen3_tts` model type
- **Flow-matching model parameters**: New `cfg_scale`, `flow_steps`, and `sigma` fields in `TTSRequest` for Ming-omni and other flow-matching architectures
- **Extended sampling controls**: Added `top_k` parameter alongside existing `top_p` and `temperature`
- **Model type detection**: `TTSEngine` now reads `model_type` from the loaded model to apply model-specific inference logic
- **`trust_remote_code=True`**: `load_model` now passes `trust_remote_code=True` to support models with custom code (e.g. Ming-omni)

### Changes

- Default `temperature` changed from `0.8` → `0.0` (deterministic by default)
- Default `top_p` changed from `0.8` → `0.7`
- `TTSEngine._run_model` now passes `top_p`, `top_k`, `cfg_scale`, `flow_steps` (`ddpm_steps`), `sigma`, and `instruct` to `model.generate`

### Fixes

- Fixed startup failure when running with system Python (conda `base`): documented that `uv run` or `.venv` activation is required

---

## 更新日志

## [1.0.0] — 2026-03-14

### 新功能

- **多模型支持**：新增对 Ming-omni-tts-0.5B（MLX 原生格式，约 2.8GB）的支持，与 Fish Audio S2 Pro BF16 并列可用
- **VoiceDesign / `instruct` 字段**：`TTSRequest` 新增 `instruct` 参数，用于描述声音风格，支持 Qwen3-TTS VoiceDesign 类模型；`qwen3_tts` 模型类型会自动应用默认风格
- **流匹配模型参数**：`TTSRequest` 新增 `cfg_scale`、`flow_steps`、`sigma` 字段，适配 Ming-omni 等流匹配架构
- **扩展采样控制**：新增 `top_k` 参数，与现有 `top_p`、`temperature` 配合使用
- **模型类型检测**：`TTSEngine` 启动时读取已加载模型的 `model_type`，据此应用模型专属推理逻辑
- **`trust_remote_code=True`**：`load_model` 调用时传入 `trust_remote_code=True`，支持含自定义代码的模型（如 Ming-omni）

### 变更

- 默认 `temperature` 从 `0.8` 改为 `0.0`（默认确定性输出）
- 默认 `top_p` 从 `0.8` 改为 `0.7`
- `TTSEngine._run_model` 现在将 `top_p`、`top_k`、`cfg_scale`、`flow_steps`（映射为 `ddpm_steps`）、`sigma`、`instruct` 传递给 `model.generate`

### 修复

- 修复使用系统 Python（conda `base` 环境）启动时报 `ModuleNotFoundError: No module named 'mlx_audio'` 的问题：文档中明确说明需使用 `uv run` 或激活 `.venv`
