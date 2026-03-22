# ComfyUI-Whisper-Cai

Whisper-based speech recognition and forced alignment nodes for ComfyUI.

## Features
- **🎤 Whisper Align (Forced Alignment)**: Provide TTS audio and its original text. Transcribes audio with word-level timestamps and aligns them to the original text segments for highly accurate subtitle generation.
- **🎤 Whisper Transcribe**: Standalone audio transcription using HuggingFace `transformers` Whisper.

## Installation
Go to `ComfyUI/custom_nodes/` and clone this repository:
```bash
git clone https://github.com/mcaishao123/ComfyUI-Whisper.git ComfyUI-Whisper-Cai
cd ComfyUI-Whisper-Cai
pip install -r requirements.txt
```

## Usage
Add `Whisper Align` downstream from any TTS node (like Qwen3-TTS) yielding audio and text. Connect its `json` output to `✂️ Audio Split (Timestamps)` to accurately segment audio for subtitles.
