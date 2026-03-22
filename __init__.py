# ComfyUI-Whisper-Cai
# Whisper-based speech recognition and forced alignment nodes for ComfyUI.

from .nodes import (
    WhisperAlignNode,
    WhisperTranscribeNode,
)

NODE_CLASS_MAPPINGS = {
    "WhisperAlign": WhisperAlignNode,
    "WhisperTranscribe": WhisperTranscribeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperAlign": "🎤 Whisper Align (Forced Alignment)",
    "WhisperTranscribe": "🎤 Whisper Transcribe",
}

__version__ = "1.0.0"
print(f"✅ ComfyUI-Whisper-Cai v{__version__} loaded")
