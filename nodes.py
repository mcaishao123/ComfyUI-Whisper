"""
ComfyUI-Whisper-Cai: Whisper-based forced alignment node for accurate subtitle timestamps.
Uses HuggingFace transformers pipeline for word-level speech recognition.
"""

import torch
import json
import re
import os
import folder_paths
from typing import Dict, Any, List, Tuple, Optional

# Global model cache
_WHISPER_CACHE = {}


def _get_whisper_pipeline(model_size: str, device: str, language: str):
    """Load or get cached Whisper pipeline."""
    global _WHISPER_CACHE

    cache_key = (model_size, device)
    if cache_key in _WHISPER_CACHE:
        print(f"[Whisper] Using cached model: {model_size}")
        return _WHISPER_CACHE[cache_key]

    from transformers import pipeline

    model_id = f"openai/whisper-{model_size}"

    # Resolve device
    if device == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    torch_dtype = torch.float16 if "cuda" in device_str else torch.float32

    print(f"[Whisper] Loading model: {model_id} on {device_str}...")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device_str,
    )

    _WHISPER_CACHE[cache_key] = pipe
    print(f"[Whisper] Model loaded: {model_id}")
    return pipe


class WhisperAlignNode:
    """
    🎤 Whisper Forced Alignment: Transcribe audio with word-level timestamps,
    then align to original text for accurate subtitle generation.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Original text to align"}),
                "model_size": (["tiny", "base", "small", "medium", "large-v3"], {"default": "medium"}),
                "language": ("STRING", {"default": "zh", "placeholder": "Language code (zh, en, ja, etc.)"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "srt")
    FUNCTION = "align"
    CATEGORY = "Whisper"
    DESCRIPTION = "Use Whisper to perform forced alignment on audio with original text, producing accurate word-level subtitle timestamps."

    def align(self, audio: Dict[str, Any], text: str, model_size: str, language: str, device: str) -> Tuple[str, str]:
        if not text.strip():
            raise RuntimeError("Text is required for alignment")

        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        # Convert to numpy float32 mono
        if waveform.dim() == 3:
            wav = waveform[0]  # [channels, samples]
        else:
            wav = waveform

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav[0]

        wav_np = wav.cpu().numpy()

        # Resample to 16kHz if needed (Whisper requires 16kHz)
        if sr != 16000:
            import torchaudio
            wav_16k = torchaudio.functional.resample(
                torch.from_numpy(wav_np).unsqueeze(0), sr, 16000
            )
            wav_np = wav_16k[0].numpy()
            print(f"[Whisper] Resampled from {sr}Hz to 16000Hz")

        # Load Whisper pipeline
        pipe = _get_whisper_pipeline(model_size, device, language)

        # Transcribe with word-level timestamps
        print(f"[Whisper] Transcribing audio ({len(wav_np)/16000:.1f}s) with word-level timestamps...")
        result = pipe(
            wav_np,
            return_timestamps="word",
            generate_kwargs={"language": language, "task": "transcribe"},
        )

        whisper_chunks = result.get("chunks", [])
        print(f"[Whisper] Got {len(whisper_chunks)} word chunks from Whisper")

        if not whisper_chunks:
            raise RuntimeError("Whisper did not produce any word-level results. Try a different model size.")

        # --- Align Whisper words to original text ---
        subtitle_segments = self._align_words_to_text(text, whisper_chunks)

        # --- Format output ---
        json_str, srt_str = self._format_output(subtitle_segments)

        return (json_str, srt_str)

    def _align_words_to_text(self, original_text: str, whisper_chunks: List[Dict]) -> List[Dict]:
        """
        Align Whisper's word-level timestamps to the original text.

        Strategy:
        1. Split original text by punctuation into subtitle segments
        2. Clean both Whisper words and original text (remove punctuation)
        3. Walk through Whisper words and match them to each segment
        4. Map start/end timestamps from matched words to each segment
        """
        # Split original text into subtitle segments by all punctuation
        segments = re.split(r'(?<=[。！？.!?，,、；;：:—…])\s*', original_text)
        segments = [s.strip() for s in segments if s.strip()]

        if not segments:
            segments = [original_text]

        # Clean function: remove all punctuation and whitespace
        def clean(t):
            return re.sub(r'[。！？.!?，,、；;：:—…·\-\s\u3000]', '', t)

        # Build a flat list of clean characters from original text with segment indices
        seg_char_map = []  # [(char, seg_index), ...]
        for seg_idx, seg in enumerate(segments):
            for ch in clean(seg):
                seg_char_map.append((ch, seg_idx))

        # Build a flat list of whisper chars with their timestamps
        whisper_chars = []  # [(char, start_time, end_time), ...]
        for chunk in whisper_chunks:
            chunk_text = clean(chunk.get("text", ""))
            ts = chunk.get("timestamp", (None, None))
            start_t = ts[0] if ts[0] is not None else 0.0
            end_t = ts[1] if ts[1] is not None else start_t

            if not chunk_text:
                continue

            # Distribute timestamp evenly across characters in this word
            char_dur = (end_t - start_t) / len(chunk_text) if len(chunk_text) > 0 else 0
            for i, ch in enumerate(chunk_text):
                whisper_chars.append((ch, start_t + i * char_dur, start_t + (i + 1) * char_dur))

        # --- Character-level alignment using simple sequential matching ---
        n_orig = len(seg_char_map)
        n_whsp = len(whisper_chars)

        # Map each original character to its best matching whisper character
        # Using greedy sequential matching
        seg_timestamps = {}  # seg_index -> (first_start, last_end)

        orig_idx = 0
        whsp_idx = 0

        while orig_idx < n_orig and whsp_idx < n_whsp:
            orig_char, seg_idx = seg_char_map[orig_idx]
            whsp_char, w_start, w_end = whisper_chars[whsp_idx]

            if orig_char == whsp_char:
                # Match found
                if seg_idx not in seg_timestamps:
                    seg_timestamps[seg_idx] = [w_start, w_end]
                else:
                    seg_timestamps[seg_idx][1] = w_end
                orig_idx += 1
                whsp_idx += 1
            else:
                # Try to find match by advancing whisper (skip extra whisper chars)
                found = False
                for look_ahead in range(1, min(5, n_whsp - whsp_idx)):
                    if whisper_chars[whsp_idx + look_ahead][0] == orig_char:
                        whsp_idx += look_ahead
                        found = True
                        break

                if not found:
                    # Try advancing original (skip unmatched original chars)
                    for look_ahead in range(1, min(5, n_orig - orig_idx)):
                        if seg_char_map[orig_idx + look_ahead][0] == whisper_chars[whsp_idx][0]:
                            orig_idx += look_ahead
                            found = True
                            break

                if not found:
                    # Skip both
                    orig_idx += 1
                    whsp_idx += 1

        # Build results
        results = []
        for seg_idx, seg_text in enumerate(segments):
            clean_text = clean(seg_text)
            if not clean_text:
                continue

            if seg_idx in seg_timestamps:
                start, end = seg_timestamps[seg_idx]
                results.append({
                    "text": clean_text,
                    "start": round(start, 3),
                    "end": round(end, 3)
                })
            else:
                # Fallback: interpolate from neighbors
                results.append({
                    "text": clean_text,
                    "start": 0.0,
                    "end": 0.0
                })

        # Fix any gaps or overlaps: ensure timestamps are monotonically increasing
        for i in range(1, len(results)):
            if results[i]["start"] < results[i-1]["end"]:
                mid = (results[i]["start"] + results[i-1]["end"]) / 2
                results[i-1]["end"] = round(mid, 3)
                results[i]["start"] = round(mid, 3)

        # Fill in zero-timestamp segments by interpolation
        for i, r in enumerate(results):
            if r["start"] == 0.0 and r["end"] == 0.0 and i > 0:
                r["start"] = results[i-1]["end"]
                if i + 1 < len(results):
                    r["end"] = results[i+1]["start"]
                else:
                    r["end"] = r["start"] + 0.5  # fallback

        return results

    @staticmethod
    def _format_output(segments: List[Dict]) -> Tuple[str, str]:
        """Format segments into JSON and SRT strings."""
        subtitles = []
        for idx, item in enumerate(segments, 1):
            subtitles.append({
                "index": idx,
                "text": item["text"],
                "start": item["start"],
                "end": item["end"]
            })

        # JSON
        json_str = json.dumps(subtitles, ensure_ascii=False, indent=2)

        # SRT
        def _srt_time(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            ms = int(round((s - int(s)) * 1000))
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        srt_lines = []
        for sub in subtitles:
            srt_lines.append(str(sub["index"]))
            srt_lines.append(f"{_srt_time(sub['start'])} --> {_srt_time(sub['end'])}")
            srt_lines.append(sub["text"])
            srt_lines.append("")

        srt_str = "\n".join(srt_lines)

        return (json_str, srt_str)


class WhisperTranscribeNode:
    """
    🎤 Whisper Transcribe: Transcribe audio to text with timestamps.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_size": (["tiny", "base", "small", "medium", "large-v3"], {"default": "medium"}),
                "language": ("STRING", {"default": "zh", "placeholder": "Language code (zh, en, ja, etc.)"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "timestamps": (["none", "word", "sentence"], {"default": "sentence"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "timestamps_json")
    FUNCTION = "transcribe"
    CATEGORY = "Whisper"
    DESCRIPTION = "Transcribe audio to text using Whisper. Optionally include word or sentence-level timestamps."

    def transcribe(self, audio: Dict[str, Any], model_size: str, language: str, device: str, timestamps: str) -> Tuple[str, str]:
        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        if waveform.dim() == 3:
            wav = waveform[0]
        else:
            wav = waveform

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav[0]

        wav_np = wav.cpu().numpy()

        if sr != 16000:
            import torchaudio
            wav_16k = torchaudio.functional.resample(
                torch.from_numpy(wav_np).unsqueeze(0), sr, 16000
            )
            wav_np = wav_16k[0].numpy()

        pipe = _get_whisper_pipeline(model_size, device, language)

        ts_param = None
        if timestamps == "word":
            ts_param = "word"
        elif timestamps == "sentence":
            ts_param = True

        kwargs = {"language": language, "task": "transcribe"}

        if ts_param is not None:
            result = pipe(wav_np, return_timestamps=ts_param, generate_kwargs=kwargs)
        else:
            result = pipe(wav_np, generate_kwargs=kwargs)

        full_text = result.get("text", "")
        chunks = result.get("chunks", [])

        ts_data = []
        for chunk in chunks:
            ts = chunk.get("timestamp", (None, None))
            ts_data.append({
                "text": chunk.get("text", "").strip(),
                "start": ts[0] if ts[0] is not None else 0.0,
                "end": ts[1] if ts[1] is not None else 0.0,
            })

        return (full_text, json.dumps(ts_data, ensure_ascii=False, indent=2))

class AudioSplitByTimestampsNode:
    """
     Split audio into segments based on JSON timestamps.
    Outputs a list of audio segments that can be connected to Save/Preview nodes.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "json_timestamps": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_segments",)
    FUNCTION = "split_audio"
    CATEGORY = "Whisper"
    DESCRIPTION = "Split audio into multiple segments based on JSON timestamps. Each segment is output as a separate AUDIO."
    OUTPUT_IS_LIST = (True,)

    def split_audio(self, audio: Dict[str, Any], json_timestamps: str):
        import json as json_lib

        # Parse JSON timestamps
        try:
            timestamps = json_lib.loads(json_timestamps)
        except json_lib.JSONDecodeError:
            raise RuntimeError("Invalid JSON timestamps format")

        if not isinstance(timestamps, list) or len(timestamps) == 0:
            raise RuntimeError("Timestamps must be a non-empty JSON array")

        waveform = audio["waveform"]  # [batch, channels, samples]
        sr = audio["sample_rate"]

        if waveform.dim() == 3:
            wav = waveform[0]  # [channels, samples]
        else:
            wav = waveform

        audio_list = []
        for item in timestamps:
            idx = item.get("index", 0)
            text = item.get("text", "")
            start = float(item.get("start", 0))
            end = float(item.get("end", 0))

            start_sample = int(start * sr)
            end_sample = int(end * sr)
            end_sample = min(end_sample, wav.shape[-1])
            start_sample = max(0, start_sample)

            if start_sample >= end_sample:
                continue

            segment = wav[:, start_sample:end_sample].unsqueeze(0)  # [1, channels, samples]
            audio_list.append({"waveform": segment, "sample_rate": sr})
            print(f"[Whisper]  Segment {idx}: [{start:.3f}s ~ {end:.3f}s] \"{text}\"")

        print(f"[Whisper]  Split into {len(audio_list)} audio segments")
        return (audio_list,)


class WhisperOpeningSplitNode:
    """
     Whisper Opening Split:
    Split audio and JSON timestamps into exactly two parts: an Opening part and a Content part,
    based on the opening text. Subtitles for the Content part are adjusted to start at 0.0s.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "whisper_json": ("STRING", {"multiline": True, "default": ""}),
                "opening_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Exact opening text, e.g. 大家好，我是..."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio_opening", "json_opening", "srt_opening", "audio_content", "json_content", "srt_content")
    FUNCTION = "split"
    CATEGORY = "Whisper"

    def split(self, audio: Dict[str, Any], whisper_json: str, opening_text: str):
        import json as json_lib
        
        try:
            timestamps = json_lib.loads(whisper_json)
        except json_lib.JSONDecodeError:
            raise RuntimeError("Invalid JSON timestamps format")

        if not isinstance(timestamps, list) or len(timestamps) == 0:
            raise RuntimeError("Timestamps must be a non-empty JSON array")
            
        def clean(t):
            return re.sub(r'[。！？.!?，,、；;：:\-\s\u3000]', '', t)
            
        clean_opening = clean(opening_text)
        if not clean_opening:
            raise RuntimeError("Opening text is empty after removing punctuation.")

        # Find the split point
        split_time = 0.0
        split_index = -1
        accumulated_clean = ""
        
        for idx, item in enumerate(timestamps):
            item_clean = clean(item.get("text", ""))
            accumulated_clean += item_clean
            # We found the segment containing the end of the opening text
            # Or exactly equal to opening text
            if clean_opening in accumulated_clean:
                split_time = float(item.get("end", 0.0))
                split_index = idx
                break
                
        if split_index == -1:
            print("[Whisper] Warning: Could not find exact match for opening text. Using full audio as content.")
            split_time = 0.0
            split_index = -1
            opening_segments = []
            content_segments = timestamps
        else:
            opening_segments = timestamps[:split_index + 1]
            content_segments = timestamps[split_index + 1:]
            
        # Adjust timestamps for content
        adjusted_content = []
        for item in content_segments:
            new_item = item.copy()
            new_item["start"] = round(max(0.0, float(item["start"]) - split_time), 3)
            new_item["end"] = round(max(0.0, float(item["end"]) - split_time), 3)
            adjusted_content.append(new_item)
            
        # Split Audio
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        split_sample = int(split_time * sr)
        end_sample = waveform.shape[-1]
        split_sample = min(split_sample, end_sample)
        split_sample = max(0, split_sample)
        
        # Audio can be 3D [batch, channels, samples] or 2D [channels, samples]
        if waveform.dim() == 3:
            wav_opening = waveform[:, :, :split_sample]
            wav_content = waveform[:, :, split_sample:]
        else:
            wav_opening = waveform[:, :split_sample]
            wav_content = waveform[:, split_sample:]
            
        audio_opening = {"waveform": wav_opening, "sample_rate": sr}
        audio_content = {"waveform": wav_content, "sample_rate": sr}
        
        json_opening, srt_opening = WhisperAlignNode._format_output(opening_segments)
        json_content, srt_content = WhisperAlignNode._format_output(adjusted_content)
        
        print(f"[Whisper]  Split complete. Opening ends at {split_time:.3f}s")
        
        return (audio_opening, json_opening, srt_opening, audio_content, json_content, srt_content)
