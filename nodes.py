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

        print(f"[Whisper] Transcribing audio ({len(wav_np)/16000:.1f}s) with word-level timestamps...")
        result = pipe(
            wav_np,
            return_timestamps="word",
            generate_kwargs={"language": language, "task": "transcribe"},
        )

        whisper_chunks = result.get("chunks", [])
        if not whisper_chunks:
            raise RuntimeError("Whisper did not produce any word-level results.")

        subtitle_segments = self._align_words_to_text(text, whisper_chunks)
        json_str, srt_str = self._format_output(subtitle_segments)

        return (json_str, srt_str)

    def _align_words_to_text(self, original_text: str, whisper_chunks: List[Dict]) -> List[Dict]:
        def clean(t):
            return re.sub(r'[。！？.!?，,、；;：:—…·\-\s\u3000]', '', t)

        segments = re.split(r'(?<=[。！？.!?，,、；;：:—…])\s*', original_text)
        segments = [s.strip() for s in segments if s.strip()]
        if not segments: segments = [original_text]

        seg_char_map = []
        for seg_idx, seg in enumerate(segments):
            for ch in clean(seg):
                seg_char_map.append((ch, seg_idx))

        whisper_chars = []
        for chunk in whisper_chunks:
            chunk_text = clean(chunk.get("text", ""))
            ts = chunk.get("timestamp", (None, None))
            start_t = ts[0] if ts[0] is not None else 0.0
            end_t = ts[1] if ts[1] is not None else start_t

            if not chunk_text: continue
            char_dur = (end_t - start_t) / len(chunk_text) if len(chunk_text) > 0 else 0
            for i, ch in enumerate(chunk_text):
                whisper_chars.append((ch, start_t + i * char_dur, start_t + (i + 1) * char_dur))

        # Greedy Character Matching
        n_orig = len(seg_char_map)
        n_whsp = len(whisper_chars)
        seg_timestamps = {}
        orig_idx = 0
        whsp_idx = 0

        while orig_idx < n_orig and whsp_idx < n_whsp:
            orig_char, seg_idx = seg_char_map[orig_idx]
            whsp_char, w_start, w_end = whisper_chars[whsp_idx]

            if orig_char == whsp_char:
                if seg_idx not in seg_timestamps:
                    seg_timestamps[seg_idx] = [w_start, w_end]
                else:
                    seg_timestamps[seg_idx][1] = max(seg_timestamps[seg_idx][1], w_end)
                orig_idx += 1
                whsp_idx += 1
            else:
                found = False
                for look_ahead in range(1, min(10, n_whsp - whsp_idx)):
                    if whisper_chars[whsp_idx + look_ahead][0] == orig_char:
                        whsp_idx += look_ahead
                        found = True; break
                if not found:
                    for look_ahead in range(1, min(10, n_orig - orig_idx)):
                        if seg_char_map[orig_idx + look_ahead][0] == whisper_chars[whsp_idx][0]:
                            orig_idx += look_ahead
                            found = True; break
                if not found:
                    orig_idx += 1; whsp_idx += 1

        results = []
        for seg_idx, seg_text in enumerate(segments):
            if not clean(seg_text): continue
            if seg_idx in seg_timestamps:
                s, e = seg_timestamps[seg_idx]
                results.append({"text": seg_text, "start": round(s, 3), "end": round(max(s, e), 3)})
            else:
                results.append({"text": seg_text, "start": -1.0, "end": -1.0})

        # --- Interpolation & Monotonicity ---
        n_res = len(results)
        if n_res == 0: return []

        # Boundaries
        if results[0]["start"] < 0:
            results[0]["start"] = 0.0
            results[0]["end"] = 0.0
        
        last_audio_time = whisper_chars[-1][2] if whisper_chars else 0.0
        if results[-1]["end"] < 0:
            results[-1]["end"] = last_audio_time
            results[-1]["start"] = last_audio_time

        # Ensure non-decreasing
        for i in range(n_res):
            if i > 0 and results[i]["start"] < results[i-1]["end"] and results[i]["start"] >= 0:
                results[i]["start"] = results[i-1]["end"]
                if results[i]["end"] < results[i]["start"] and results[i]["end"] >= 0:
                    results[i]["end"] = results[i]["start"]
            
            if results[i]["start"] >= 0 and results[i]["end"] < 0:
                for j in range(i + 1, n_res):
                    if results[j]["start"] >= 0:
                        results[i]["end"] = results[j]["start"]; break
                if results[i]["end"] < 0: results[i]["end"] = results[i]["start"]

        # Proportional Fill Gaps
        idx = 0
        while idx < n_res:
            # Check if this segment or sequence of segments are "point" markers (interpolation gaps)
            if results[idx]["start"] == results[idx]["end"]:
                gap_start_idx = idx
                gap_end_idx = idx
                while gap_end_idx < n_res and results[gap_end_idx]["start"] == results[gap_end_idx]["end"]:
                    gap_end_idx += 1
                
                # Gap is between results[gap_start_idx-1]["end"] and results[gap_end_idx]["start"]
                t_start = results[gap_start_idx-1]["end"] if gap_start_idx > 0 else 0.0
                t_end = last_audio_time
                if gap_end_idx < n_res:
                    t_end = results[gap_end_idx]["start"]
                
                if t_end > t_start:
                    gap_segs = results[gap_start_idx : gap_end_idx]
                    total_chars = sum(len(clean(r["text"])) for r in gap_segs)
                    if total_chars > 0:
                        dur = t_end - t_start
                        curr = t_start
                        for r in gap_segs:
                            step = (len(clean(r["text"])) / total_chars) * dur
                            r["start"] = round(curr, 3)
                            r["end"] = round(curr + step, 3)
                            curr += step
                idx = gap_end_idx
            else:
                idx += 1

        return results

    @staticmethod
    def _format_output(segments: List[Dict]) -> Tuple[str, str]:
        subtitles = []
        for idx, item in enumerate(segments, 1):
            subtitles.append({"index": idx, "text": item["text"], "start": item["start"], "end": item["end"]})
        json_str = json.dumps(subtitles, ensure_ascii=False, indent=2)

        def _srt_time(s):
            h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
            ms = int(round((s - int(s)) * 1000))
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        srt_lines = []
        for sub in subtitles:
            srt_lines.extend([str(sub["index"]), f"{_srt_time(sub['start'])} --> {_srt_time(sub['end'])}", sub["text"], ""])
        return json_str, "\n".join(srt_lines)


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
            wav = waveform[0]
        else:
            wav = waveform

        audio_list = []
        n = len(timestamps)
        
        for i in range(n):
            item = timestamps[i]
            idx = item.get("index", i + 1)
            text = item.get("text", "")
            
            # --- Midpoint Logic ---
            # Start: Midpoint between prev_end and current_start
            s_val = float(item.get("start", 0))
            if i > 0:
                prev_e = float(timestamps[i-1].get("end", s_val))
                actual_start = (prev_e + s_val) / 2
            else:
                actual_start = 0.0 # First segment starts at beginning or specified 0.0
            
            # End: Midpoint between current_end and next_start
            e_val = float(item.get("end", s_val))
            if i < n - 1:
                next_s = float(timestamps[i+1].get("start", e_val))
                actual_end = (e_val + next_s) / 2
            else:
                actual_end = wav.shape[-1] / sr # Last segment goes to end
            
            # Convert to samples
            start_sample = int(actual_start * sr)
            end_sample = int(actual_end * sr)
            
            # Boundary checks
            end_sample = min(end_sample, wav.shape[-1])
            start_sample = max(0, start_sample)

            if start_sample >= end_sample:
                # Fallback to exact if midpoint logic fails (e.g. overlapping data)
                start_sample = int(s_val * sr)
                end_sample = int(e_val * sr)
                if start_sample >= end_sample: continue

            segment = wav[:, start_sample:end_sample].unsqueeze(0)
            audio_list.append({"waveform": segment, "sample_rate": sr})
            print(f"[Whisper] ✂️ Segment {idx} (Midpoint): [{actual_start:.3f}s ~ {actual_end:.3f}s] \"{text}\"")

        print(f"[Whisper] ✂️ Split into {len(audio_list)} seamless audio segments")
        return (audio_list,)


class WhisperOpeningSplitNode:
    """
    ✂️ Whisper Opening Split:
    Split audio and JSON timestamps into exactly two parts based on the midpoint of the gap 
    between the opening and content.
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
            return re.sub(r'[。！？.!?，,、；;：:—…·\-\s\u3000]', '', t)
            
        clean_opening = clean(opening_text)
        if not clean_opening:
            raise RuntimeError("Opening text is empty after removing punctuation.")

        # Find the split point (the last segment of the opening)
        split_index = -1
        accumulated_clean = ""
        for idx, item in enumerate(timestamps):
            item_clean = clean(item.get("text", ""))
            accumulated_clean += item_clean
            if clean_opening in accumulated_clean:
                split_index = idx
                break
                
        if split_index == -1:
            print("[Whisper] Warning: Could not find exact match for opening text. Using all as content.")
            split_time = 0.0
            opening_segments = []
            content_segments = timestamps
        else:
            opening_segments = timestamps[:split_index + 1]
            content_segments = timestamps[split_index + 1:]
            
            # --- Midpoint Logic ---
            t_opening_end = float(opening_segments[-1].get("end", 0.0))
            if content_segments:
                t_content_start = float(content_segments[0].get("start", t_opening_end))
                # Split exactly in the middle of the pause
                split_time = (t_opening_end + t_content_start) / 2
            else:
                split_time = t_opening_end

        # Adjust timestamps for content based on split_time
        adjusted_content = []
        for item in content_segments:
            new_item = item.copy()
            new_item["start"] = round(max(0.0, float(item["start"]) - split_time), 3)
            new_item["end"] = round(max(0.0, float(item["end"]) - split_time), 3)
            adjusted_content.append(new_item)
            
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        split_sample = int(split_time * sr)
        
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
        
        print(f"[Whisper] ✂️ Midpoint Split complete. Gap cut at {split_time:.3f}s")
        return (audio_opening, json_opening, srt_opening, audio_content, json_content, srt_content)
