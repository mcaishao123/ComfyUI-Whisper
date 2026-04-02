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

# Global model caches
_WHISPER_CACHE = {}
_SENSEVOICE_CACHE = {}
_MMS_FA_CACHE = {}


def _get_mms_fa_model(device: str):
    """Load or get cached MMS Forced Alignment model."""
    global _MMS_FA_CACHE
    import torchaudio
    
    if device == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    
    if device_str in _MMS_FA_CACHE:
        return _MMS_FA_CACHE[device_str]
    
    print(f"[Whisper] Loading MMS Forced Alignment model on {device_str}...")
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(device_str)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    
    _MMS_FA_CACHE[device_str] = (model, tokenizer, aligner, bundle.sample_rate)
    print(f"[Whisper] MMS FA model loaded.")
    return _MMS_FA_CACHE[device_str]


def _get_sensevoice_model(device: str):
    """Load or get cached SenseVoice model."""
    global _SENSEVOICE_CACHE

    if device == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    if device_str in _SENSEVOICE_CACHE:
        print(f"[Whisper] Using cached SenseVoiceSmall on {device_str}")
        return _SENSEVOICE_CACHE[device_str]

    try:
        from funasr import AutoModel
        print(f"[Whisper] Loading SenseVoiceSmall on {device_str}...")
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device_str,
            trust_remote_code=True
        )
        _SENSEVOICE_CACHE[device_str] = model
        return model
    except ImportError:
        raise ImportError("The 'funasr' and 'modelscope' packages are required for SenseVoice. Please ensure they are installed and restart ComfyUI.")


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
                "engine": (["ForcedAlign", "Whisper", "SenseVoice"], {"default": "ForcedAlign"}),
                "model_size": (["tiny", "base", "small", "medium", "large-v3"], {"default": "medium"}),
                "language": ("STRING", {"default": "zh", "placeholder": "语言代码 (zh, en, ja 等)"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "srt")
    FUNCTION = "align"
    CATEGORY = "Whisper"
    DESCRIPTION = "Use Whisper to perform forced alignment on audio with original text, producing accurate word-level subtitle timestamps."

    def align(self, audio: Dict[str, Any], text: str, engine: str, model_size: str, language: str, device: str) -> Tuple[str, str]:
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

        if engine == "ForcedAlign":
            return self._forced_align(wav_np, sr, text, device)
        elif engine == "SenseVoice":
            model = _get_sensevoice_model(device)
            print(f"[Whisper] (SenseVoice) Transcribing audio ({len(wav_np)/16000:.1f}s)...")
            # SenseVoice expects input at 16k mono
            res = model.generate(
                input=wav_np,
                cache={},
                language=language if language != "auto" else "auto",
                use_itn=True,
                batch_size_s=60,
            )
            
            # Map SenseVoice output to Whisper-like chunks
            whisper_chunks = []
            if res and isinstance(res, list) and len(res) > 0:
                # Check for timestamp or sentence_info
                # SenseVoice/FunASR result format varies by version, usually: 
                # [{'text': '...', 'timestamp': [[s, e], ...], 'sentence_info': [...]}]
                item = res[0]
                text_content = item.get("text", "")
                
                # If we have sentence-level timestamps from VAD
                if "timestamp" in item:
                    # Some versions return [[s, e], ...]
                    # We need to pair them with text splits or use as is
                    for ts in item["timestamp"]:
                        start_t, end_t = ts[0] / 1000.0, ts[1] / 1000.0
                        whisper_chunks.append({
                            "text": "", # Text mapping happens in align_words_to_text
                            "text": "", 
                            "timestamp": (ts[0] / 1000.0, ts[1] / 1000.0)
                        })
                else:
                    whisper_chunks.append({"text": text_content, "timestamp": (0.0, len(wav_np)/16000)})
        else:
            pipe = _get_whisper_pipeline(model_size, device, language)
            print(f"[Whisper] Transcribing audio ({len(wav_np)/16000:.1f}s) with word-level timestamps...")
            result = pipe(
                wav_np,
                return_timestamps="word",
                generate_kwargs={"language": language, "task": "transcribe"},
            )
            whisper_chunks = result.get("chunks", [])

        if not whisper_chunks:
            raise RuntimeError(f"{engine} did not produce any results.")

        subtitle_segments = self._align_words_to_text(text, whisper_chunks)
        
        # --- RMS Energy Refinement ---
        try:
            subtitle_segments = self._refine_with_energy(wav_np, 16000, subtitle_segments)
        except Exception as e:
            print(f"[Whisper] Warning: RMS refinement failed: {e}")

        json_str, srt_str = self._format_output(subtitle_segments)
        return (json_str, srt_str)

    def _forced_align(self, wav_np, sr: int, text: str, device: str) -> Tuple[str, str]:
        """Frame-level forced alignment using MMS CTC model with Chinese pinyin support."""
        import torchaudio
        import numpy as np
        from pypinyin import pinyin, Style
        
        model, tokenizer, aligner, model_sr = _get_mms_fa_model(device)
        
        if device == "auto":
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device
        
        # 1. Prepare audio
        waveform = torch.from_numpy(wav_np).unsqueeze(0).float()
        if sr != model_sr:
            waveform = torchaudio.functional.resample(waveform, sr, model_sr)
        waveform = waveform.to(device_str)
        
        # 2. Split text into sentences by punctuation
        segments = re.split(r'(?<=[。！？.!?，,、；;：:—…])\s*', text)
        segments = [s.strip() for s in segments if s.strip()]
        if not segments:
            segments = [text]
        
        def clean(t):
            return re.sub(r'[。！？.!?，,、；;：:—…·\-\s\u3000]', '', t)
        
        all_clean_chars = clean(text)
        if not all_clean_chars:
            raise RuntimeError("Text contains no alignable characters after cleaning.")
        
        # 3. Convert each Chinese character to pinyin, track mapping
        # char_pinyin_map: list of (original_char, pinyin_str, token_count)
        char_pinyin_map = []
        all_tokens = []
        
        for char in all_clean_chars:
            # Get pinyin for this character (lowercase, no tone numbers)
            py_list = pinyin(char, style=Style.NORMAL, errors='default')
            py_str = py_list[0][0] if py_list else char
            
            # Tokenize the pinyin
            try:
                char_tokens = tokenizer(py_str)
                # Flatten nested lists
                flat_tokens = []
                for t in char_tokens:
                    if isinstance(t, list):
                        flat_tokens.extend(t)
                    else:
                        flat_tokens.append(t)
                # Filter out blank tokens (0)
                flat_tokens = [t for t in flat_tokens if t != 0]
                
                if flat_tokens:
                    char_pinyin_map.append((char, py_str, len(flat_tokens)))
                    all_tokens.extend(flat_tokens)
                else:
                    # Unknown character, skip
                    char_pinyin_map.append((char, py_str, 0))
            except Exception:
                char_pinyin_map.append((char, py_str, 0))
        
        if not all_tokens:
            raise RuntimeError("MMS tokenizer produced no valid tokens after pinyin conversion.")
        
        print(f"[Whisper] (ForcedAlign) {len(all_clean_chars)} chars → {len(all_tokens)} phoneme tokens")
        
        # 4. Compute emissions
        with torch.inference_mode():
            emission, _ = model(waveform)
        emission = emission[0].cpu()  # [T, C]
        
        # 5. Run CTC forced alignment
        token_tensor = torch.tensor(all_tokens, dtype=torch.int32)
        
        try:
            aligned_tokens, scores = torchaudio.functional.forced_align(
                emission.unsqueeze(0), token_tensor.unsqueeze(0), blank=0
            )
            aligned_tokens = aligned_tokens[0]  # [T]
        except Exception as e:
            print(f"[Whisper] ForcedAlign CTC failed: {e}, falling back to proportional.")
            total_dur = len(wav_np) / model_sr
            results = []
            char_dur = total_dur / len(all_clean_chars)
            ci = 0
            for seg_text in segments:
                cs = clean(seg_text)
                if not cs: continue
                results.append({"text": cs, "start": round(ci * char_dur, 3), "end": round((ci + len(cs)) * char_dur, 3)})
                ci += len(cs)
            return self._format_output(results)
        
        # 6. Extract per-phoneme-token timestamps
        frame_dur = 320.0 / model_sr  # 20ms per frame
        
        # Find start/end frame for each non-blank token occurrence
        token_spans = []  # list of (start_frame, end_frame) for each token in all_tokens
        token_idx = 0
        frame_idx = 0
        
        while frame_idx < len(aligned_tokens) and token_idx < len(all_tokens):
            tok_val = aligned_tokens[frame_idx].item()
            if tok_val == 0:
                frame_idx += 1
                continue
            
            # Found start of a token
            start_f = frame_idx
            expected_tok = all_tokens[token_idx]
            
            # Advance until token changes or becomes blank
            while frame_idx < len(aligned_tokens) and aligned_tokens[frame_idx].item() == expected_tok:
                frame_idx += 1
            
            token_spans.append((start_f, frame_idx))
            token_idx += 1
        
        # 7. Map phoneme-token spans back to characters
        char_times = []
        span_idx = 0
        for char, py_str, n_tokens in char_pinyin_map:
            if n_tokens == 0 or span_idx >= len(token_spans):
                # Character had no valid tokens; interpolate from neighbors
                if char_times:
                    last_end = char_times[-1]["end"]
                    char_times.append({"start": last_end, "end": last_end + 0.05})
                else:
                    char_times.append({"start": 0.0, "end": 0.05})
                continue
            
            # This character spans n_tokens phoneme tokens
            end_span = min(span_idx + n_tokens, len(token_spans))
            char_start = token_spans[span_idx][0] * frame_dur
            char_end = token_spans[end_span - 1][1] * frame_dur
            char_times.append({"start": char_start, "end": char_end})
            span_idx = end_span
        
        print(f"[Whisper] (ForcedAlign) Aligned {len(char_times)} characters at {frame_dur*1000:.0f}ms resolution")
        
        # 8. Map character times to sentence segments
        results = []
        char_idx = 0
        for seg_text in segments:
            clean_seg = clean(seg_text)
            if not clean_seg:
                continue
            
            seg_len = len(clean_seg)
            end_idx = min(char_idx + seg_len - 1, len(char_times) - 1)
            
            if char_idx < len(char_times):
                s = char_times[char_idx]["start"]
                e = char_times[end_idx]["end"]
            else:
                s = char_times[-1]["end"] if char_times else 0.0
                e = s + seg_len * 0.1
            
            results.append({"text": clean_seg, "start": round(s, 3), "end": round(e, 3)})
            char_idx += seg_len
        
        # 9. RMS refinement
        try:
            results = self._refine_with_energy(wav_np, model_sr, results)
        except Exception as e:
            print(f"[Whisper] Warning: RMS refinement failed: {e}")
        
        json_str, srt_str = self._format_output(results)
        return (json_str, srt_str)


    def _refine_with_energy(self, wav: Any, sr: int, segments: List[Dict]) -> List[Dict]:
        """Refine boundaries by snapping to silence using RMS energy."""
        import numpy as np
        
        # 1. Compute RMS energy in 10ms windows
        win_size = int(sr * 0.02) # 20ms
        hop_size = int(sr * 0.01) # 10ms
        
        # Simple RMS calculation
        n_frames = (len(wav) - win_size) // hop_size + 1
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_size
            chunk = wav[start : start + win_size]
            rms[i] = np.sqrt(np.mean(chunk**2))
            
        # Threshold: 30% of median non-zero energy or a small floor
        active_rms = rms[rms > 0]
        threshold = np.percentile(active_rms, 30) if len(active_rms) > 0 else 0.005
        threshold = max(threshold, 0.005)

        def is_voiced(t):
            frame_idx = int(t * sr / hop_size)
            if frame_idx < 0 or frame_idx >= len(rms): return False
            return rms[frame_idx] > threshold

        # 2. Refine Boundaries with tight snapping
        refined = []
        for i, seg in enumerate(segments):
            s, e = seg["start"], seg["end"]
            
            # Snap END forward: only if very clear voice
            if is_voiced(e):
                furthest_v = e
                for delta in np.arange(0.01, 0.4, 0.01):
                    if is_voiced(e + delta):
                        furthest_v = e + delta
                    else:
                        break # Silence found
                    if i + 1 < len(segments) and segments[i+1]["start"] >= 0:
                        if (e + delta) >= segments[i+1]["start"]: break
                e = furthest_v
            else:
                # Snap END backward: trim silence more aggressively
                for delta in np.arange(0.01, 0.3, 0.01):
                    if is_voiced(e - delta):
                        e = e - delta + 0.01
                        break

            # Snap START: usually Whisper is good at start, but let's be safe
            if is_voiced(s):
                furthest_v = s
                for delta in np.arange(0.01, 0.2, 0.01):
                    if is_voiced(s - delta):
                        furthest_v = s - delta
                    else:
                        break
                    if i > 0 and (s - delta) <= segments[i-1]["end"]: break
                s = furthest_v
            
            seg["start"], seg["end"] = round(s, 3), round(e, 3)
            refined.append(seg)

        # 3. FINAL PASS: Force perfect non-overlap 
        for i in range(1, len(refined)):
            # If segments are touching or overlapping, ensure a tiny safety gap
            if refined[i]["start"] <= refined[i-1]["end"]:
                overlap = refined[i-1]["end"] - refined[i]["start"]
                if overlap < 0.15:
                    q = self._find_quietest_point(wav, sr, refined[i]["start"], refined[i-1]["end"])
                    # Subtract a tiny bias (10ms) from end to favor the previous segment's silence
                    refined[i-1]["end"] = max(0.0, q - 0.01)
                    refined[i]["start"] = q
                else: 
                    mid = (refined[i-1]["end"] + refined[i]["start"]) / 2
                    refined[i-1]["end"] = max(0.0, mid - 0.01)
                    refined[i]["start"] = mid
            
            if refined[i]["end"] < refined[i]["start"]:
                refined[i]["end"] = refined[i]["start"] + 0.02
        return refined

    @staticmethod
    def _find_quietest_point(wav: Any, sr: int, start_t: float, end_t: float) -> float:
        """Find the timestamp with minimum RMS energy, biased towards the center."""
        import numpy as np
        
        if start_t >= end_t: return start_t
        
        max_t = len(wav) / sr
        start_t = max(0.0, min(start_t, max_t))
        end_t = max(0.0, min(end_t, max_t))
        
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        
        if end_sample - start_sample < 100: return (start_t + end_t) / 2
        
        chunk = wav[start_sample:end_sample]
        
        # Windowed RMS
        win_size = int(sr * 0.005) # 5ms
        hop_size = int(sr * 0.002) # 2ms
        
        n_frames = (len(chunk) - win_size) // hop_size + 1
        if n_frames <= 0: return (start_t + end_t) / 2
        
        # Pick the minimum RMS point with a bias toward the original estimated center
        # This prevents the cut from jumping too far into the next word if it's also quiet
        center_frame = n_frames / 2
        min_score = float('inf')
        best_frame = center_frame
        
        for i in range(n_frames):
            f_start = i * hop_size
            f_end = f_start + win_size
            f_rms = np.sqrt(np.mean(chunk[f_start:f_end]**2))
            
            # Distance penalty: 0.1% increase in "effective energy" for every frame away from center
            # This makes the algorithm prefer the Whisper-estimated boundary if silence quality is similar
            dist_penalty = 1.0 + (abs(i - center_frame) / n_frames) * 0.2
            score = f_rms * dist_penalty
            
            if score < min_score:
                min_score = score
                best_frame = i
        
        return start_t + (best_frame * hop_size + win_size / 2) / sr


    def _align_words_to_text(self, original_text: str, whisper_chunks: List[Dict]) -> List[Dict]:
        import difflib
        
        def clean(t):
            return re.sub(r'[。！？.!?，,、；;：:—…·\-\s\u3000]', '', t)

        segments = re.split(r'(?<=[。！？.!?，,、；;：:—…])\s*', original_text)
        segments = [s.strip() for s in segments if s.strip()]
        if not segments: segments = [original_text]

        # 1. Prepare original characters with segment info
        orig_chars = []
        for i, seg in enumerate(segments):
            for c in clean(seg):
                orig_chars.append({"char": c, "seg_idx": i})
        
        # 2. Extract Whisper characters with refined timestamps
        whsp_chars = []
        for i, chunk in enumerate(whisper_chunks):
            text = clean(chunk.get("text", ""))
            if not text: continue
            
            ts = chunk.get("timestamp", (None, None))
            start_t = ts[0] if ts[0] is not None else (whsp_chars[-1]["end"] if whsp_chars else 0.0)
            
            # Refine end time: if missing, use next start or small offset
            end_t = ts[1]
            if end_t is None:
                if i + 1 < len(whisper_chunks):
                    next_ts = whisper_chunks[i+1].get("timestamp", (None, None))
                    end_t = next_ts[0] if next_ts[0] is not None else (start_t + 0.2)
                else:
                    end_t = start_t + 0.2
            
            if end_t <= start_t: end_t = start_t + 0.1
            
            char_dur = (end_t - start_t) / len(text)
            for j, c in enumerate(text):
                whsp_chars.append({"char": c, "start": start_t + j * char_dur, "end": start_t + (j + 1) * char_dur})

        # 3. Robust Sequence Alignment
        matcher = difflib.SequenceMatcher(None, [x["char"] for x in orig_chars], [x["char"] for x in whsp_chars])
        
        # Map segment index to list of matched timestamps
        seg_times = {i: [] for i in range(len(segments))}
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    o_idx = i1 + k
                    w_idx = j1 + k
                    s_idx = orig_chars[o_idx]["seg_idx"]
                    seg_times[s_idx].append(whsp_chars[w_idx]["start"])
                    seg_times[s_idx].append(whsp_chars[w_idx]["end"])

        # 4. Initial Results Building
        results = []
        last_matched_end = 0.0
        for i, seg_text in enumerate(segments):
            clean_text = clean(seg_text)
            if not clean_text: continue
            
            times = sorted(seg_times[i])
            if times:
                s, e = times[0], times[-1]
                # Duration heuristic: force minimum 0.1s per character
                # This prevents "teleporting" if Whisper matched words too closely
                min_dur = len(clean_text) * 0.1
                if (e - s) < min_dur:
                    e = s + min_dur
                
                results.append({"text": clean_text, "start": round(s, 3), "end": round(e, 3)})
                last_matched_end = e
            else:
                # Unmatched: placeholder
                results.append({"text": clean_text, "start": -1.0, "end": -1.0})

        # 5. Global Interpolation & Non-decreasing constraint
        n_res = len(results)
        if n_res == 0: return []

        # Boundaries
        total_audio_dur = whsp_chars[-1]["end"] if whsp_chars else 0.0
        if results[0]["start"] < 0: results[0]["start"] = 0.0; results[0]["end"] = 0.0
        if results[-1]["end"] < 0: results[-1]["start"] = total_audio_dur; results[-1]["end"] = total_audio_dur

        # Forward pass: ensure no overlaps AND non-negative
        for i in range(n_res):
            if i > 0 and results[i]["start"] < results[i-1]["end"] and results[i]["start"] >= 0:
                results[i]["start"] = results[i-1]["end"]
                if results[i]["end"] < results[i]["start"] and results[i]["end"] >= 0:
                    results[i]["end"] = results[i]["start"]
            
            if results[i]["start"] >= 0 and results[i]["end"] < 0:
                # Find next known
                for j in range(i+1, n_res):
                    if results[j]["start"] >= 0:
                        results[i]["end"] = results[j]["start"]; break
                if results[i]["end"] < 0: results[i]["end"] = results[i]["start"]

        # Proportional GAP Filling (Final smoothing)
        idx = 0
        while idx < n_res:
            if results[idx]["start"] == results[idx]["end"]:
                gap_start_idx = idx
                while idx < n_res and results[idx]["start"] == results[idx]["end"]:
                    idx += 1
                
                t_start = results[gap_start_idx-1]["end"] if gap_start_idx > 0 else 0.0
                t_end = total_audio_dur 
                if idx < n_res: t_end = results[idx]["start"]
                
                if t_end > t_start:
                    gap_segs = results[gap_start_idx : idx]
                    total_c = sum(len(clean(r["text"])) for r in gap_segs)
                    curr = t_start
                    for r in gap_segs:
                        step = (len(clean(r["text"])) / total_c) * (t_end - t_start)
                        r["start"] = round(curr, 3); r["end"] = round(curr + step, 3)
                        curr += step
            else:
                idx += 1
        
        # Bridge small gaps between sentences to make it look like a Continuous stream
        # (Optional: Only if gap is < 0.5s)
        for i in range(1, n_res):
            gap = results[i]["start"] - results[i-1]["end"]
            if 0 < gap < 0.3:
                results[i-1]["end"] = (results[i-1]["end"] + results[i]["start"]) / 2
                results[i]["start"] = results[i-1]["end"]

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

    @staticmethod
    def _apply_fade(waveform: torch.Tensor, sr: int, fade_ms: int = 50) -> torch.Tensor:
        """Apply a micro-fade in/out to prevent clicks and trailing noise."""
        fade_samples = int(sr * fade_ms / 1000)
        if waveform.shape[-1] < 2 * fade_samples:
            return waveform
            
        # Linear fade
        fade_in = torch.linspace(0.0, 1.0, fade_samples, device=waveform.device)
        fade_out = torch.linspace(1.0, 0.0, fade_samples, device=waveform.device)
        
        # Clone to avoid in-place modification of shared buffers if any
        wav = waveform.clone()
        wav[..., :fade_samples] *= fade_in
        wav[..., -fade_samples:] *= fade_out
        return wav


class WhisperTranscribeNode:
    """
    🎤 Whisper Transcribe: Transcribe audio to text with timestamps.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "engine": (["Whisper", "SenseVoice"], {"default": "SenseVoice"}),
                "model_size": (["tiny", "base", "small", "medium", "large-v3"], {"default": "medium"}),
                "language": ("STRING", {"default": "zh", "placeholder": "语言代码 (zh, en, ja 等)"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "timestamps": (["none", "word", "sentence"], {"default": "sentence"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "timestamps_json")
    FUNCTION = "transcribe"
    CATEGORY = "Whisper"
    DESCRIPTION = "Transcribe audio to text using Whisper. Optionally include word or sentence-level timestamps."

    def transcribe(self, audio: Dict[str, Any], engine: str, model_size: str, language: str, device: str, timestamps: str) -> Tuple[str, str]:
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

        if engine == "SenseVoice":
            model = _get_sensevoice_model(device)
            print(f"[Whisper] (SenseVoice) Transcribing audio ({len(wav_np)/16000:.1f}s)...")
            res = model.generate(
                input=wav_np,
                cache={},
                language=language if language != "auto" else "auto",
                use_itn=True,
                batch_size_s=60,
            )
            
            full_text = ""
            chunks = []
            if res and isinstance(res, list) and len(res) > 0:
                item = res[0]
                full_text = item.get("text", "")
                
                # If we have sentence-level timestamps from VAD
                if "timestamp" in item:
                    for ts_entry in item["timestamp"]:
                        start_t, end_t = ts_entry[0] / 1000.0, ts_entry[1] / 1000.0
                        chunks.append({
                            "text": "", 
                            "timestamp": (start_t, end_t)
                        })
                else:
                    chunks.append({"text": full_text, "timestamp": (0.0, len(wav_np)/16000)})
        else:
            pipe = _get_whisper_pipeline(model_size, device, language)
            ts_param = "word" if timestamps == "word" else (True if timestamps == "sentence" else None)
            
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

        # Prepare wav_np for quietest point detection
        if wav.shape[0] > 1:
            wav_mono = wav.mean(dim=0)
        else:
            wav_mono = wav[0]
        wav_np = wav_mono.cpu().numpy()

        audio_list = []
        n = len(timestamps)
        
        for i in range(n):
            item = timestamps[i]
            idx = item.get("index", i + 1)
            text = item.get("text", "")
            
            # --- Quietest Point Logic ---
            # Search window: ±0.1s around Whisper boundaries (Tight Window)
            s_val = float(item.get("start", 0))
            if i > 0:
                prev_e = float(timestamps[i-1].get("end", s_val))
                # Search around the gap
                search_s = max(0.0, prev_e - 0.1)
                search_e = min(wav.shape[-1]/sr, s_val + 0.1)
                actual_start = WhisperAlignNode._find_quietest_point(wav_np, sr, search_s, search_e)
            else:
                actual_start = 0.0 
            
            e_val = float(item.get("end", s_val))
            if i < n - 1:
                next_s = float(timestamps[i+1].get("start", e_val))
                search_s = max(0.0, e_val - 0.1)
                search_e = min(wav.shape[-1]/sr, next_s + 0.1)
                actual_end = WhisperAlignNode._find_quietest_point(wav_np, sr, search_s, search_e)
            else:
                actual_end = wav.shape[-1] / sr
            
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
            # Apply Micro-fade
            segment = WhisperAlignNode._apply_fade(segment, sr, 20)
            
            audio_list.append({"waveform": segment, "sample_rate": sr})
            print(f"[Whisper] ✂️ Segment {idx} (Quiet-Point + Fade): [{actual_start:.3f}s ~ {actual_end:.3f}s] \"{text}\"")

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
        
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        # Prepare wav_np for quietest point detection
        if waveform.dim() == 3:
            wav_tmp = waveform[0]
        else:
            wav_tmp = waveform
        if wav_tmp.shape[0] > 1:
            wav_mono = wav_tmp.mean(dim=0)
        else:
            wav_mono = wav_tmp[0]
        wav_np = wav_mono.cpu().numpy()
        
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
            
            # --- Quiet Point Logic ---
            t_opening_end = float(opening_segments[-1].get("end", 0.0))
            if content_segments:
                t_content_start = float(content_segments[0].get("start", t_opening_end))
                # Tight window: Search ±0.1s around the boundary
                search_s = max(0.0, t_opening_end - 0.1)
                search_e = min(waveform.shape[-1]/sr, t_content_start + 0.1)
                split_time = WhisperAlignNode._find_quietest_point(wav_np, sr, search_s, search_e)
                
                # Sync timestamps to the actual acoustic cut
                opening_segments[-1]["end"] = round(split_time, 3)
                # Content will be re-zeroed below
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
            waveform_v = waveform # [batch, channels, samples]
        else:
            wav_opening = waveform[:, :split_sample]
            wav_content = waveform[:, split_sample:]
            waveform_v = waveform.unsqueeze(0)

        # Apply Fades
        wav_opening = WhisperAlignNode._apply_fade(wav_opening, sr, 20)
        wav_content = WhisperAlignNode._apply_fade(wav_content, sr, 20)

        audio_opening = {"waveform": wav_opening, "sample_rate": sr}
        audio_content = {"waveform": wav_content, "sample_rate": sr}
        
        json_opening, srt_opening = WhisperAlignNode._format_output(opening_segments)
        json_content, srt_content = WhisperAlignNode._format_output(adjusted_content)
        
        print(f"[Whisper] ✂️ Quietest-Point Split complete. Gap cut at {split_time:.3f}s")
        return (audio_opening, json_opening, srt_opening, audio_content, json_content, srt_content)

class WhisperSaveSubtitlesNode:
    """
     Whisper Save Subtitles:
    Saves subtitle strings (SRT, JSON, etc.) to a file. 
    Bypasses rigid path security checks from other node suites.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "output_dir": ("STRING", {"default": "subtitles"}),
                "filename": ("STRING", {"default": "captions.srt"}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "Whisper"
    OUTPUT_NODE = True

    def save(self, text, output_dir, filename, add_timestamp):
        import time
        
        # Get base output directory
        base_output = folder_paths.get_output_directory()
        full_dir = os.path.abspath(os.path.join(base_output, output_dir))
        
        if not os.path.exists(full_dir):
            os.makedirs(full_dir, exist_ok=True)
            
        name, ext = os.path.splitext(filename)
        if add_timestamp:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{ts}{ext}"
            
        file_path = os.path.join(full_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"[Whisper]  Subtitles saved to: {file_path}")
        return (file_path,)
