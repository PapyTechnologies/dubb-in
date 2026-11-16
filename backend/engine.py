import sys
import os

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    try:
        if sys.stdout:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{APP_DIR};" + os.environ.get("PATH", "")

FFMPEG_WINGET = Path(
    r"C:\Users\topar\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
)
if FFMPEG_WINGET.exists() and str(FFMPEG_WINGET) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{FFMPEG_WINGET};" + os.environ.get("PATH", "")

os.environ["COQUI_TOS_AGREED"] = "1"

RATE = 24000

LANGUAGES = {
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Italien": "it",
    "Portugais": "pt",
    "Polonais": "pl",
    "Turc": "tr",
    "Russe": "ru",
    "Néerlandais": "nl",
    "Tchèque": "cs",
    "Arabe": "ar",
    "Chinois": "zh-cn",
    "Japonais": "ja",
    "Hongrois": "hu",
    "Coréen": "ko",
}


def run(args, timeout=1800, cwd=None):
    process = subprocess.run(
        [str(argument) for argument in args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )
    if process.returncode:
        raise RuntimeError(
            process.stderr[-6000:]
            or "La commande a échoué."
        )
    return process.stdout


def probe(path):
    output = run([
        "ffprobe",
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        path,
    ])
    return json.loads(output)


def duration(path):
    value = float(
        probe(path)["format"]["duration"]
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Durée non valide.")
    return value


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def detect_hardware_profile():
    import psutil
    import torch
    cpu_count = psutil.cpu_count(logical=True) or 4
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else None
    vram_gb = 0.0
    if has_cuda:
        try:
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        except Exception:
            vram_gb = 0.0

    if has_cuda and vram_gb >= 6.0:
        tier = "Ultra (GPU Dédié)"
        whisper_model = "large-v3-turbo"
        demucs_capable = True
        xtts_device = "cuda"
        desc = f"NVIDIA {gpu_name} ({vram_gb} Go VRAM) — Accélération matérielle maximale"
    elif has_cuda and vram_gb >= 4.0:
        tier = "Performant (GPU 4 Go)"
        whisper_model = "small"
        demucs_capable = True
        xtts_device = "cuda"
        desc = f"NVIDIA {gpu_name} ({vram_gb} Go VRAM) — Mode optimisé GPU"
    elif has_cuda and vram_gb > 0:
        tier = "Hybride (GPU Léger + CPU)"
        whisper_model = "small"
        demucs_capable = True
        xtts_device = "cpu"
        desc = f"NVIDIA {gpu_name} ({vram_gb} Go VRAM) — Whisper GPU + XTTS CPU protégé"
    elif cpu_count >= 8 and ram_gb >= 15:
        tier = "Pro (CPU Puissant)"
        whisper_model = "small"
        demucs_capable = True
        xtts_device = "cpu"
        desc = f"CPU {cpu_count} cœurs, {ram_gb} Go RAM — Profil haute précision"
    elif cpu_count >= 4 and ram_gb >= 8:
        tier = "Équilibré (Standard)"
        whisper_model = "small"
        demucs_capable = True
        xtts_device = "cpu"
        desc = f"CPU {cpu_count} cœurs, {ram_gb} Go RAM — Profil optimisé fluidité"
    else:
        tier = "Économique (Portable/Léger)"
        whisper_model = "base"
        demucs_capable = False
        xtts_device = "cpu"
        desc = f"CPU {cpu_count} cœurs, {ram_gb} Go RAM — Profil basse consommation"
    return {
        "tier": tier,
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "recommended_whisper": whisper_model,
        "xtts_device": xtts_device,
        "demucs_capable": demucs_capable,
        "description": desc,
    }


def check_tools():
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise RuntimeError(
                f"{name} est absent du PATH."
            )


def validate(rows, total, require_text=True):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Aucun segment.")
    previous_end = 0.0
    for number, row in enumerate(rows, 1):
        start = float(row["start"])
        end = float(row["end"])
        finite = all(
            math.isfinite(value)
            for value in (start, end)
        )
        if not finite:
            raise ValueError(
                f"Horodatage non fini : segment {number}."
            )
        if not 0 <= start < end <= total + 0.05:
            raise ValueError(
                f"Horodatages invalides : segment {number}."
            )
        if start < previous_end:
            raise ValueError(
                f"Chevauchement : segment {number}."
            )
        text = row.get("target")
        if require_text:
            valid_text = (
                isinstance(text, str)
                and bool(text.strip())
                and len(text) <= 600
            )
            if not valid_text:
                raise ValueError(
                    f"Traduction vide ou trop longue : "
                    f"segment {number}."
                )
        previous_end = end


def stamp(seconds):
    milliseconds = round(
        float(seconds) * 1000
    )
    hours, milliseconds = divmod(
        milliseconds,
        3600000,
    )
    minutes, milliseconds = divmod(
        milliseconds,
        60000,
    )
    seconds, milliseconds = divmod(
        milliseconds,
        1000,
    )
    return (
        f"{hours:02}:{minutes:02}:"
        f"{seconds:02},{milliseconds:03}"
    )


def srt(rows):
    blocks = []
    for number, row in enumerate(rows, 1):
        blocks.append(
            f"{number}\n"
            f"{stamp(row['start'])} --> "
            f"{stamp(row['end'])}\n"
            f"{row['target'].strip()}\n"
        )
    return "\n".join(blocks)


def generate_ass_subtitles(rows, output_path, style="hormozi_yellow"):
    def ass_time(seconds):
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        if cs >= 100:
            cs = 99
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    all_text = " ".join(str(r.get("target", "") or r.get("source", "")) for r in rows)
    has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF or 0x3040 <= ord(c) <= 0x30FF or 0xAC00 <= ord(c) <= 0xD7AF for c in all_text)
    is_rtl = any(
        (0x0590 <= ord(c) <= 0x05FF)
        or (0x0600 <= ord(c) <= 0x06FF)
        or (0x0750 <= ord(c) <= 0x077F)
        or (0x08A0 <= ord(c) <= 0x08FF)
        for c in all_text
    )

    spacing = 1
    primary_color = "&H00FFFFFF&"

    if is_rtl:
        font_name = "Segoe UI"
        font_size = 50
        outline_size = 5
        spacing = 0
        if style == "hormozi_green":
            primary_color = "&H0000FF00&"
        elif style == "mrbeast":
            primary_color = "&H00FFFF08&"
        elif style == "minimal":
            primary_color = "&H00FFFFFF&"
        else:
            primary_color = "&H0000FFFF&"
        highlight_color = primary_color
    elif has_cjk:
        font_name = "Microsoft YaHei"
        font_size = 46
        outline_size = 5
        highlight_color = "&H0000FFFF&"
    elif style == "hormozi_green":
        font_name = "Arial Black"
        font_size = 48
        highlight_color = "&H0000FF00&"
        outline_size = 6
    elif style == "mrbeast":
        font_name = "Impact"
        font_size = 52
        highlight_color = "&H00FFFF08&"
        outline_size = 7
    elif style == "minimal":
        font_name = "Arial"
        font_size = 38
        highlight_color = "&H0000FFFF&"
        outline_size = 3
    else:
        font_name = "Arial Black"
        font_size = 48
        highlight_color = "&H0000FFFF&"
        outline_size = 6

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTokStyle,{font_name},{font_size},{primary_color},&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,{spacing},0,1,{outline_size},2,2,40,40,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for row in rows:
        text = str(row.get("target", "") or row.get("source", "")).strip()
        if not text:
            continue
        words = text.split()
        if not words:
            continue
        seg_start = float(row["start"])
        seg_end = float(row["end"])
        seg_duration = max(0.2, seg_end - seg_start)

        total_len = sum(max(1, len(w)) for w in words)
        word_timings = []
        cur_time = seg_start
        for w in words:
            w_dur = (max(1, len(w)) / total_len) * seg_duration
            word_timings.append((w, cur_time, cur_time + w_dur))
            cur_time += w_dur

        if is_rtl:
            chunk_size = 4
            for i in range(0, len(word_timings), chunk_size):
                chunk = word_timings[i:i + chunk_size]
                c_start = chunk[0][1]
                c_end = chunk[-1][2]
                c_text = " ".join(cw for cw, _, _ in chunk)
                events.append(f"Dialogue: 0,{ass_time(c_start)},{ass_time(c_end)},TikTokStyle,,0,0,0,,{c_text}")
        else:
            chunk_size = 3
            for i in range(0, len(word_timings), chunk_size):
                chunk = word_timings[i:i + chunk_size]
                for active_idx, (w, w_s, w_e) in enumerate(chunk):
                    formatted_words = []
                    for idx, (cw, _, _) in enumerate(chunk):
                        u_word = cw.upper()
                        if idx == active_idx:
                            formatted_words.append(f"{{\\c{highlight_color}\\b1}}{u_word}{{\\c&H00FFFFFF&\\b0}}")
                        else:
                            formatted_words.append(u_word)
                    line_text = " ".join(formatted_words)
                    events.append(f"Dialogue: 0,{ass_time(w_s)},{ass_time(w_e)},TikTokStyle,,0,0,0,,{line_text}")

    out_file = Path(output_path)
    out_file.write_text(header + "\n".join(events), encoding="utf-8")
    return out_file


def prepare(project):
    project = Path(project)
    check_tools()
    information = probe(
        project / "video.input"
    )
    for kind in ("audio", "video"):
        found = any(
            stream.get("codec_type") == kind
            for stream in information["streams"]
        )
        if not found:
            raise ValueError(
                f"Piste manquante : {kind}."
            )
    total = duration(
        project / "video.input"
    )
    free_space = shutil.disk_usage(
        project
    ).free
    if free_space < 2 * 1024**3:
        raise ValueError(
            "Au moins 2 Go libres sont requis "
            "pour traiter ce projet."
        )
    reference_duration = duration(
        project / "reference.input"
    )
    if not 3 <= reference_duration <= 60:
        raise ValueError(
            "La référence vocale doit durer "
            "entre 3 et 60 secondes."
        )
    conversions = [
        ("video.input", "source.wav", 16000),
        ("reference.input", "reference.wav", RATE),
    ]
    for source, destination, rate in conversions:
        run([
            "ffmpeg",
            "-y",
            "-v", "error",
            "-i", project / source,
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", rate,
            "-c:a", "pcm_s16le",
            project / destination,
        ])
    return total


def merge_segments(segments, max_gap=0.8, max_duration=12.0):
    if not segments:
        return []
    merged = []
    current = None
    for s in segments:
        text = str(s.get("source", "")).strip()
        if not text:
            continue
        if not current:
            current = dict(s)
            continue
        prev_text = str(current.get("source", "")).strip()
        gap = float(s["start"]) - float(current["end"])
        dur = float(s["end"]) - float(current["start"])

        ends_sentence = prev_text.endswith((".", "!", "?"))
        last_word = prev_text.split()[-1].lower() if prev_text.split() else ""
        is_continuation = (
            not ends_sentence
            or len(text.split()) <= 3
            or text[0].islower()
            or last_word in {"and", "or", "because", "to", "this", "can", "will", "the", "a", "so", "but", "if", "that", "as"}
        )

        if gap <= max_gap and dur <= max_duration and is_continuation:
            current["end"] = round(float(s["end"]), 3)
            current["source"] = f"{current['source']} {text}".strip()
            if current.get("target") or s.get("target"):
                t1 = str(current.get("target", "")).strip()
                t2 = str(s.get("target", "")).strip()
                current["target"] = f"{t1} {t2}".strip()
        else:
            merged.append(current)
            current = dict(s)
    if current:
        merged.append(current)
    return merged


def extract_background_stems(project):
    project = Path(project)
    bgm_path = project / "bgm.wav"
    if bgm_path.exists() and bgm_path.stat().st_size > 1000:
        return bgm_path

    source_path = project / "source.wav"
    if not source_path.exists():
        raise FileNotFoundError("source.wav introuvable.")

    from demucs.api import Separator
    import soundfile as sf
    import numpy as np

    separator = Separator(model="htdemucs", segment=7)
    origin, separated = separator.separate_audio_file(str(source_path))

    drums = separated.get("drums")
    bass = separated.get("bass")
    other = separated.get("other")
    bgm_tensor = drums + bass + other

    bgm_np = bgm_tensor.cpu().numpy()
    if bgm_np.ndim == 2:
        bgm_np = bgm_np.T

    peak = float(np.max(np.abs(bgm_np)))
    if peak > 0.95:
        bgm_np *= (0.95 / peak)

    sf.write(str(bgm_path), bgm_np, separator.samplerate, subtype="PCM_16")
    return bgm_path


def transcribe(project, model_name=None, progress=None):
    from faster_whisper import WhisperModel
    project = Path(project)
    total_audio_duration = duration(project / "source.wav")
    hw = detect_hardware_profile()
    if not model_name:
        model_name = hw["recommended_whisper"]
    dev = "cuda" if hw["has_cuda"] else "cpu"
    comp = "float16" if hw["has_cuda"] else "int8"
    cpu_threads = hw["cpu_count"] if dev == "cpu" else 4
    model = WhisperModel(
        model_name,
        device=dev,
        compute_type=comp,
        cpu_threads=cpu_threads,
    )
    iterator, information = model.transcribe(
        str(project / "source.wav"),
        vad_filter=True,
        beam_size=5,
    )
    rows = []
    for segment in iterator:
        text = segment.text.strip()
        if not text:
            continue
        rows.append({
            "start": round(
                float(segment.start),
                3,
            ),
            "end": round(
                float(segment.end),
                3,
            ),
            "source": text,
            "target": "",
        })
        if progress and total_audio_duration > 0:
            fraction = min(0.99, float(segment.end) / total_audio_duration)
            progress(fraction)
    if not rows:
        raise ValueError(
            "Aucune parole détectée."
        )
    rows = merge_segments(rows)
    if progress:
        progress(1.0)
    document = {
        "source_language": information.language,
        "target_language": "en",
        "segments": rows,
    }
    save(
        project / "document.json",
        document,
    )
    return document


def translate(document, target):
    import json
    import time
    import urllib.parse
    import urllib.request

    result = json.loads(
        json.dumps(document)
    )

    def _translate_single(text, tgt):
        q = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={tgt}&dt=t&q={q}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "".join([part[0] for part in data[0] if part and part[0]]).strip()

    texts = [str(row["source"]).strip() for row in result["segments"]]
    translated_texts = []

    # Attempt smart batching by chunks of 15 lines with newline delimiter
    chunk_size = 15
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        combined = "\n".join(chunk)
        success = False
        try:
            trans_chunk = _translate_single(combined, target)
            lines = [line.strip() for line in trans_chunk.splitlines() if line.strip()]
            if len(lines) == len(chunk):
                translated_texts.extend(lines)
                success = True
        except Exception:
            pass

        if not success:
            for text in chunk:
                if len(text) > 4500:
                    raise ValueError("Un segment de transcription est trop long.")
                translated = ""
                for attempt in range(3):
                    try:
                        translated = _translate_single(text, target)
                        if translated:
                            break
                    except Exception:
                        time.sleep(0.3 * (attempt + 1))
                if not translated:
                    try:
                        from deep_translator import GoogleTranslator
                        translated = GoogleTranslator(source="auto", target=target).translate(text)
                    except Exception:
                        translated = text
                translated_texts.append(translated)
                time.sleep(0.05)

    for row, translated in zip(result["segments"], translated_texts):
        row["target"] = translated or str(row["source"])

    result["target_language"] = target
    return adapt_document_dialogues(result, target)


def compact_for_dubbing(text, duration_sec, language="fr"):
    if not text or duration_sec <= 0:
        return text

    words = text.split()
    target_words = max(3, int(duration_sec * 2.3))
    if len(words) <= target_words:
        return text

    import re
    res = text
    lang_code = language.lower()[:2]
    if lang_code in ("fr", "français", "french"):
        rules = [
            (r"\b[Tt]out ce que vous avez à faire (est de|pour|c'est de|c'est)?\s*", "Il suffit de "),
            (r"\b[Ii]l vous suffit de\s*", ""),
            (r"\b[Vv]ous pouvez vous inscrire dès maintenant\b", "Inscrivez-vous maintenant"),
            (r"\b[Vv]ous pouvez (simplement )?\b", ""),
            (r"\b[Vv]ous n'avez qu'à\b", "Il suffit de "),
            (r"\b[Dd]ès maintenant\b", "maintenant"),
            (r"\ble lien (se trouve|est) dans la description\b", "lien en description"),
            (r"\bafin de\s*", "pour "),
            (r"\best en train de\s*", ""),
            (r"\bpar conséquent\b", "donc"),
            (r"\btout simplement\b", "simplement"),
            (r"\ben réalité\b", ""),
            (r"\bà ce moment-là\b", "alors"),
            (r"\bc'est-à-dire que\b", "donc"),
            (r"\bil est possible de\b", "on peut"),
            (r"\best-ce que vous pouvez\b", "pouvez-vous"),
            (r"\bje suis en train d'\b", "je vais "),
            (r"\bn'hésitez pas à\b", ""),
            (r"\bde manière à\b", "pour"),
        ]
        for pat, rep in rules:
            res = re.sub(pat, rep, res, flags=re.IGNORECASE)

    cur_words = res.split()
    if len(cur_words) > target_words:
        fillers = {
            "fr": ["vraiment", "absolument", "totalement", "actuellement", "exactement", "particulièrement", "simplement", "en fait"],
            "en": ["really", "actually", "basically", "literally", "absolutely", "definitely", "simply"],
            "es": ["realmente", "básicamente", "absolutamente", "simplemente", "actualmente"],
        }
        f_list = fillers.get(lang_code, fillers.get("fr", []))
        filtered = [w for w in cur_words if w.lower().strip(".,;:!?") not in f_list]
        if len(filtered) >= 3:
            res = " ".join(filtered)

    res = re.sub(r"\s+", " ", res).strip(" ,")
    if res:
        res = res[0].upper() + res[1:]
    return res


def adapt_document_dialogues(document, target_language="fr"):
    new_doc = json.loads(json.dumps(document))
    for row in new_doc.get("segments", []):
        dur = float(row.get("end", 0)) - float(row.get("start", 0))
        target_text = str(row.get("target", "")).strip()
        if target_text:
            row["target"] = compact_for_dubbing(target_text, dur, language=target_language)
    return new_doc


def build_atempo_chain(factor):
    f = float(factor)
    f = max(0.25, min(4.0, f))
    filters = []
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    filters.append(f"atempo={f:.6f}")
    return ",".join(filters)


def stretch_audio(audio_data, factor):
    """Ajuste le tempo audio avec FFmpeg atempo sans altérer la hauteur tonale (pitch)."""
    if abs(factor - 1.0) < 0.03:
        return audio_data
    import tempfile
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_path = in_f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = out_f.name
    try:
        sf.write(in_path, audio_data, RATE)
        f_chain = build_atempo_chain(factor)
        run([
            "ffmpeg", "-y", "-v", "error",
            "-i", in_path,
            "-af", f_chain,
            "-ar", str(RATE),
            out_path,
        ])
        stretched, _ = sf.read(out_path, dtype="float32")
        return stretched
    except Exception:
        return audio_data
    finally:
        for p in (in_path, out_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def render(project, document, progress=None, keep_background=False, subtitle_style="none", vocal_speed=1.12):
    import hashlib
    import numpy as np
    import soundfile as sf
    import torch
    from TTS.api import TTS
    project = Path(project)
    total = duration(
        project / "video.input"
    )
    rows = document["segments"]
    validate(rows, total)
    hw = detect_hardware_profile()
    device = hw.get("xtts_device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        try:
            torch.set_num_threads(hw["cpu_count"])
        except Exception:
            pass
    model = TTS(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    ).to(device)
    target = document["target_language"]
    if target not in model.languages:
        raise ValueError(
            "La langue n'est pas prise en charge "
            "par le modèle chargé."
        )
    track = np.zeros(
        math.ceil(total * RATE),
        dtype=np.float32,
    )
    cache = project / "cache"
    cache.mkdir(exist_ok=True)
    speaker_wav_file = project / "reference.wav"
    ref_dur = duration(speaker_wav_file)
    if ref_dur > 16.0:
        calibrated_ref = project / "reference_calibrated.wav"
        if not calibrated_ref.exists():
            run([
                "ffmpeg", "-y", "-v", "error",
                "-ss", "3.0",
                "-i", str(speaker_wav_file),
                "-t", "12.0",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar", "24000",
                "-ac", "1",
                str(calibrated_ref),
            ])
        speaker_wav_file = calibrated_ref

    reference_hash = hashlib.sha256(
        speaker_wav_file.read_bytes()
    ).hexdigest()

    # Détection et chargement de l'empreinte vocale neuronale (.dvf) si disponible
    fp_file = project / "fingerprint.dvf"
    use_neural_fp = fp_file.exists()
    gpt_cond_latent = None
    speaker_embedding = None
    if use_neural_fp:
        try:
            from voice_learning import VoiceFingerprintEngine
            gpt_cond_latent, speaker_embedding, _ = VoiceFingerprintEngine.load_fingerprint(str(fp_file), device=device)
        except Exception:
            use_neural_fp = False
    vocal_speed = float(vocal_speed)
    if vocal_speed < 0.5 or vocal_speed > 2.0:
        vocal_speed = 1.12
    last_end_sample = 0
    aligned_rows = []

    for index, row in enumerate(rows):
        identity = json.dumps(
            [
                row["target"],
                target,
                reference_hash,
                "xtts-v2-natural-warmth",
                f"speed_{vocal_speed:.2f}",
                "neural_fp" if use_neural_fp else "raw_wav",
            ],
            ensure_ascii=False,
        )
        key = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()
        raw = cache / f"{key}.wav"
        if not raw.exists():
            temporary_audio = (
                cache / f"{key}.part.wav"
            )
            if use_neural_fp and gpt_cond_latent is not None:
                out = model.synthesizer.tts_model.inference(
                    text=row["target"].strip(),
                    language=target,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.75,
                    repetition_penalty=5.0,
                    top_p=0.85,
                    speed=vocal_speed,
                    enable_text_splitting=False,
                )
                wav_tensor = out["wav"]
                if isinstance(wav_tensor, torch.Tensor):
                    wav_data = wav_tensor.detach().cpu().numpy()
                else:
                    wav_data = np.array(wav_tensor, dtype=np.float32)
                sf.write(str(temporary_audio), wav_data, RATE)
            else:
                model.tts_to_file(
                    text=row["target"].strip(),
                    speaker_wav=str(speaker_wav_file),
                    language=target,
                    file_path=str(temporary_audio),
                    split_sentences=False,
                    temperature=0.75,
                    repetition_penalty=5.0,
                    top_p=0.85,
                    speed=vocal_speed,
                )
            temporary_audio.replace(raw)

        audio, rate = sf.read(
            raw,
            dtype="float32",
        )
        if rate != RATE:
            raise RuntimeError(
                "Fréquence audio inattendue."
            )

        # Nettoyage propre des silences de bordure
        non_silent = np.where(np.abs(audio) > 0.008)[0]
        if len(non_silent) > 0:
            first_speech = max(0, non_silent[0] - int(0.015 * RATE))
            last_speech = min(len(audio), non_silent[-1] + int(0.06 * RATE))
            audio = audio[first_speech:last_speech]

        # Calcul de la fenêtre temporelle allouée pour ce segment
        seg_start = float(row["start"])
        seg_end = float(row["end"])
        target_dur = max(0.2, seg_end - seg_start)
        next_start = float(rows[index + 1]["start"]) if index + 1 < len(rows) else total

        # Fenêtre disponible avant collision avec la prochaine prise de parole
        available_dur = max(target_dur, next_start - seg_start - 0.05)
        available_dur = min(available_dur, max(0.3, total - seg_start))

        # Ajustement dynamique du débit : compression temporelle FFmpeg atempo si nécessaire
        curr_dur = len(audio) / RATE
        if curr_dur > available_dur and available_dur > 0.3:
            ratio = curr_dur / available_dur
            stretch_factor = min(1.48, max(1.0, ratio))
            if stretch_factor > 1.04:
                audio = stretch_audio(audio, stretch_factor)
                curr_dur = len(audio) / RATE

        # Si l'audio dépasse encore la fenêtre disponible, on tronque doucement avec un fondu
        max_samples = int(available_dur * RATE)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Micro-fondu anti-clic de 8ms
        fade_samples = int(0.008 * RATE)
        if len(audio) > 2 * fade_samples:
            audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        # Positionnement précis synchronisé à la timeline originale
        target_start_sample = round(seg_start * RATE)
        min_start_sample = last_end_sample + int(0.03 * RATE) if last_end_sample > 0 else 0
        actual_start_sample = max(target_start_sample, min_start_sample)

        total_track_samples = len(track)
        if actual_start_sample < total_track_samples:
            write_end = min(total_track_samples, actual_start_sample + len(audio))
            track[actual_start_sample:write_end] = audio[:write_end - actual_start_sample]
            last_end_sample = write_end
        else:
            last_end_sample = total_track_samples

        aligned_rows.append({
            "start": round(actual_start_sample / RATE, 3),
            "end": round(last_end_sample / RATE, 3),
            "source": row.get("source", ""),
            "target": row.get("target", ""),
        })

        if progress:
            try:
                progress((index + 1) / len(rows), index + 1, len(rows))
            except TypeError:
                progress((index + 1) / len(rows))
    peak = float(
        np.max(np.abs(track))
    )
    if peak > 0.98:
        track *= 0.98 / peak
    sf.write(
        project / "dub.wav",
        track,
        RATE,
        subtype="PCM_16",
    )
    temporary_video = (
        project / "export.part.mp4"
    )
    has_subtitles = subtitle_style and subtitle_style != "none"
    if has_subtitles:
        generate_ass_subtitles(aligned_rows, project / "subtitles_tiktok.ass", style=subtitle_style)

    final_audio_dur = last_end_sample / RATE
    final_total = total
    extra_pad = 0.0
    tpad_prefix = ""

    if keep_background:
        extract_background_stems(project)
        bgm_path = project / "bgm.wav"
        if has_subtitles:
            filter_str = (
                f"[0:v]{tpad_prefix}ass=subtitles_tiktok.ass[vout];"
                "[2:a]volume=0.30[bg];"
                "[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            v_map = "[vout]"
        else:
            filter_str = (
                f"[0:v]{tpad_prefix}null[vout];" if tpad_prefix else "0:v:0"
            )
            if tpad_prefix:
                filter_str = (
                    f"[0:v]{tpad_prefix}null[vout];"
                    "[2:a]volume=0.30[bg];"
                    "[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                )
                v_map = "[vout]"
            else:
                filter_str = (
                    "[2:a]volume=0.30[bg];"
                    "[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                )
                v_map = "0:v:0"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-i", str(project / "video.input"),
            "-i", str(project / "dub.wav"),
            "-i", str(bgm_path),
            "-filter_complex", filter_str,
            "-map", v_map,
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(final_total),
            "-metadata",
            f"comment=Doublage vocal genere par IA (Demucs BGM | {subtitle_style})",
            "-movflags", "+faststart",
            str(temporary_video),
        ]
    else:
        if has_subtitles:
            filter_str = f"[0:v]{tpad_prefix}ass=subtitles_tiktok.ass[vout]"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-v", "error",
                "-i", str(project / "video.input"),
                "-i", str(project / "dub.wav"),
                "-filter_complex", filter_str,
                "-map", "[vout]",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(final_total),
                "-metadata",
                f"comment=Doublage vocal genere par IA (Voix pure | {subtitle_style})",
                "-movflags", "+faststart",
                str(temporary_video),
            ]
        else:
            if tpad_prefix:
                filter_str = f"[0:v]{tpad_prefix}null[vout]"
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-v", "error",
                    "-i", str(project / "video.input"),
                    "-i", str(project / "dub.wav"),
                    "-filter_complex", filter_str,
                    "-map", "[vout]",
                    "-map", "1:a:0",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-t", str(final_total),
                    "-metadata",
                    f"comment=Doublage vocal genere par IA (Voix pure | {subtitle_style})",
                    "-movflags", "+faststart",
                    str(temporary_video),
                ]
            else:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-v", "error",
                    "-i", str(project / "video.input"),
                    "-i", str(project / "dub.wav"),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-t", str(final_total),
                    "-metadata",
                    f"comment=Doublage vocal genere par IA (Voix pure | {subtitle_style})",
                    "-movflags", "+faststart",
                    str(temporary_video),
                ]
    run(ffmpeg_cmd, cwd=project)
    exported_duration = duration(
        temporary_video
    )
    if abs(exported_duration - final_total) > 2.0:
        raise RuntimeError(
            "La durée exportée est inattendue."
        )
    temporary_video.replace(
        project / "export.mp4"
    )
    (
        project / "subtitles.srt"
    ).write_text(
        srt(rows),
        encoding="utf-8",
    )
    save(
        project / "rendered_document.json",
        document,
    )
    save(
        project / "report.json",
        {
            "duration": total,
            "segments": len(rows),
            "target_language": target,
            "synthetic_audio": True,
            "lip_sync": False,
            "original_background_audio": False,
            "voice_mode": "single_authorized_reference",
        },
    )


VOICES_DIR = Path(__file__).resolve().parent / "voices"
DB_PATH = VOICES_DIR / "voices.db"


def init_voices_db():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from voice_learning import VoiceFingerprintEngine
        VoiceFingerprintEngine.init_db()
    except Exception:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    duration REAL NOT NULL
                )
            """)
            conn.commit()


def list_voice_profiles():
    init_voices_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, created_at, source_type, source_url, filename, sha256, duration, "
            "is_neural_fingerprint, f0_median, vocal_type, clarity_score, snr_db, sample_count, dvf_filename "
            "FROM voice_profiles ORDER BY name ASC"
        )
        rows = cursor.fetchall()
        profiles = []
        for r in rows:
            wav_path = VOICES_DIR / r[5]
            if wav_path.exists():
                is_neural = bool(r[8]) if len(r) > 8 and r[8] else False
                f0 = float(r[9]) if len(r) > 9 and r[9] is not None else 0.0
                vocal_type = str(r[10]) if len(r) > 10 and r[10] else "Standard"
                clarity = float(r[11]) if len(r) > 11 and r[11] is not None else 80.0
                snr = float(r[12]) if len(r) > 12 and r[12] is not None else 25.0
                sample_count = int(r[13]) if len(r) > 13 and r[13] is not None else 1
                dvf_name = r[14] if len(r) > 14 else None
                profiles.append({
                    "id": r[0],
                    "name": r[1],
                    "created_at": r[2],
                    "source_type": r[3],
                    "source_url": r[4],
                    "filename": r[5],
                    "sha256": r[6],
                    "duration": r[7],
                    "is_neural_fingerprint": is_neural,
                    "f0_median": f0,
                    "vocal_type": vocal_type,
                    "clarity_score": clarity,
                    "snr_db": snr,
                    "sample_count": sample_count,
                    "dvf_filename": dvf_name,
                    "path": wav_path,
                })
        return profiles


def add_voice_profile(name, source_path_or_bytes, source_type="upload", source_url=None, start_time=0.0, duration_seconds=None):
    init_voices_db()
    name = (name or "").strip()
    if not name:
        raise ValueError("Le nom du profil vocal ne peut pas être vide.")

    temp_input = VOICES_DIR / f"temp_{uuid.uuid4().hex}.raw"
    target_filename = f"{uuid.uuid4().hex}.wav"
    target_path = VOICES_DIR / target_filename

    try:
        if isinstance(source_path_or_bytes, (bytes, bytearray)):
            temp_input.write_bytes(source_path_or_bytes)
        else:
            shutil.copyfile(source_path_or_bytes, temp_input)

        total_dur = duration(temp_input)
        if total_dur < 3:
            raise ValueError(
                f"L'enregistrement est trop court (minimum 3 secondes, durée : {total_dur:.1f}s)."
            )

        cmd = ["ffmpeg", "-y", "-v", "error"]
        if start_time is not None and float(start_time) > 0:
            cmd.extend(["-ss", str(float(start_time))])
        cmd.extend(["-i", str(temp_input)])

        if duration_seconds is not None and float(duration_seconds) > 0:
            eff_dur = min(float(duration_seconds), 60.0)
            cmd.extend(["-t", str(eff_dur)])
        elif total_dur > 60:
            cmd.extend(["-t", "30.0"])

        cmd.extend([
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(RATE),
            "-c:a", "pcm_s16le",
            str(target_path),
        ])
        run(cmd)

        final_dur = duration(target_path)
        sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
        created = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO voice_profiles (name, created_at, source_type, source_url, filename, sha256, duration) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, created, source_type, source_url, target_filename, sha, final_dur),
            )
            conn.commit()

        return {
            "name": name,
            "filename": target_filename,
            "path": target_path,
            "duration": final_dur,
            "sha256": sha,
        }
    finally:
        if temp_input.exists():
            temp_input.unlink()


def delete_voice_profile(profile_id_or_name):
    init_voices_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if isinstance(profile_id_or_name, int) or (isinstance(profile_id_or_name, str) and profile_id_or_name.isdigit()):
            cursor.execute("SELECT id, filename, dvf_filename FROM voice_profiles WHERE id = ?", (int(profile_id_or_name),))
        else:
            cursor.execute("SELECT id, filename, dvf_filename FROM voice_profiles WHERE name = ?", (str(profile_id_or_name),))
        row = cursor.fetchone()
        if row:
            p_id, filename, dvf_filename = row
            file_to_del = VOICES_DIR / filename
            if file_to_del.exists():
                file_to_del.unlink()
            if dvf_filename:
                dvf_to_del = VOICES_DIR / "fingerprints" / dvf_filename
                if dvf_to_del.exists():
                    dvf_to_del.unlink()
            conn.execute("DELETE FROM voice_profiles WHERE id = ?", (p_id,))
            conn.commit()


def download_media_from_url(url, output_path, extract_audio=False, start_time=None, duration_seconds=None):
    import yt_dlp
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_id = uuid.uuid4().hex
    temp_target = output_path.parent / f"ytdl_{temp_id}"

    ydl_opts = {
        "format": "bestaudio/best" if extract_audio else "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": f"{str(temp_target)}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = Path(ydl.prepare_filename(info))

    if not downloaded_file.exists():
        matches = list(output_path.parent.glob(f"ytdl_{temp_id}.*"))
        if matches:
            downloaded_file = matches[0]
        else:
            raise RuntimeError("Échec du téléchargement du média depuis l'URL.")

    try:
        ffmpeg_cmd = ["ffmpeg", "-y", "-v", "error"]
        if start_time is not None and float(start_time) > 0:
            ffmpeg_cmd.extend(["-ss", str(float(start_time))])
        ffmpeg_cmd.extend(["-i", str(downloaded_file)])
        if duration_seconds is not None and float(duration_seconds) > 0:
            ffmpeg_cmd.extend(["-t", str(float(duration_seconds))])

        if extract_audio:
            ffmpeg_cmd.extend([
                "-map", "0:a:0",
                "-vn",
                "-ac", "1",
                "-ar", str(RATE),
                "-c:a", "pcm_s16le",
                str(output_path),
            ])
        else:
            ffmpeg_cmd.extend([
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ])

        run(ffmpeg_cmd)
    finally:
        if downloaded_file.exists():
            downloaded_file.unlink()

    return output_path


def list_audio_input_devices():
    try:
        proc = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        output = proc.stderr
    except Exception:
        return []

    import re
    devices = []
    lines = output.splitlines()
    curr_name = None
    mic_idx = 1
    for line in lines:
        if "(audio)" in line:
            m = re.search(r'"([^"]+)"\s+\(audio\)', line)
            if m:
                curr_name = m.group(1)
        elif curr_name and "Alternative name" in line:
            m_alt = re.search(r'Alternative name\s+"([^"]+)"', line)
            if m_alt:
                alt = m_alt.group(1)
                label = f"{curr_name} — Entrée {mic_idx}"
                mic_idx += 1
                devices.append({
                    "name": curr_name,
                    "target": alt,
                    "label": label,
                })
                curr_name = None
    if not devices:
        devices.append({
            "name": "Microphone (High Definition Audio Device)",
            "target": "Microphone (High Definition Audio Device)",
            "label": "Microphone par défaut",
        })
    return devices


ACTIVE_RECORDERS = {}


def start_recording_hardware(session_key="default", duration_seconds=60, output_path=None, device_target=None):
    for k in list(ACTIVE_RECORDERS.keys()):
        stop_recording_hardware(k)

    if output_path is None:
        output_path = VOICES_DIR / f"mic_{uuid.uuid4().hex}.wav"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not device_target:
        devs = list_audio_input_devices()
        device_target = devs[0]["target"] if devs else "Microphone (High Definition Audio Device)"

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "dshow",
        "-i", f"audio={device_target}",
        "-t", str(int(duration_seconds)),
        "-ac", "1",
        "-ar", str(RATE),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,volume=3dB",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    ACTIVE_RECORDERS[session_key] = {"process": proc, "path": output_path}
    return output_path


def stop_recording_hardware(session_key="default"):
    rec = ACTIVE_RECORDERS.pop(session_key, None)
    if not rec:
        return None
    proc = rec["process"]
    path = rec["path"]
    if proc:
        try:
            proc.communicate(input=b"q", timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    return path


def is_recording_hardware(session_key="default"):
    rec = ACTIVE_RECORDERS.get(session_key)
    if not rec:
        return False
    proc = rec["process"]
    if proc.poll() is None:
        return True
    return False


def record_microphone_hardware(duration_seconds=10, output_path=None, device_target=None):
    if output_path is None:
        output_path = VOICES_DIR / f"mic_{uuid.uuid4().hex}.wav"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not device_target:
        devs = list_audio_input_devices()
        device_target = devs[0]["target"] if devs else "Microphone (High Definition Audio Device)"

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "dshow",
        "-i", f"audio={device_target}",
        "-t", str(int(duration_seconds)),
        "-ac", "1",
        "-ar", str(RATE),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,volume=3dB",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    run(cmd, timeout=int(duration_seconds) + 20)
    return output_path

