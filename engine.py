"""
Dubb-in AI Studio — High-Performance Audio/Video Dubbing Core Engine
Dynamic WSOLA Speech Alignment, Stem Demixing, and Voice Synthesis
https://github.com/papygaming1977/dubb-in
"""
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

import psutil
import io
import torch

# Correctif critique contre OSError: [WinError -1073741818] Windows Error 0xc0000006
# Force PyTorch à lire les fichiers .pth via un flux standard au lieu du Memory Mapping (mmap) Windows
if not hasattr(torch, "_orig_torch_load_safe_patched"):
    _orig_torch_load = torch.load

    def _safe_windows_torch_load(f, *args, **kwargs):
        kwargs.pop("mmap", None)  # Interdit le mmap qui fait planter le pagefile.sys
        if isinstance(f, (str, os.PathLike)):
            with open(f, "rb") as fp:
                return _orig_torch_load(fp, *args, **kwargs)
        return _orig_torch_load(f, *args, **kwargs)

    torch.load = _safe_windows_torch_load
    torch._orig_torch_load_safe_patched = True

# Détection dynamique du nombre de coeurs physiques réels (4 sur i5-3570, 8/12/16 sur gros PC)
_PHYSICAL_CORES = str(max(2, psutil.cpu_count(logical=False) or 4))
os.environ["OMP_NUM_THREADS"] = _PHYSICAL_CORES
os.environ["MKL_NUM_THREADS"] = _PHYSICAL_CORES
os.environ["OPENBLAS_NUM_THREADS"] = _PHYSICAL_CORES
os.environ["COQUI_TOS_AGREED"] = "1"
# Empêche PyTorch de saturer la mémoire virtuelle Windows lors du chargement
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

RATE = 24000

# Cache global singleton en mémoire pour éviter de recharger/requantifier XTTS-v2
_GLOBAL_XTTS_CACHE = {
    "model": None,
    "mode": None,
    "device": None,
    "dml_device": None
}

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


import functools


@functools.lru_cache(maxsize=1)
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

    # Détection directe via Windows Management si PyTorch est en build CPU
    if not gpu_name:
        try:
            import subprocess
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"],
                text=True, timeout=3
            ).strip()
            if out:
                gpu_data = json.loads(out)
                if isinstance(gpu_data, list):
                    dedicated = next((g for g in gpu_data if "nvidia" in g.get("Name", "").lower() or "geforce" in g.get("Name", "").lower()), gpu_data[0])
                    gpu_data = dedicated
                d_name = gpu_data.get("Name", "")
                d_ram = gpu_data.get("AdapterRAM", 0)
                if d_name:
                    gpu_name = d_name
                    vram_gb = round(d_ram / (1024**3), 1) if d_ram else 2.0
                    has_cuda = "nvidia" in gpu_name.lower() or "geforce" in gpu_name.lower()
        except Exception:
            pass

    g_label = gpu_name if gpu_name and gpu_name.lower().startswith("nvidia") else f"NVIDIA {gpu_name}"

    if has_cuda and vram_gb >= 6.0:
        tier = "Ultra (GPU Dédié)"
        whisper_model = "large-v3-turbo"
        demucs_capable = True
        xtts_device = "cuda"
        desc = f"{g_label} ({vram_gb} Go VRAM) — Accélération matérielle maximale"
    elif has_cuda and vram_gb >= 4.0:
        tier = "Performant (GPU 4 Go)"
        whisper_model = "small"
        demucs_capable = True
        xtts_device = "cuda"
        desc = f"{g_label} ({vram_gb} Go VRAM) — Mode optimisé GPU"
    elif has_cuda and vram_gb > 0:
        tier = "Hybride (GTX 960 Dédiée)"
        whisper_model = "small"
        demucs_capable = True
        xtts_device = "cpu"
        desc = f"{g_label} ({vram_gb} Go VRAM Maxwell) — Inférence Lip-Sync & Whisper"
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
    has_dml = False
    try:
        import torch_directml
        has_dml = torch_directml.is_available()
    except Exception:
        has_dml = False

    if has_cuda and vram_gb >= 1.8:
        accel_mode = "cuda_fp16_lowvram" if vram_gb <= 3.0 else "cuda_fp16_highvram"
    elif has_dml:
        accel_mode = "hybrid_dml_int8"
    else:
        accel_mode = "cpu_avx_int8"

    return {
        "tier": tier,
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "has_cuda": has_cuda,
        "has_dml": has_dml,
        "accel_mode": accel_mode,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "recommended_whisper": whisper_model,
        "xtts_device": xtts_device,
        "demucs_capable": demucs_capable,
        "description": desc,
    }


def get_optimized_xtts_model(hw_profile: dict = None):
    """
    Chargement sécurisé anti-erreur 0xc0000006 (Windows Page Error) pour i5 (AVX) + 12 Go DDR3 :
    - N'utilise PAS fbgemm (qui requiert AVX2 et sature le pagefile.sys Windows).
    - Active Intel MKL-DNN (OneDNN) optimisé nativement pour AVX (Ivy Bridge).
    - Décharge le Vocodeur HiFi-GAN sur la GTX 960 (DirectML ou CUDA) sans pic de RAM.
    """
    global _GLOBAL_XTTS_CACHE
    if _GLOBAL_XTTS_CACHE["model"] is not None:
        return (
            _GLOBAL_XTTS_CACHE["model"],
            _GLOBAL_XTTS_CACHE["mode"],
            _GLOBAL_XTTS_CACHE["dml_device"]
        )

    if hw_profile is None:
        hw_profile = detect_hardware_profile()

    # 1. Purge de la RAM avant d'allouer les 1.8 Go de XTTS-v2 (évite WinError 0xc0000006)
    import gc
    gc.collect()

    cpu_cores = hw_profile.get("cpu_count", 4)
    vram_gb = hw_profile.get("vram_gb", 2.0)

    try:
        import torch
        torch.set_num_threads(cpu_cores)
        torch.set_num_interop_threads(1)
        # Activation du moteur Intel OneDNN / MKL-DNN (100% compatible AVX1 i5-3470/3570)
        torch.backends.mkldnn.enabled = True
        torch.set_flush_denormals(True)  # Évite les ralentissements extrêmes sur les petits flottants CPU !
    except Exception:
        pass

    accel_mode = hw_profile.get("accel_mode", "cpu_avx_mkldnn")
    dml_device = None

    print(f"[DUBB-IN OPTIMIZER] Chargement sécurisé de XTTS-v2 en RAM...")
    from TTS.api import TTS
    # Chargement direct sur CPU sans mmap agressif
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    xtts = model.synthesizer.tts_model
    xtts.eval()

    # Désactive explicitement tous les gradients pour libérer la mémoire virtuelle Windows
    for param in xtts.parameters():
        param.requires_grad = False

    gc.collect()

    # --- MODE 1 : Accélération GPU NVIDIA (Auto-Scalable & Zéro conflit Float/Half) ---
    if hw_profile.get("has_cuda", False) and vram_gb >= 1.8:
        try:
            torch.cuda.empty_cache()
            compute_major = hw_profile.get("cuda_capability", (5, 2))[0]

            if vram_gb <= 4.5:
                print(f"[DUBB-IN GPU] Profil Low-VRAM ({vram_gb} Go) : Compression GPT FP16 + HiFi-GAN Float32...")
                if vram_gb <= 3.0:
                    torch.cuda.set_per_process_memory_fraction(0.85, 0)
                xtts.gpt.half()
                xtts.to("cuda")
                active_gpu_mode = "cuda_fp16"
            else:
                print(f"[DUBB-IN GPU] Profil Studio High-End ({vram_gb} Go) : Transfert direct GPU débridé...")
                if compute_major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                xtts.to("cuda")
                if vram_gb < 10.0:
                    xtts.gpt.half()
                    active_gpu_mode = "cuda_fp16"
                else:
                    active_gpu_mode = "cuda_tf32_full"

            if hasattr(model, "synthesizer"):
                model.synthesizer.tts_model = xtts
                if hasattr(model.synthesizer, "device"):
                    model.synthesizer.device = "cuda"

            allocated_mb = round(torch.cuda.memory_allocated(0) / (1024**2), 1)
            print(f"[DUBB-IN GPU] SUCCÈS ! XTTS-v2 actif sur {hw_profile.get('gpu_name', 'GPU')} ({allocated_mb} Mo VRAM | Mode: {active_gpu_mode})")
            _GLOBAL_XTTS_CACHE.update({"model": model, "mode": active_gpu_mode, "dml_device": None})
            return model, active_gpu_mode, None
        except Exception as e:
            print(f"[WARN GPU] Repli sur CPU suite à une erreur CUDA : {e}")
            xtts.to("cpu")
            xtts.float()

    # --- MODE 2 : Déchargement du Vocodeur HiFi-GAN sur la GTX 960 via DirectML ---
    if hw_profile.get("has_dml", False):
        try:
            import torch_directml
            dml_device = torch_directml.device()
            xtts.hifigan_decoder.to(dml_device)
            orig_hifigan_forward = xtts.hifigan_decoder.forward

            def dml_hifigan_forward(latents, g=None):
                lat_dml = latents.to(dml_device)
                g_dml = g.to(dml_device) if g is not None else None
                with torch.no_grad():
                    wav_dml = orig_hifigan_forward(lat_dml, g=g_dml)
                return wav_dml.cpu()

            xtts.hifigan_decoder.forward = dml_hifigan_forward
            print("[DUBB-IN OPTIMIZER] HiFi-GAN sur GTX 960 (DirectML) + GPT MKL-DNN AVX !")
            _GLOBAL_XTTS_CACHE.update({"model": model, "mode": "hybrid_dml_mkldnn", "dml_device": dml_device})
            return model, "hybrid_dml_mkldnn", dml_device
        except Exception as e:
            print(f"[WARN] DirectML ignoré ({e})")
            try:
                xtts.hifigan_decoder.to("cpu")
            except Exception:
                pass

    print("[DUBB-IN OPTIMIZER] Mode CPU MKL-DNN AVX + Flush-Denormals activé !")
    _GLOBAL_XTTS_CACHE.update({"model": model, "mode": "cpu_avx_mkldnn", "dml_device": None})
    return model, "cpu_avx_mkldnn", None


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
        try:
            start = float(row.get("start", 0.0))
            end = float(row.get("end", start + 0.5))
        except (ValueError, TypeError):
            start, end = 0.0, 1.0

        if not (math.isfinite(start) and math.isfinite(end)):
            start, end = previous_end, previous_end + 1.0

        # Auto-correction et recalage temporel
        start = max(0.0, start)
        if start < previous_end:
            start = previous_end

        # Si le segment dépasse légèrement la fin de la vidéo, recaler sur la fin
        if total and end > total:
            end = max(start + 0.05, total)

        if end <= start:
            end = start + 0.2

        row["start"] = round(start, 3)
        row["end"] = round(end, 3)

        text = row.get("target")
        if not text:
            row["target"] = str(row.get("source", "...")).strip()
        elif require_text and len(str(text)) > 1500:
            row["target"] = str(text)[:1500]

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


def generate_ass_subtitles(rows, output_path, style="hormozi_yellow", alignment="bottom", fontsize=None):
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

    if fontsize is not None:
        try:
            fs = int(fontsize)
            if fs > 0:
                font_size = int(fs * 1.5)
        except Exception:
            pass

    align_code = 2
    margin_v = 90
    al_str = str(alignment).lower()
    if any(k in al_str for k in ("top", "haut", "8", "6")):
        align_code = 8
        margin_v = 80
    elif any(k in al_str for k in ("mid", "cent", "milieu", "5", "10")):
        align_code = 5
        margin_v = 0
    else:
        align_code = 2
        margin_v = 90

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTokStyle,{font_name},{font_size},{primary_color},&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,{spacing},0,1,{outline_size},2,{align_code},40,40,{margin_v},1

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
    import ctranslate2
    project = Path(project)
    total_audio_duration = duration(project / "source.wav")
    hw = detect_hardware_profile()
    if not model_name:
        model_name = hw.get("recommended_whisper", "small")

    # Normalisation des alias
    if model_name in ("large-v3-turbo", "large_v3_turbo"):
        model_name = "large-v3"

    dev = "cuda" if hw["has_cuda"] else "cpu"
    vram_gb = hw.get("vram_gb", 2.0)
    # Sur GTX 960 (2 Go VRAM), les modèles 'large' dépassent la mémoire GPU -> bascule CPU
    if dev == "cuda" and vram_gb < 4.0 and any(m in model_name.lower() for m in ("large", "distil-large")):
        dev = "cpu"

    if dev == "cuda":
        try:
            import ctypes
            ctypes.CDLL("cublas64_12.dll")
        except Exception:
            dev = "cpu"

    def _run_transcribe(target_dev):
        try:
            supported = ctranslate2.get_supported_compute_types(target_dev)
        except Exception:
            supported = {"int8"} if target_dev == "cpu" else {"float32"}

        comp = "int8" if target_dev == "cpu" else "float32"
        for pref in (("int8", "float32") if target_dev == "cpu" else ("float16", "int8_float16", "int8_float32", "int8", "float32")):
            if pref in supported:
                comp = pref
                break

        cpu_threads = hw["cpu_count"] if target_dev == "cpu" else 4
        m = WhisperModel(
            model_name,
            device=target_dev,
            compute_type=comp,
            cpu_threads=cpu_threads,
        )
        it, info = m.transcribe(
            str(project / "source.wav"),
            vad_filter=True,
            beam_size=5,
        )
        res = []
        for segment in it:
            t = segment.text.strip()
            if not t:
                continue
            res.append({
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "source": t,
                "target": "",
            })
            if progress and total_audio_duration > 0:
                fraction = min(0.99, float(segment.end) / total_audio_duration)
                progress(fraction)
        detected_lang = getattr(info, "language", "fr") if info else "fr"
        return res, detected_lang

    try:
        rows, detected_lang = _run_transcribe(dev)
    except Exception:
        if dev != "cpu":
            rows, detected_lang = _run_transcribe("cpu")
        else:
            raise
    if not rows:
        raise ValueError(
            "Aucune parole détectée."
        )
    rows = merge_segments(rows)
    if progress:
        progress(1.0)
    document = {
        "source_language": detected_lang,
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


translate_document = translate


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


def stretch_audio_in_memory_wsola(audio_data, factor: float, sr: int = RATE):
    """
    WSOLA (Waveform Similarity Overlap-Add) pur NumPy en mémoire vive.
    - Préserve 100% le pitch (F0) et les formants de la voix clonée.
    - Exécution en ~0.6 ms sans usure disque ni sous-processus FFmpeg.
    """
    import numpy as np
    if abs(factor - 1.0) < 0.025 or len(audio_data) < 1024:
        return audio_data

    win_size = 720
    hop_syn = win_size // 2
    hop_ana = int(round(hop_syn * factor))
    search_tol = 120

    in_len = len(audio_data)
    out_len = int(round(in_len / factor))
    if out_len < win_size:
        return audio_data

    hann = np.hanning(win_size).astype(np.float32)
    out = np.zeros(out_len + win_size, dtype=np.float32)
    norm = np.zeros(out_len + win_size, dtype=np.float32)

    ana_pos = 0
    syn_pos = 0
    prev_natural_next = None

    while syn_pos + win_size < out_len and ana_pos + win_size + search_tol < in_len:
        best_offset = 0
        if prev_natural_next is not None and len(prev_natural_next) == hop_syn:
            search_start = max(0, ana_pos - search_tol)
            search_end = min(in_len - win_size, ana_pos + search_tol)
            if search_end > search_start:
                candidates = np.lib.stride_tricks.sliding_window_view(
                    audio_data[search_start:search_end + hop_syn], hop_syn
                )
                scores = np.dot(candidates, prev_natural_next)
                best_idx = int(np.argmax(scores))
                best_offset = (search_start + best_idx) - ana_pos

        actual_ana = max(0, min(in_len - win_size, ana_pos + best_offset))
        frame = audio_data[actual_ana:actual_ana + win_size]

        out[syn_pos:syn_pos + win_size] += frame * hann
        norm[syn_pos:syn_pos + win_size] += hann

        if actual_ana + hop_syn + hop_syn <= in_len:
            prev_natural_next = audio_data[actual_ana + hop_syn:actual_ana + win_size]
        else:
            prev_natural_next = None

        ana_pos += hop_ana
        syn_pos += hop_syn

    valid = norm > 1e-4
    out[valid] /= norm[valid]
    return out[:out_len]


def stretch_audio(audio_data, factor):
    """Point d'entrée compatible utilisant le moteur WSOLA RAM ultra-rapide."""
    return stretch_audio_in_memory_wsola(audio_data, factor, RATE)


CANCEL_PROJECTS = set()

def cancel_render(project):
    CANCEL_PROJECTS.add(str(Path(project).resolve()))

def is_render_cancelled(project):
    return str(Path(project).resolve()) in CANCEL_PROJECTS


def render(
    project,
    document=None,
    progress=None,
    keep_background=False,
    subtitle_style="none",
    vocal_speed=1.12,
    subtitles_style=None,
    speed=None,
    subtitle_alignment="bottom",
    subtitle_fontsize=None,
    progress_callback=None,
    **kwargs,
):
    import hashlib
    import numpy as np
    import soundfile as sf
    import torch
    from TTS.api import TTS

    project = Path(project)
    CANCEL_PROJECTS.discard(str(project.resolve()))
    if document is None:
        doc_file = project / "document.json"
        if doc_file.exists():
            document = json.loads(doc_file.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"Document introuvable dans le projet : {doc_file}")

    effective_sub_style = subtitles_style or subtitle_style or "none"
    effective_speed = speed if speed is not None else vocal_speed
    effective_progress = progress_callback or progress

    total = duration(
        project / "video.input"
    )
    rows = document["segments"]
    validate(rows, total)
    hw = detect_hardware_profile()
    model, active_mode, dml_device = get_optimized_xtts_model(hw)
    xtts = model.synthesizer.tts_model
    infer_device = "cuda" if active_mode.startswith("cuda") else "cpu"

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

    # Détection et pré-extraction UNIQUE de l'empreinte vocale neuronale (.dvf)
    fp_file = project / "fingerprint.dvf"
    gpt_cond_latent = None
    speaker_embedding = None

    if fp_file.exists():
        try:
            from voice_learning import VoiceFingerprintEngine
            gpt_cond_latent, speaker_embedding, _ = VoiceFingerprintEngine.load_fingerprint(str(fp_file), device=infer_device)
        except Exception:
            try:
                payload = torch.load(fp_file, map_location=infer_device)
                gpt_cond_latent = payload["gpt_cond_latent"].to(infer_device)
                speaker_embedding = payload["speaker_embedding"].to(infer_device)
            except Exception:
                pass

    if gpt_cond_latent is None or speaker_embedding is None:
        with torch.inference_mode():
            gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
                audio_path=[str(speaker_wav_file)]
            )
            gpt_cond_latent = gpt_cond_latent.to(infer_device)
            speaker_embedding = speaker_embedding.to(infer_device)
            try:
                torch.save({
                    "gpt_cond_latent": gpt_cond_latent.detach().cpu(),
                    "speaker_embedding": speaker_embedding.detach().cpu(),
                    "speaker_name": "reference"
                }, fp_file)
            except Exception:
                pass

    # Latents conservés en float32 (attendus par HiFi-GAN et gérés par autocast pour le GPT)
    gpt_cond_latent = gpt_cond_latent.float()
    speaker_embedding = speaker_embedding.float()

    vocal_speed = float(effective_speed)
    if vocal_speed < 0.5 or vocal_speed > 2.0:
        vocal_speed = 1.12
    last_end_sample = 0
    aligned_rows = []

    with torch.inference_mode():
        for index, row in enumerate(rows):
            if is_render_cancelled(project):
                raise InterruptedError("Synthèse interrompue par l'utilisateur.")
            text_clean = str(row.get("target", "")).strip()
            identity = json.dumps(
                [
                    text_clean,
                    target,
                    reference_hash,
                    "xtts-v2-opt",
                    f"speed_{vocal_speed:.2f}",
                    "neural_fp",
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
                use_autocast = (infer_device == "cuda" and active_mode == "cuda_fp16")
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16):
                    out = xtts.inference(
                        text=text_clean,
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
                audio = (
                    wav_tensor.detach().float().cpu().numpy()
                    if isinstance(wav_tensor, torch.Tensor)
                    else np.asarray(wav_tensor, dtype=np.float32)
                )
                sf.write(str(temporary_audio), audio, RATE)
                temporary_audio.replace(raw)
            else:
                audio, rate = sf.read(
                    raw,
                    dtype="float32",
                )

            # --- CONCORDANCE LABIALE HAUTE PRÉCISION (Alignement Paroles / Bouche) ---
            # 1. Élimination chirurgicale des silences d'attaque et d'extinction
            peak_val = float(np.max(np.abs(audio))) if len(audio) > 0 else 1.0
            thresh = max(0.003, 0.035 * peak_val)
            voiced_indices = np.where(np.abs(audio) > thresh)[0]
            if len(voiced_indices) > 0:
                pad_lead = int(0.012 * RATE)  # 12 ms pour préserver les consonnes d'attaque
                pad_trail = int(0.025 * RATE) # 25 ms pour l'extinction naturelle
                first_speech = max(0, int(voiced_indices[0]) - pad_lead)
                last_speech = min(len(audio), int(voiced_indices[-1]) + pad_trail)
                audio = audio[first_speech:last_speech]

            # 2. Fenêtre temporelle visuelle de l'interlocuteur
            seg_start = float(row["start"])
            seg_end = float(row["end"])
            target_dur = max(0.20, seg_end - seg_start)
            next_start = float(rows[index + 1]["start"]) if index + 1 < len(rows) else total

            # Marge de pause naturelle tolérée avant la phrase suivante (max 0.20s)
            slack_breath = min(0.20, max(0.0, next_start - seg_end - 0.04))
            max_allowed_dur = max(target_dur + slack_breath, 0.25)

            # 3. Ajustement dynamique bidirectionnel du tempo via WSOLA (0.85x à 1.35x)
            curr_dur = len(audio) / RATE
            if curr_dur > max_allowed_dur and max_allowed_dur > 0.25:
                # Phrase doublée trop longue : accélération fluide sans distorsion de pitch
                stretch_factor = min(1.35, max(1.0, curr_dur / max_allowed_dur))
                if stretch_factor > 1.025:
                    audio = stretch_audio_in_memory_wsola(audio, stretch_factor, RATE)
            elif curr_dur < target_dur * 0.85 and target_dur > 0.6:
                # Phrase doublée trop courte : ralentissement doux pour couvrir le mouvement des lèvres
                slow_factor = max(0.85, curr_dur / target_dur)
                if slow_factor < 0.96:
                    audio = stretch_audio_in_memory_wsola(audio, slow_factor, RATE)

            # Micro-fondus anti-clic de bordure
            if len(audio) > 80:
                fade_samples = min(48, len(audio) // 6)
                audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

            # 4. Ancrage strict sur l'ouverture des lèvres (ZÉRO dérive cumulative)
            actual_start_sample = round(seg_start * RATE)
            total_track_samples = len(track)

            # Si le segment précédent mordait très légèrement sur ce début, micro-crossfade propre
            if actual_start_sample < last_end_sample and actual_start_sample > 0:
                overlap = min(last_end_sample - actual_start_sample, int(0.05 * RATE))
                if overlap > 0:
                    track[actual_start_sample:actual_start_sample + overlap] *= np.linspace(1.0, 0.0, overlap, dtype=np.float32)

            # Borne de fin absolue avant l'ouverture de bouche suivante
            next_sample_bound = round((next_start - 0.03) * RATE) if index + 1 < len(rows) else total_track_samples
            write_end = min(total_track_samples, actual_start_sample + len(audio))
            if write_end > next_sample_bound and next_sample_bound > actual_start_sample + int(0.2 * RATE):
                clip_len = next_sample_bound - actual_start_sample
                audio = audio[:clip_len]
                f_tail = min(int(0.03 * RATE), len(audio) // 4)
                if f_tail > 0:
                    audio[-f_tail:] *= np.linspace(1.0, 0.0, f_tail, dtype=np.float32)
                write_end = actual_start_sample + len(audio)

            if actual_start_sample < total_track_samples:
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

            if effective_progress:
                try:
                    effective_progress((index + 1) / len(rows), index + 1, len(rows))
                except TypeError:
                    effective_progress((index + 1) / len(rows))
            if (index + 1) % 6 == 0:
                import gc
                gc.collect()
                if active_mode == "cuda_fp16":
                    torch.cuda.empty_cache()
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
    has_subtitles = effective_sub_style and effective_sub_style != "none"
    if has_subtitles:
        generate_ass_subtitles(
            aligned_rows,
            project / "subtitles_tiktok.ass",
            style=effective_sub_style,
            alignment=subtitle_alignment,
            fontsize=subtitle_fontsize,
        )

    final_audio_dur = last_end_sample / RATE
    final_total = total
    extra_pad = 0.0
    tpad_prefix = ""

    # Encodeur vidéo universel x264 (évite l'incompatibilité de version NVENC Driver < 610)
    video_encoder_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]

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
            *video_encoder_args,
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
                *video_encoder_args,
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
                    *video_encoder_args,
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

