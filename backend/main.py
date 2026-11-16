import os
import sys
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import engine

app = FastAPI(title="Dubb-in Studio Mobile API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECTS_DIR = BASE_DIR / "projects"
VOICES_DIR = BASE_DIR / "voices"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR.parent / "frontend"

for d in [PROJECTS_DIR, VOICES_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Voix par défaut si disponible
default_ref = BASE_DIR.parent.parent / "VoxTranslate" / "voices" / "default.wav"
if not (VOICES_DIR / "default.wav").exists() and default_ref.exists():
    shutil.copyfile(default_ref, VOICES_DIR / "default.wav")
    (VOICES_DIR / "default.json").write_text(json.dumps({"id": "default", "name": "Ma voix de référence"}, indent=2), encoding="utf-8")

TASKS: Dict[str, Dict] = {}

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def get_root():
    if (STATIC_DIR / "index.html").exists():
        return FileResponse(
            str(STATIC_DIR / "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return {"app": "Dubb-in Mobile API", "status": "online"}

@app.get("/manifest.json")
def get_manifest():
    f = STATIC_DIR / "manifest.json"
    if f.exists():
        return FileResponse(str(f), media_type="application/manifest+json")
    raise HTTPException(status_code=404)

@app.get("/sw.js")
def get_sw():
    f = STATIC_DIR / "sw.js"
    if f.exists():
        return FileResponse(
            str(f),
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    raise HTTPException(status_code=404)

@app.get("/icon.png")
def get_icon():
    f = STATIC_DIR / "icon.png"
    if f.exists():
        return FileResponse(str(f), media_type="image/png")
    raise HTTPException(status_code=404)

@app.api_route("/Dubb-in.apk", methods=["GET", "HEAD"])
def get_apk():
    f = STATIC_DIR / "Dubb-in.apk"
    if f.exists():
        return FileResponse(
            str(f),
            media_type="application/vnd.android.package-archive",
            filename="Dubb-in_AI_Studio.apk"
        )
    raise HTTPException(status_code=404)

@app.get("/api/health")
def health():
    hw = engine.detect_hardware_profile()
    return {"status": "ok", "app": "Dubb-in Mobile API", "hardware": hw}

@app.get("/api/voices")
def list_voices():
    voices = []
    if (VOICES_DIR / "default.wav").exists():
        voices.append({"id": "default", "name": "Voix par défaut (Ma Voix)"})
    for j in VOICES_DIR.glob("*.json"):
        if j.stem == "default":
            continue
        try:
            info = json.loads(j.read_text(encoding="utf-8"))
            voices.append(info)
        except Exception:
            pass
    if not voices:
        voices.append({"id": "auto", "name": "Extrait vocal de la vidéo (Automatique)"})
    return {"voices": voices}

@app.post("/api/voices/register")
async def register_voice(name: str = Form(...), audio: UploadFile = File(...)):
    voice_id = str(uuid.uuid4())[:8]
    wav_path = VOICES_DIR / f"{voice_id}.wav"
    raw_path = VOICES_DIR / f"{voice_id}.raw"
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(raw_path),
        "-ar", "24000", "-ac", "1", str(wav_path)
    ])
    if raw_path.exists():
        try: raw_path.unlink()
        except Exception: pass

    info = {"id": voice_id, "name": name.strip() or f"Profil {voice_id}"}
    (VOICES_DIR / f"{voice_id}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"status": "ok", "voice": info}

@app.post("/api/voices/calibrate")
async def calibrate_voice(
    name: str = Form(...),
    sample1: UploadFile = File(...),
    sample2: UploadFile = File(...),
    sample3: UploadFile = File(...)
):
    voice_id = str(uuid.uuid4())[:8]
    temp_files = []
    converted_files = []

    for idx, sample in enumerate([sample1, sample2, sample3]):
        raw_p = VOICES_DIR / f"{voice_id}_raw_{idx}.raw"
        wav_p = VOICES_DIR / f"{voice_id}_{idx}.wav"
        with open(raw_p, "wb") as f:
            shutil.copyfileobj(sample.file, f)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(raw_p),
            "-ar", "24000", "-ac", "1", str(wav_p)
        ])
        temp_files.append(raw_p)
        converted_files.append(wav_p)

    for p in temp_files:
        try: p.unlink()
        except Exception: pass

    concat_list = VOICES_DIR / f"{voice_id}_list.txt"
    concat_list.write_text("\n".join(f"file '{p.name}'" for p in converted_files), encoding="utf-8")
    master_wav = VOICES_DIR / f"{voice_id}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(master_wav)
    ])
    try: concat_list.unlink()
    except Exception: pass

    info = {
        "id": voice_id,
        "name": f"⭐ {name.strip()} (Calibrage HD)",
        "calibrated": True,
        "samples": 3
    }
    (VOICES_DIR / f"{voice_id}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"status": "ok", "voice": info}

@app.post("/api/voices/register_url")
async def register_voice_url(
    name: str = Form(...),
    url: str = Form(...),
    start_time: float = Form(0.0),
    duration: float = Form(20.0)
):
    voice_id = str(uuid.uuid4())[:8]
    wav_path = VOICES_DIR / f"{voice_id}.wav"
    try:
        engine.download_media_from_url(
            url.strip(),
            wav_path,
            extract_audio=True,
            start_time=start_time,
            duration_seconds=duration
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Échec de l'extraction audio depuis le lien : {str(e)}")

    info = {"id": voice_id, "name": name.strip() or f"Profil {voice_id}"}
    (VOICES_DIR / f"{voice_id}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"status": "ok", "voice": info}

@app.get("/api/voices/{voice_id}/audio")
def get_voice_audio(voice_id: str):
    wav_path = VOICES_DIR / f"{voice_id}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Audio introuvable")
    return FileResponse(str(wav_path), media_type="audio/wav")

@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    if voice_id == "default":
        raise HTTPException(status_code=400, detail="Impossible de supprimer la voix par défaut")
    wav_path = VOICES_DIR / f"{voice_id}.wav"
    json_path = VOICES_DIR / f"{voice_id}.json"
    if wav_path.exists():
        try: wav_path.unlink()
        except Exception: pass
    if json_path.exists():
        try: json_path.unlink()
        except Exception: pass
    return {"status": "ok"}

def update_task(task_id: str, proj_dir: Path, data: dict):
    TASKS[task_id] = data
    try:
        (proj_dir / "status.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def run_render_task(task_id: str, proj_dir: Path, subtitle_style: str, keep_background: bool):
    try:
        doc_path = proj_dir / "document.json"
        if not doc_path.exists():
            raise FileNotFoundError("document.json introuvable")
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        num_segments = len(doc.get("segments", []))
        update_task(task_id, proj_dir, {"status": "processing", "step": f"4/4 Clonage vocal XTTS-v2 (0/{num_segments})...", "progress": 60})

        def render_progress(fraction, current=None, total_segs=None):
            if current is not None and total_segs is not None:
                pct = int(60 + fraction * 38)
                update_task(task_id, proj_dir, {
                    "status": "processing",
                    "step": f"4/4 Clonage vocal XTTS-v2 ({current}/{total_segs})...",
                    "progress": pct
                })
            else:
                pct = int(60 + fraction * 38)
                update_task(task_id, proj_dir, {
                    "status": "processing",
                    "step": "4/4 Clonage vocal XTTS-v2 & export vidéo...",
                    "progress": pct
                })

        engine.render(proj_dir, doc, progress=render_progress, keep_background=keep_background, subtitle_style=subtitle_style)

        update_task(task_id, proj_dir, {
            "status": "completed",
            "step": "✅ Doublage terminé avec succès !",
            "progress": 100,
            "download_url": f"/api/download/{task_id}"
        })
    except Exception as e:
        update_task(task_id, proj_dir, {
            "status": "error",
            "step": f"Erreur: {str(e)}",
            "progress": 0
        })

def run_dubbing_task(task_id: str, proj_dir: Path, target_lang: str, subtitle_style: str, keep_background: bool, review_subtitles: bool = False):
    try:
        update_task(task_id, proj_dir, {"status": "processing", "step": "1/4 Analyse et extraction audio...", "progress": 15})
        total = engine.prepare(proj_dir)

        update_task(task_id, proj_dir, {"status": "processing", "step": "2/4 Transcription Faster-Whisper des dialogues...", "progress": 35})
        doc = engine.transcribe(proj_dir)

        update_task(task_id, proj_dir, {"status": "processing", "step": f"3/4 Traduction neuronale en {target_lang}...", "progress": 55})
        doc = engine.translate(doc, target_lang)
        (proj_dir / "document.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        if review_subtitles:
            update_task(task_id, proj_dir, {
                "status": "awaiting_review",
                "step": "📝 Sous-titres prêts pour révision !",
                "progress": 55,
                "task_id": task_id
            })
            return

        run_render_task(task_id, proj_dir, subtitle_style, keep_background)
    except Exception as e:
        update_task(task_id, proj_dir, {
            "status": "error",
            "step": f"Erreur: {str(e)}",
            "progress": 0
        })

@app.post("/api/upload_chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk: UploadFile = File(...)
):
    upload_dir = UPLOADS_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = upload_dir / f"chunk_{chunk_index:05d}.part"
    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(chunk.file, f)

    parts = list(upload_dir.glob("chunk_*.part"))
    if len(parts) >= total_chunks:
        complete_video = upload_dir / "complete.mp4"
        with open(complete_video, "wb") as outfile:
            for i in range(total_chunks):
                p = upload_dir / f"chunk_{i:05d}.part"
                if p.exists():
                    with open(p, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
                    try: p.unlink()
                    except Exception: pass
        return {"status": "complete", "upload_id": upload_id}
    return {"status": "chunk_received", "received": len(parts), "total": total_chunks}

@app.post("/api/dubbing_from_upload")
async def start_dubbing_from_upload(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    voice_id: str = Form("default"),
    target_language: str = Form("en"),
    subtitle_style: str = Form("hormozi_yellow"),
    keep_background: bool = Form(False),
    review_subtitles: bool = Form(False),
    reference_audio: Optional[UploadFile] = File(None)
):
    upload_dir = UPLOADS_DIR / upload_id
    video_source = upload_dir / "complete.mp4"
    if not video_source.exists():
        raise HTTPException(status_code=400, detail="Fichier vidéo incomplet ou introuvable sur le serveur")

    task_id = str(uuid.uuid4())[:8]
    proj_dir = PROJECTS_DIR / task_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    video_input = proj_dir / "video.input"
    shutil.move(str(video_source), str(video_input))
    try: shutil.rmtree(upload_dir)
    except Exception: pass

    ref_input = proj_dir / "reference.input"
    if reference_audio and reference_audio.filename:
        with open(ref_input, "wb") as f:
            shutil.copyfileobj(reference_audio.file, f)
        if voice_id and voice_id != "default":
            v_wav = VOICES_DIR / f"{voice_id}.wav"
            try: shutil.copyfile(ref_input, v_wav)
            except Exception: pass
    else:
        voice_path = VOICES_DIR / f"{voice_id}.wav"
        if voice_path.exists():
            shutil.copyfile(voice_path, ref_input)
        elif (VOICES_DIR / "default.wav").exists():
            shutil.copyfile(VOICES_DIR / "default.wav", ref_input)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", str(video_input),
                "-t", "12.0", "-f", "wav", "-ar", "24000", "-ac", "1", str(ref_input)
            ])

    init_data = {"status": "queued", "step": "Vidéo reçue et validée. Lancement de l'IA...", "progress": 5}
    update_task(task_id, proj_dir, init_data)
    background_tasks.add_task(run_dubbing_task, task_id, proj_dir, target_language, subtitle_style, keep_background, review_subtitles)
    return {"status": "ok", "task_id": task_id}

@app.post("/api/dubbing")
async def start_dubbing(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    voice_id: str = Form("default"),
    target_language: str = Form("en"),
    subtitle_style: str = Form("hormozi_yellow"),
    keep_background: bool = Form(False),
    review_subtitles: bool = Form(False),
    reference_audio: Optional[UploadFile] = File(None)
):
    task_id = str(uuid.uuid4())[:8]
    proj_dir = PROJECTS_DIR / task_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    video_input = proj_dir / "video.input"
    with open(video_input, "wb") as f:
        shutil.copyfileobj(video.file, f)

    ref_input = proj_dir / "reference.input"
    if reference_audio and reference_audio.filename:
        with open(ref_input, "wb") as f:
            shutil.copyfileobj(reference_audio.file, f)
    else:
        voice_path = VOICES_DIR / f"{voice_id}.wav"
        if voice_path.exists():
            shutil.copyfile(voice_path, ref_input)
        elif (VOICES_DIR / "default.wav").exists():
            shutil.copyfile(VOICES_DIR / "default.wav", ref_input)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", str(video_input),
                "-t", "12.0", "-f", "wav", "-ar", "24000", "-ac", "1", str(ref_input)
            ])

    init_data = {"status": "queued", "step": "Projet initialisé, lancement...", "progress": 5}
    update_task(task_id, proj_dir, init_data)
    background_tasks.add_task(run_dubbing_task, task_id, proj_dir, target_language, subtitle_style, keep_background, review_subtitles)
    return {"status": "ok", "task_id": task_id}

@app.get("/api/project/{task_id}/subtitles")
def get_project_subtitles(task_id: str):
    proj_dir = PROJECTS_DIR / task_id
    doc_path = proj_dir / "document.json"
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Sous-titres non encore générés")
    try:
        data = json.loads(doc_path.read_text(encoding="utf-8"))
        for s in data.get("segments", []):
            src = s.get("source") or s.get("text") or ""
            tgt = s.get("target") or s.get("translation") or src
            s["source"] = src
            s["text"] = src
            s["target"] = tgt
            s["translation"] = tgt
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/{task_id}/resume_render")
async def resume_render(
    task_id: str,
    background_tasks: BackgroundTasks,
    subtitle_style: str = Form("hormozi_yellow"),
    keep_background: bool = Form(False),
    segments_json: Optional[str] = Form(None)
):
    proj_dir = PROJECTS_DIR / task_id
    doc_path = proj_dir / "document.json"
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Projet introuvable")

    if segments_json:
        try:
            updated_segs = json.loads(segments_json)
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            for s in updated_segs:
                val = s.get("target") or s.get("translation") or s.get("source") or ""
                s["target"] = val
                s["translation"] = val
            doc["segments"] = updated_segs
            doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Format invalide: {str(e)}")

    update_task(task_id, proj_dir, {"status": "processing", "step": "Lancement du clonage vocal et rendu final...", "progress": 58})
    background_tasks.add_task(run_render_task, task_id, proj_dir, subtitle_style, keep_background)
    return {"status": "ok", "task_id": task_id}

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    if task_id in TASKS:
        return TASKS[task_id]
    proj_dir = PROJECTS_DIR / task_id
    status_file = proj_dir / "status.json"
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            TASKS[task_id] = data
            return data
        except Exception:
            pass
    if (proj_dir / "export.mp4").exists():
        data = {
            "status": "completed",
            "step": "✅ Doublage terminé avec succès !",
            "progress": 100,
            "download_url": f"/api/download/{task_id}"
        }
        TASKS[task_id] = data
        return data
    raise HTTPException(status_code=404, detail="Tâche introuvable")

@app.get("/api/download/{task_id}")
def download_result(task_id: str):
    proj_dir = PROJECTS_DIR / task_id
    export_video = proj_dir / "export.mp4"
    if not export_video.exists():
        raise HTTPException(status_code=404, detail="Vidéo finale introuvable ou en cours de rendu")
    return FileResponse(
        str(export_video),
        media_type="video/mp4",
        filename=f"Dubb-in_{task_id}.mp4"
    )
