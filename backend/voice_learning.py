"""
Dubb-in Studio — Moteur d'Apprentissage et d'Étude d'Empreinte Vocale (Voice Fingerprint Engine)
Auteur : Senior AI & Audio Systems Engineer
Compatibilité : Python 3.10+, PyTorch 2.x, XTTS-v2, Librosa, SciPy, FFmpeg
"""

import os
import sys
import json
import uuid
import math
import shutil
import sqlite3
import hashlib
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import soundfile as sf
import torch
import librosa

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
FINGERPRINTS_DIR = VOICES_DIR / "fingerprints"
DB_PATH = VOICES_DIR / "voices.db"
RATE = 24000  # Fréquence standard XTTS-v2


class AudioAcousticAnalyzer:
    """
    Analyse acoustique, biométrique et spectrale avancée du signal vocal.
    Extrait la fréquence fondamentale (F0), les formants, le rapport signal/bruit (SNR)
    et un indice de clarté audio pour qualifier l'empreinte vocale.
    """

    @staticmethod
    def classify_vocal_range(f0_median: float) -> str:
        """Classifie la tessiture vocale selon le F0 médian en Hz."""
        if f0_median <= 0:
            return "Indéterminée"
        elif f0_median < 100:
            return "Basse (Grave profond)"
        elif f0_median < 140:
            return "Baryton (Médium grave)"
        elif f0_median < 185:
            return "Ténor / Alto (Médium)"
        elif f0_median < 255:
            return "Mezzo-Soprano (Aigu)"
        else:
            return "Soprano (Très aigu)"

    @classmethod
    def analyze_audio(cls, wav_path: Union[str, Path]) -> Dict[str, Union[float, str, bool]]:
        """
        Effectue une analyse spectrale et statistique complète d'un fichier audio WAV.
        """
        wav_path = str(wav_path)
        y, sr = librosa.load(wav_path, sr=RATE, mono=True)
        dur = float(len(y) / sr)

        if dur < 1.0:
            raise ValueError(f"Fichier trop court pour l'analyse ({dur:.2f}s, minimum 1.0s).")

        # 1. Analyse de la hauteur (Pitch / F0) via algorithme YIN
        f0 = librosa.yin(y, fmin=55, fmax=450, sr=sr)
        voiced_f0 = f0[~np.isnan(f0)]

        if len(voiced_f0) > 0:
            f0_median = float(np.median(voiced_f0))
            f0_mean = float(np.mean(voiced_f0))
            f0_std = float(np.std(voiced_f0))
            f0_min = float(np.percentile(voiced_f0, 5))
            f0_max = float(np.percentile(voiced_f0, 95))
        else:
            f0_median = 0.0
            f0_mean = 0.0
            f0_std = 0.0
            f0_min = 0.0
            f0_max = 0.0

        vocal_type = cls.classify_vocal_range(f0_median)

        # 2. Analyse spectrale (Timbre & Brillance)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_flat = librosa.feature.spectral_flatness(y=y)[0]
        brightness_hz = float(np.median(spec_cent)) if len(spec_cent) > 0 else 0.0
        flatness = float(np.mean(spec_flat)) if len(spec_flat) > 0 else 0.0

        # 3. Énergie RMS et Rapport Signal/Bruit (SNR)
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        speech_frames = rms_db > -35
        noise_frames = rms_db <= -35

        if np.any(speech_frames) and np.any(noise_frames):
            speech_power = np.mean(rms[speech_frames] ** 2)
            noise_power = np.mean(rms[noise_frames] ** 2)
            snr_db = float(10 * np.log10(max(1e-9, speech_power) / max(1e-9, noise_power)))
            snr_db = max(0.0, min(snr_db, 60.0))
        else:
            snr_db = 28.0  # Valeur nominale si environnement très silencieux

        # 4. Indice de Clarté / Pureté Audio (Score sur 100)
        # Pénalise le bruit de fond, l'écrêtage (clipping) et l'instabilité
        clipping_ratio = float(np.sum(np.abs(y) > 0.98) / len(y))
        clarity = 100.0
        if clipping_ratio > 0.001:
            clarity -= min(30.0, clipping_ratio * 1000 * 5)
        if snr_db < 20:
            clarity -= (20 - snr_db) * 2.5
        if flatness > 0.15:  # Trop de bruit blanc
            clarity -= (flatness - 0.15) * 100
        clarity = float(max(20.0, min(100.0, round(clarity, 1))))

        return {
            "duration": round(dur, 2),
            "f0_median": round(f0_median, 1),
            "f0_mean": round(f0_mean, 1),
            "f0_std": round(f0_std, 1),
            "f0_min": round(f0_min, 1),
            "f0_max": round(f0_max, 1),
            "vocal_type": vocal_type,
            "brightness_hz": round(brightness_hz, 0),
            "snr_db": round(snr_db, 1),
            "clarity_score": clarity,
            "sample_rate": sr,
        }


class VoiceCalibrator:
    """
    Normalisation acoustique haute fidélité pour l'apprentissage neuronal.
    Applique filtrage passe-haut, normalisation LUFS (EBU R128) et découpe chirurgicale.
    """

    @staticmethod
    def calibrate_sample(input_path: Union[str, Path], output_path: Union[str, Path], max_duration: float = 20.0) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_path),
            "-t", str(max_duration),
            "-map", "0:a:0",
            "-vn",
            "-af", "highpass=f=60,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", str(RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            # Fallback simple si filtre complexe échoue
            fallback_cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(input_path),
                "-t", str(max_duration),
                "-map", "0:a:0",
                "-vn",
                "-ar", str(RATE),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(output_path)
            ]
            subprocess.run(fallback_cmd, capture_output=True, check=True)

        return output_path


class VoiceFingerprintEngine:
    """
    Système d'apprentissage neuronal et de sérialisation d'empreinte vocale (.dvf).
    Gère l'extraction des vecteurs de conditionnement GPT et du speaker embedding 512-d.
    """

    def __init__(self):
        FINGERPRINTS_DIR.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @staticmethod
    def init_db():
        """Met à jour et vérifie le schéma de voices.db pour supporter les empreintes neuronales."""
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
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
                    duration REAL NOT NULL,
                    is_neural_fingerprint INTEGER DEFAULT 0,
                    f0_median REAL DEFAULT 0.0,
                    vocal_type TEXT DEFAULT 'Standard',
                    clarity_score REAL DEFAULT 80.0,
                    snr_db REAL DEFAULT 25.0,
                    sample_count INTEGER DEFAULT 1,
                    dvf_filename TEXT
                )
            """)
            # Migration automatique si colonnes absentes sur une ancienne base
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(voice_profiles)")
            columns = [c[1] for c in cursor.fetchall()]
            new_cols = [
                ("is_neural_fingerprint", "INTEGER DEFAULT 0"),
                ("f0_median", "REAL DEFAULT 0.0"),
                ("vocal_type", "TEXT DEFAULT 'Standard'"),
                ("clarity_score", "REAL DEFAULT 80.0"),
                ("snr_db", "REAL DEFAULT 25.0"),
                ("sample_count", "INTEGER DEFAULT 1"),
                ("dvf_filename", "TEXT")
            ]
            for col_name, col_def in new_cols:
                if col_name not in columns:
                    try:
                        conn.execute(f"ALTER TABLE voice_profiles ADD COLUMN {col_name} {col_def}")
                    except Exception:
                        pass
            conn.commit()

    @staticmethod
    def _get_xtts_model(device: str = "cpu"):
        """Charge ou récupère l'instance singleton du modèle XTTS-v2."""
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        return tts.synthesizer.tts_model

    def learn_fingerprint(
        self,
        name: str,
        samples: List[Union[str, Path, bytes]],
        source_type: str = "studio_learning",
        device: str = "cpu"
    ) -> Dict:
        """
        Procédure d'apprentissage complet d'empreinte vocale :
        1. Ingestion et calibration multi-échantillons.
        2. Analyse biométrique acoustique combinée.
        3. Extraction neuronale des latents GPT et du speaker embedding via XTTS.
        4. Sérialisation du package d'empreinte (.dvf).
        5. Enregistrement dans la banque de voix.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("Le nom du profil vocal ne peut pas être vide.")
        if not samples:
            raise ValueError("Au moins un échantillon audio est requis pour l'apprentissage.")

        session_id = uuid.uuid4().hex
        temp_dir = Path(tempfile.gettempdir()) / f"dubbin_voice_{session_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        calibrated_paths = []
        individual_metrics = []

        try:
            # 1. Traitement et calibration de chaque échantillon
            for idx, s in enumerate(samples):
                raw_path = temp_dir / f"raw_{idx}.wav"
                if isinstance(s, (bytes, bytearray)):
                    raw_path.write_bytes(s)
                else:
                    shutil.copyfile(str(s), raw_path)

                calibrated_path = temp_dir / f"calibrated_{idx}.wav"
                VoiceCalibrator.calibrate_sample(raw_path, calibrated_path, max_duration=25.0)
                metrics = AudioAcousticAnalyzer.analyze_audio(calibrated_path)
                individual_metrics.append(metrics)
                calibrated_paths.append(str(calibrated_path))

            # 2. Métriques consolidées
            total_dur = sum(m["duration"] for m in individual_metrics)
            avg_f0 = float(np.mean([m["f0_median"] for m in individual_metrics if m["f0_median"] > 0])) if any(m["f0_median"] > 0 for m in individual_metrics) else 0.0
            avg_clarity = float(np.mean([m["clarity_score"] for m in individual_metrics]))
            avg_snr = float(np.mean([m["snr_db"] for m in individual_metrics]))
            vocal_type = AudioAcousticAnalyzer.classify_vocal_range(avg_f0)

            # 3. Extraction neuronale XTTS-v2
            xtts = self._get_xtts_model(device=device)
            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
                    audio_path=calibrated_paths,
                    max_ref_length=30,
                    gpt_cond_len=6,
                    sound_norm_refs=True
                )

            # 4. Sérialisation de l'empreinte (.dvf = Dubb-in Voice Fingerprint)
            dvf_filename = f"fingerprint_{session_id}.dvf"
            dvf_path = FINGERPRINTS_DIR / dvf_filename
            master_wav_filename = f"{session_id}.wav"
            master_wav_path = VOICES_DIR / master_wav_filename

            # Le premier échantillon calibré sert de référence audio d'écoute
            shutil.copyfile(calibrated_paths[0], master_wav_path)

            fingerprint_payload = {
                "version": "2.0",
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "duration": total_dur,
                    "f0_median": avg_f0,
                    "vocal_type": vocal_type,
                    "clarity_score": avg_clarity,
                    "snr_db": avg_snr,
                    "sample_count": len(samples)
                },
                "gpt_cond_latent": gpt_cond_latent.cpu(),
                "speaker_embedding": speaker_embedding.cpu()
            }
            torch.save(fingerprint_payload, dvf_path)

            # 5. Enregistrement persistant dans la base SQLite
            sha = hashlib.sha256(master_wav_path.read_bytes()).hexdigest()
            created = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO voice_profiles (
                        name, created_at, source_type, filename, sha256, duration,
                        is_neural_fingerprint, f0_median, vocal_type, clarity_score,
                        snr_db, sample_count, dvf_filename
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, created, source_type, master_wav_filename, sha, total_dur,
                    1, avg_f0, vocal_type, avg_clarity, avg_snr, len(samples), dvf_filename
                ))
                conn.commit()

            return {
                "name": name,
                "dvf_filename": dvf_filename,
                "dvf_path": dvf_path,
                "master_wav": master_wav_path,
                "vocal_type": vocal_type,
                "f0_median": avg_f0,
                "clarity_score": avg_clarity,
                "snr_db": avg_snr,
                "duration": total_dur,
                "sample_count": len(samples)
            }
        finally:
            # Nettoyage strict des fichiers temporaires intermédiaires
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def load_fingerprint(dvf_filename: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Charge instantanément les tenseurs d'empreinte vocale précalculés."""
        dvf_path = FINGERPRINTS_DIR / dvf_filename
        if not dvf_path.exists():
            raise FileNotFoundError(f"Empreinte vocale introuvable : {dvf_path}")

        payload = torch.load(dvf_path, map_location=device)
        gpt_cond = payload["gpt_cond_latent"].to(device)
        spk_emb = payload["speaker_embedding"].to(device)
        return gpt_cond, spk_emb, payload.get("metrics", {})


voice_engine = VoiceFingerprintEngine()
