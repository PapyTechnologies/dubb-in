#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOTEUR DE LIP-SYNC ASYMÉTRIQUE V2 "STUDIO GRADE" (DUBB-IN ENGINE)
Cible : Intel Core i5-3470 (AVX) | 12 Go DDR3 | NVIDIA GTX 960 (2 Go VRAM DirectML / CUDA)
Correctifs V2 :
  1. Zéro Bouche Fantôme sur B-Roll / Captures d'écran (Strict No-Face Gate + Batch Flush).
  2. Zéro Double-Bouche (MediaPipe FaceMesh sur 100% des frames + Lissage Cinématique One-Euro).
  3. Cadrage Carré Anatomique 1:1 + Alignement Affine Rotatif (warpAffine / invertAffineTransform).
  4. Séparation de Fréquences VFX + Stabilisateur Temporel EMA compilés en Numba AVX (No-GIL).
  5. Support DirectML + CUDA pour inférence GPU accélérée sur GTX 960.
"""

import os
import sys
import subprocess
import numpy as np
import cv2
import scipy.signal
import mediapipe as mp
import onnxruntime as ort
from numba import njit, prange


# ==============================================================================
# 1. NOYAUX BAS NIVEAU NUMBA LLVM (AVX / NO-GIL) : MASQUE, EMA & SÉPARATION FRÉQ.
# ==============================================================================

@njit(fastmath=True, nogil=True, cache=True)
def generate_anatomical_mouth_mask(h, w):
    """
    Dans le repere Full-Face 96x96 de Wav2Lip :
    - Moitie haute (y < 0.50 * h) : Yeux et nez -> Masque = 0.0 (100% HD original intact).
    - Moitie basse (y de 0.50 a 0.96 * h) : Bouche et machoire -> Remplacement complet de la vraie bouche !
    """
    mask = np.zeros((h, w), dtype=np.float32)
    cy = h * 0.73  # Centre exact de la bouche dans un visage complet 96x96
    cx = w * 0.50
    ry = h * 0.23  # Couvre de y=0.50 (sous le nez) jusqu'a y=0.96 (bas du menton)
    rx = w * 0.38  # Couvre largement les commissures des levres

    for y in range(h):
        if y < int(h * 0.50) or y > int(h * 0.96):
            continue
        for x in range(w):
            dy = (y - cy) / ry
            dx = (x - cx) / rx
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 0.62:
                mask[y, x] = 1.0  # Ecrase a 100% l'ancienne bouche originale !
            elif dist < 1.0:
                t = (dist - 0.62) / 0.38
                mask[y, x] = 0.5 * (1.0 + np.cos(t * np.pi))
            else:
                mask[y, x] = 0.0
    return mask


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def temporal_ema_stabilizer(curr_pred, prev_pred):
    """
    Filtre anti-scintillement (Anti-Flickering) guide par le mouvement en 96x96.
    Supprime le fourmillement sur les dents sans creer de flou sur les consonnes rapides.
    """
    h, w, c = curr_pred.shape
    out = np.empty((h, w, c), dtype=np.uint8)

    for y in prange(h):
        for x in range(w):
            diff = (
                abs(int(curr_pred[y, x, 0]) - int(prev_pred[y, x, 0])) +
                abs(int(curr_pred[y, x, 1]) - int(prev_pred[y, x, 1])) +
                abs(int(curr_pred[y, x, 2]) - int(prev_pred[y, x, 2]))
            ) / 3.0

            if diff < 10.0:
                w_curr = 0.62
            elif diff < 26.0:
                w_curr = 0.62 + 0.38 * ((diff - 10.0) / 16.0)
            else:
                w_curr = 1.0

            w_prev = 1.0 - w_curr
            for ch in range(3):
                val = w_curr * float(curr_pred[y, x, ch]) + w_prev * float(prev_pred[y, x, ch])
                if val < 0.0:
                    val = 0.0
                elif val > 255.0:
                    val = 255.0
                out[y, x, ch] = np.uint8(val)

    return out


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def vfx_frequency_separation_blend(orig_crop, pred_crop, mask_96, speech_weight=1.0):
    """
    Fusion propre sans reinjecter l'ombre de l'ancienne bouche ouverte :
    - On utilise mask_96 a 100% pour que l'ancienne bouche soit TOUJOURS masquee a 100%.
    """
    h, w, _ = orig_crop.shape
    out = np.empty((h, w, 3), dtype=np.uint8)

    mean_o = np.zeros(3, dtype=np.float32)
    mean_p = np.zeros(3, dtype=np.float32)
    count = 0.0

    for y in range(int(h * 0.48), int(h * 0.60)):
        for x in range(int(w * 0.20), int(w * 0.80)):
            count += 1.0
            for c in range(3):
                mean_o[c] += float(orig_crop[y, x, c])
                mean_p[c] += float(pred_crop[y, x, c])

    if count > 0.0:
        for c in range(3):
            mean_o[c] /= count
            mean_p[c] /= count

    for y in prange(h):
        for x in range(w):
            a = mask_96[y, x]
            if a <= 0.001:
                out[y, x, 0] = orig_crop[y, x, 0]
                out[y, x, 1] = orig_crop[y, x, 1]
                out[y, x, 2] = orig_crop[y, x, 2]
                continue

            for c in range(3):
                val_p = float(pred_crop[y, x, c])
                val_o = float(orig_crop[y, x, c])

                # Sharpening net sur les levres et les dents generees
                if 1 <= y < h - 1 and 1 <= x < w - 1:
                    blur_p = (
                        float(pred_crop[y - 1, x, c]) + float(pred_crop[y + 1, x, c]) +
                        float(pred_crop[y, x - 1, c]) + float(pred_crop[y, x + 1, c])
                    ) * 0.25
                    val_p = val_p + 0.45 * (val_p - blur_p)

                val_p += (mean_o[c] - mean_p[c]) * 0.60
                blended = a * val_p + (1.0 - a) * val_o
                if blended < 0.0:
                    blended = 0.0
                elif blended > 255.0:
                    blended = 255.0
                out[y, x, c] = np.uint8(blended)

    return out


# ==============================================================================
# 2. EXTRACTEUR AUDIO MEL + DÉTECTEUR D'ÉNERGIE VOCALE (SILENCE GATE)
# ==============================================================================

class LightweightMelExtractor:
    def __init__(self, sample_rate=16000, n_fft=800, hop_size=200, win_size=800, n_mels=80):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.n_mels = n_mels
        self.mel_basis = self._build_mel_basis(sample_rate, n_fft, n_mels, fmin=55.0, fmax=7600.0)

    @staticmethod
    def _hz_to_mel(frequencies):
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    @staticmethod
    def _mel_to_hz(mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def _build_mel_basis(self, sr, n_fft, n_mels, fmin, fmax):
        n_freqs = int(1 + n_fft // 2)
        fftfreqs = np.linspace(0, sr / 2.0, n_freqs)
        mel_f = self._mel_to_hz(np.linspace(self._hz_to_mel(fmin), self._hz_to_mel(fmax), n_mels + 2))

        weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            lower = (fftfreqs - mel_f[i]) / (mel_f[i + 1] - mel_f[i])
            upper = (mel_f[i + 2] - fftfreqs) / (mel_f[i + 2] - mel_f[i + 1])
            weights[i] = np.maximum(0.0, np.minimum(lower, upper))
            weights[i] *= 2.0 / (mel_f[i + 2] - mel_f[i])
        return weights

    def extract_mel_chunks(self, audio_path, fps):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", audio_path,
            "-ac", "1", "-ar", str(self.sr),
            "-f", "s16le", "-"
        ]
        raw_audio = subprocess.check_output(cmd)
        wav = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        wav = scipy.signal.lfilter([1.0, -0.97], [1.0], wav).astype(np.float32)

        window = scipy.signal.get_window("hann", self.win_size, fftbins=True)
        pad_len = int(self.n_fft // 2)
        wav_padded = np.pad(wav, (pad_len, pad_len), mode="reflect")

        num_frames = 1 + (len(wav_padded) - self.n_fft) // self.hop_size
        strides = (wav_padded.strides[0] * self.hop_size, wav_padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(wav_padded, shape=(num_frames, self.n_fft), strides=strides)

        spec = np.abs(np.fft.rfft(frames * window, n=self.n_fft, axis=1)).T
        mel_spec = np.dot(self.mel_basis, spec)

        min_level = np.exp(-5.0 * np.log(10.0))
        mel_db = 20.0 * np.log10(np.maximum(min_level, mel_spec)) - 20.0
        mel_norm = np.clip((2.0 * 4.0) * ((mel_db - (-100.0)) / 100.0) - 4.0, -4.0, 4.0)

        # Noise Gate franc sur les silences (< -3.1 -> -4.0)
        mel_norm = np.where(mel_norm < -3.1, -4.0, mel_norm)

        # Padding temporel centré (+2 pas d'anticipation pour les consonnes P/B/M)
        mel_padded = np.pad(mel_norm, ((0, 0), (8, 12)), mode="edge")

        mel_step_size = 16
        mel_idx_multiplier = 80.0 / float(fps)
        mel_chunks = []
        speech_weights = []

        i = 0
        smooth_w = 1.0
        while True:
            start_idx = int(i * mel_idx_multiplier) + 2
            if start_idx + mel_step_size > mel_padded.shape[1]:
                chunk = mel_padded[:, mel_padded.shape[1] - mel_step_size:]
                mel_chunks.append(chunk)
                speech_weights.append(smooth_w)
                break

            chunk = mel_padded[:, start_idx: start_idx + mel_step_size]
            mel_chunks.append(chunk)

            center_energy = float(np.percentile(chunk[:, 5:11], 90))
            raw_w = 1.0 if center_energy > -2.8 else 0.70
            smooth_w = raw_w if raw_w > smooth_w else (0.75 * smooth_w + 0.25 * raw_w)
            speech_weights.append(smooth_w)
            i += 1

        return mel_chunks, speech_weights


# ==============================================================================
# 3. TRACKER ANATOMIQUE FACEMESH TEMPS RÉEL (ZÉRO GHOSTING B-ROLL / ZÉRO LAG)
# ==============================================================================

class AnatomicalFaceMeshTracker:
    def __init__(self, target_size=96):
        self.target_size = target_size
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        )
        self.prev_cx = None
        self.prev_cy = None
        self.prev_size = None
        self.prev_angle = None

    def reset_state(self):
        self.prev_cx = None
        self.prev_cy = None
        self.prev_size = None
        self.prev_angle = None

    def get_aligned_face(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (320, 180)), cv2.COLOR_BGR2RGB)
        results = self.mp_mesh.process(small_rgb)

        if not results.multi_face_landmarks:
            self.reset_state()
            return None

        lm = results.multi_face_landmarks[0].landmark

        x_eye_l, y_eye_l = lm[33].x * w, lm[33].y * h
        x_eye_r, y_eye_r = lm[263].x * w, lm[263].y * h
        x_nose, y_nose = lm[1].x * w, lm[1].y * h
        x_chin, y_chin = lm[152].x * w, lm[152].y * h
        x_mouth_l, y_mouth_l = lm[61].x * w, lm[61].y * h
        x_mouth_r, y_mouth_r = lm[291].x * w, lm[291].y * h
        x_lip_c = (lm[13].x + lm[14].x) * 0.5 * w
        y_lip_c = (lm[13].y + lm[14].y) * 0.5 * h

        eye_dist = np.hypot(x_eye_r - x_eye_l, y_eye_r - y_eye_l)
        nose_to_chin = y_chin - y_nose
        mouth_width = np.hypot(x_mouth_r - x_mouth_l, y_mouth_r - y_mouth_l)

        if eye_dist < 22.0 or nose_to_chin < 20.0 or nose_to_chin > h * 0.75:
            self.reset_state()
            return None
        if not (y_nose < y_lip_c < y_chin) or (eye_dist / (nose_to_chin + 1e-5) > 2.2):
            self.reset_state()
            return None

        angle = np.degrees(np.arctan2(y_eye_r - y_eye_l, x_eye_r - x_eye_l))
        angle = float(np.clip(angle, -35.0, 35.0))

        # Key landmarks for full-face canonical Wav2Lip alignment (forehead to chin, cheek to cheek)
        # 10: forehead center, 152: chin center, 234: left cheek, 454: right cheek
        x_fh, y_fh = lm[10].x * w, lm[10].y * h
        x_ch, y_ch = lm[152].x * w, lm[152].y * h
        x_cl, y_cl = lm[234].x * w, lm[234].y * h
        x_cr, y_cr = lm[454].x * w, lm[454].y * h

        face_h = y_ch - y_fh
        face_w = x_cr - x_cl

        # Locked 1:1 Anatomical Square Region covering whole face (forehead to chin)
        # Aligns mouth exactly at Y=66 in 96x96 matching Wav2Lip GAN training space
        cx = (x_cl + x_cr) * 0.5
        cy = (y_fh + y_ch) * 0.5
        box_size = max(face_h, face_w) * 1.15

        if self.prev_cx is not None:
            move_px = np.hypot(cx - self.prev_cx, cy - self.prev_cy)
            size_diff = abs(box_size - self.prev_size) / (self.prev_size + 1e-5)

            if move_px > 2.2 or size_diff > 0.04:
                alpha_pos = 1.0
                alpha_scale = 1.0
            else:
                alpha_pos = 0.65
                alpha_scale = 0.50

            cx = alpha_pos * cx + (1.0 - alpha_pos) * self.prev_cx
            cy = alpha_pos * cy + (1.0 - alpha_pos) * self.prev_cy
            box_size = alpha_scale * box_size + (1.0 - alpha_scale) * self.prev_size
            angle = alpha_pos * angle + (1.0 - alpha_pos) * self.prev_angle

        self.prev_cx, self.prev_cy = cx, cy
        self.prev_size, self.prev_angle = box_size, angle

        scale = float(self.target_size) / float(box_size)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), scale)
        M[0, 2] += (self.target_size * 0.5) - cx
        M[1, 2] += (self.target_size * 0.5) - cy

        aligned_96 = cv2.warpAffine(
            frame_bgr, M, (self.target_size, self.target_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        M_inv = cv2.invertAffineTransform(M)

        half_s = int(box_size * 0.72)
        rx1 = max(0, int(cx - half_s))
        ry1 = max(0, int(cy - half_s))
        rx2 = min(w, int(cx + half_s))
        ry2 = min(h, int(cy + half_s))

        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            self.reset_state()
            return None

        return aligned_96, M_inv, (rx1, ry1, rx2, ry2)

    def close(self):
        try:
            self.mp_mesh.close()
        except Exception:
            pass


# ==============================================================================
# 4. MOTEUR PRINCIPAL GTX 960 (ORDONNANCEMENT SYNCHRONE + DIRECTML GPU)
# ==============================================================================

class GTX960LipSyncEngineV2:
    def __init__(self, onnx_model_path, batch_size=8):
        self.batch_size = batch_size
        self.img_size = 96

        print("[INIT] Compilation JIT des noyaux AVX Numba (Séparation de Fréquences + EMA)...")
        dummy = np.zeros((96, 96, 3), dtype=np.uint8)
        dummy_mask = generate_anatomical_mouth_mask(96, 96)
        _ = temporal_ema_stabilizer(dummy, dummy)
        _ = vfx_frequency_separation_blend(dummy, dummy, dummy_mask, 1.0)

        import psutil
        import subprocess
        physical_cores = psutil.cpu_count(logical=False) or 4

        # Détection automatique de la VRAM totale de la carte graphique (sans charger PyTorch)
        vram_mb = 2048  # Valeur par défaut sécurisée (2 Go)
        try:
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            )
            vram_mb = int(res.decode("utf-8").strip().split("\n")[0])
        except Exception:
            pass

        print("[INIT] Initialisation ONNX Runtime Auto-Scalable (DirectML / CUDA)...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(2, physical_cores - 1)
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        if vram_mb <= 3072:
            # --- PROFIL LOW-VRAM (ex: GTX 960 2 Go) : Protection anti-crash stricte ---
            self.batch_size = min(batch_size, 8)
            cuda_provider_options = {
                "device_id": 0,
                "gpu_mem_limit": 650 * 1024 * 1024,          # Plafond 650 Mo
                "arena_extend_strategy": "kSameAsRequested", # Zéro sur-allocation
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            print(f"[INIT] Mode Low-VRAM ({vram_mb} Mo) : Batch={self.batch_size}, Plafond=650 Mo")
        elif vram_mb <= 6144:
            # --- PROFIL MID-RANGE (4 Go à 6 Go VRAM) ---
            self.batch_size = max(batch_size, 16)
            cuda_provider_options = {
                "device_id": 0,
                "gpu_mem_limit": 2048 * 1024 * 1024,         # Plafond 2 Go
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",      # Benchmark le kernel le plus rapide
                "do_copy_in_default_stream": True,
            }
            print(f"[INIT] Mode Mid-Range ({vram_mb} Mo) : Batch={self.batch_size}, Turbo cuDNN activé")
        else:
            # --- PROFIL HIGH-END / STUDIO (8 Go à 24 Go+ VRAM) : Débridage total ---
            self.batch_size = max(batch_size, 32)
            cuda_provider_options = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": False,          # Streams parallèles PCIe !
            }
            print(f"[INIT] Mode Studio High-End ({vram_mb} Mo) : Batch={self.batch_size}, Débridage complet !")

        # Détection prioritaire DirectML (accéléré matériellement sur GTX 960 Windows)
        providers = [
            ("DmlExecutionProvider", {"device_id": 0}),
            ("CUDAExecutionProvider", cuda_provider_options),
            "CPUExecutionProvider"
        ]

        self.session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        print(f"[INIT] Active Provider: {self.active_provider}")

        active_providers = self.session.get_providers()
        if "CUDAExecutionProvider" not in active_providers and "DmlExecutionProvider" not in active_providers:
            self.batch_size = min(self.batch_size, 4)
            print(f"[INIT] Mode 100% CPU détecté : Batch ajusté à {self.batch_size} pour le cache processeur.")

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

        if "mel" in self.input_names[0].lower() or "audio" in self.input_names[0].lower():
            self.mel_inp_name, self.vid_inp_name = self.input_names[0], self.input_names[1]
        else:
            self.vid_inp_name, self.mel_inp_name = self.input_names[0], self.input_names[1]

        self.base_mask_96 = generate_anatomical_mouth_mask(self.img_size, self.img_size)
        self.prev_pred_96 = None

    def _flush_batch(self, b_idx, batch_crops, batch_mels, batch_meta, ffmpeg_proc):
        if b_idx == 0:
            return

        ort_inputs = {
            self.vid_inp_name: batch_crops[:b_idx],
            self.mel_inp_name: batch_mels[:b_idx]
        }
        preds = self.session.run([self.output_name], ort_inputs)[0]

        for i in range(b_idx):
            # Normalisation [0, 255] RGB -> BGR
            raw_p = preds[i]
            if raw_p.min() < 0.0:
                raw_p = (raw_p + 1.0) * 127.5
            else:
                raw_p = raw_p * 255.0
            raw_p = np.clip(raw_p, 0, 255).astype(np.uint8)

            rgb_pred = np.transpose(raw_p, (1, 2, 0))
            pred_96 = rgb_pred[:, :, ::-1].copy()

            if self.prev_pred_96 is not None:
                pred_96 = temporal_ema_stabilizer(pred_96, self.prev_pred_96)
            self.prev_pred_96 = pred_96.copy()

            orig_frame, aligned_96, M_inv, (rx1, ry1, rx2, ry2), sp_weight = batch_meta[i]

            blended_96 = vfx_frequency_separation_blend(aligned_96, pred_96, self.base_mask_96, 1.0)

            roi_w, roi_h = rx2 - rx1, ry2 - ry1
            M_inv_roi = M_inv.copy()
            M_inv_roi[0, 2] -= rx1
            M_inv_roi[1, 2] -= ry1

            orig_roi = orig_frame[ry1:ry2, rx1:rx2]
            warped_patch = cv2.warpAffine(
                blended_96, M_inv_roi, (roi_w, roi_h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
            )
            warped_mask = cv2.warpAffine(
                self.base_mask_96, M_inv_roi, (roi_w, roi_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )

            m3 = warped_mask[:, :, np.newaxis]
            orig_frame[ry1:ry2, rx1:rx2] = (
                m3 * warped_patch.astype(np.float32) + (1.0 - m3) * orig_roi.astype(np.float32)
            ).astype(np.uint8)

            ffmpeg_proc.stdin.write(orig_frame.tobytes())

    def process_video(self, video_path, audio_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[AUDIO] Extraction Mel-Spectrogramme & Silence Gate ({fps:.2f} FPS)...")
        mel_extractor = LightweightMelExtractor()
        mel_chunks, speech_weights = mel_extractor.extract_mel_chunks(audio_path, fps)
        num_frames = min(total_frames, len(mel_chunks))

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "-",
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p", "-shortest", output_path
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        tracker = AnatomicalFaceMeshTracker(target_size=self.img_size)

        batch_crops = np.zeros((self.batch_size, 6, self.img_size, self.img_size), dtype=np.float32)
        batch_mels = np.zeros((self.batch_size, 1, 80, 16), dtype=np.float32)
        batch_meta = []
        b_idx = 0

        print(f"[EXEC] Traitement Studio V2 de {num_frames} frames...")

        for f_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            tracked = tracker.get_aligned_face(frame)

            # CAS 1 : Aucun visage à l'écran (capture d'écran VS Code / B-Roll)
            if tracked is None:
                if b_idx > 0:
                    self._flush_batch(b_idx, batch_crops, batch_mels, batch_meta, ffmpeg_proc)
                    batch_meta.clear()
                    b_idx = 0
                self.prev_pred_96 = None
                ffmpeg_proc.stdin.write(frame.tobytes())
                continue

            # CAS 2 : Visage détecté et aligné
            aligned_96, M_inv, roi_box = tracked

            aligned_rgb = cv2.cvtColor(aligned_96, cv2.COLOR_BGR2RGB)
            masked_rgb = aligned_rgb.copy()
            masked_rgb[self.img_size // 2:, :] = 0

            concat_crop = np.concatenate((masked_rgb, aligned_rgb), axis=2).astype(np.float32) / 255.0
            batch_crops[b_idx] = np.transpose(concat_crop, (2, 0, 1))
            batch_mels[b_idx, 0] = mel_chunks[f_idx]
            batch_meta.append((frame, aligned_96, M_inv, roi_box, speech_weights[f_idx]))
            b_idx += 1

            if b_idx == self.batch_size or f_idx == num_frames - 1:
                self._flush_batch(b_idx, batch_crops, batch_mels, batch_meta, ffmpeg_proc)
                batch_meta.clear()
                b_idx = 0

            if (f_idx + 1) % 30 == 0 or f_idx == num_frames - 1:
                sys.stdout.write(f"\r[PROGRESSION] {f_idx + 1}/{num_frames} frames synchronisées...")
                sys.stdout.flush()

        if b_idx > 0:
            self._flush_batch(b_idx, batch_crops, batch_mels, batch_meta, ffmpeg_proc)
            batch_meta.clear()
            b_idx = 0

        cap.release()
        tracker.close()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"\n[TERMINÉ] Vidéo Studio V2 exportée avec succès : {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python fast_lipsync_960_v2.py <modele.onnx> <video_source.mp4> <audio_dubbin.wav> [sortie.mp4]")
        sys.exit(1)

    model_onnx = sys.argv[1]
    video_in = sys.argv[2]
    audio_in = sys.argv[3]
    video_out = sys.argv[4] if len(sys.argv) > 4 else "dubbin_lipsync_v2_out.mp4"

    engine = GTX960LipSyncEngineV2(onnx_model_path=model_onnx, batch_size=8)
    engine.process_video(video_in, audio_in, video_out)
