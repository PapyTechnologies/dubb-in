"""
Dubb-in LipSync — Master Asymmetric Pipeline V2
Target Hardware: Intel Core i5-3470 (AVX) | 12 GB DDR3 | NVIDIA GTX 960 (2 GB VRAM)
V2 Architecture Improvements:
  1. Strict No-Face Gate + FFmpeg Batch Flush: Zero ghost mouth on B-Roll/code editors.
  2. MediaPipe FaceMesh on 100% of frames: Zero double mouth via instant motion response.
  3. Locked 1:1 Anatomical Square & Roll Angle Affine Alignment (warpAffine).
  4. Numba AVX Frequency Separation VFX Blend & Temporal EMA Stabilizer in 96x96.
  5. Direct GPU execution via DirectML on NVIDIA GTX 960.
"""

import os
import sys
import subprocess
import time
import argparse
from typing import Optional, List
import cv2
import numpy as np

try:
    from .config import CONFIG, LipSyncConfig
    from .audio_mel import LightweightMelExtractor
    from .tracker import AnatomicalFaceMeshTracker, SparseFaceTracker
    from .blender import LipBlender, temporal_ema_stabilizer
    from .gpu_engine import Wav2LipGPUEngine
except ImportError:
    from config import CONFIG, LipSyncConfig
    from audio_mel import LightweightMelExtractor
    from tracker import AnatomicalFaceMeshTracker, SparseFaceTracker
    from blender import LipBlender, temporal_ema_stabilizer
    from gpu_engine import Wav2LipGPUEngine


class DubbInLipSyncPipeline:
    def __init__(self, model_path: str = CONFIG.model_path, batch_size: int = CONFIG.hw.batch_size):
        self.cfg = CONFIG
        self.batch_size = batch_size
        self.model_path = model_path
        self.mouth_size = self.cfg.mouth_size

        print("[INIT] Initializing Numba AVX blending kernels...")
        self.blender = LipBlender(self.cfg)

        print(f"[INIT] Initializing GPU Engine with model {model_path}...")
        self.gpu_engine = Wav2LipGPUEngine(model_path=model_path, batch_size=batch_size)
        self.mel_extractor = LightweightMelExtractor(self.cfg)
        self.prev_pred_96 = None

    def _flush_batch(self, b_idx: int, batch_crops: np.ndarray, batch_mels: np.ndarray,
                     batch_meta: List, ffmpeg_proc):
        """
        Executes GPU inference on accumulated micro-batch and pipes blended frames
        directly to FFmpeg in strict chronological order.
        """
        if b_idx == 0:
            return

        crops_sub = batch_crops[:b_idx]
        mels_sub = batch_mels[:b_idx]

        # Inférence GPU DirectML (GTX 960)
        preds = self.gpu_engine.run_micro_batch(crops_sub, mels_sub)

        for i in range(b_idx):
            pred_96 = preds[i]  # Format uint8 BGR (96, 96, 3)

            # 1. Stabilisation temporelle EMA guidée par le mouvement (anti-flickering dents)
            if self.prev_pred_96 is not None:
                pred_96 = temporal_ema_stabilizer(pred_96, self.prev_pred_96)
            self.prev_pred_96 = pred_96.copy()

            orig_frame, aligned_96, M_inv, roi_box, sp_weight = batch_meta[i]

            # 2. Fusion VFX (Couleur + Sharpening + Grain HD) et reprojection affine inverse
            blended_frame = self.blender.blend_affine(
                orig_frame=orig_frame,
                pred_96=pred_96,
                aligned_96=aligned_96,
                M_inv=M_inv,
                roi_box=roi_box,
                speech_weight=sp_weight
            )

            ffmpeg_proc.stdin.write(blended_frame.tobytes())

    def process(self, video_path: str, audio_path: str, output_path: str, dlf_cache_path: str = None, progress_callback=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"\n[VIDEO] Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

        # Step 1: Extraction Audio Mel-Spectrogramme & Voice Activity Detection (Silence Gate)
        print("[AUDIO] Computing 80-bin Mel spectrogram & speech weights...")
        mel_chunks, speech_weights = self.mel_extractor.extract_from_file(audio_path, fps, return_weights=True)
        num_frames = min(total_frames, len(mel_chunks))
        print(f"[AUDIO] {len(mel_chunks)} audio mel frames computed for {num_frames} video frames.")

        # Step 2: FFmpeg Direct Rawvideo Pipe
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

        # Step 3: Tracker Anatomique MediaPipe FaceMesh V2 (100% des frames)
        tracker = AnatomicalFaceMeshTracker(target_size=self.mouth_size)

        batch_crops = np.zeros((self.batch_size, 6, self.mouth_size, self.mouth_size), dtype=np.float32)
        batch_mels = np.zeros((self.batch_size, 1, 80, 16), dtype=np.float32)
        batch_meta = []
        b_idx = 0
        self.prev_pred_96 = None

        cap = cv2.VideoCapture(video_path)
        start_time = time.time()

        print(f"[SYNC] Processing {num_frames} frames with Strict No-Face Gate & DirectML GPU...")

        for f_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            tracked = tracker.get_aligned_face(frame)

            # CAS 1 : Aucun visage detecte (capture d'ecran VS Code / B-Roll / Plan de coupe)
            if tracked is None:
                # Flush immediat des frames accumulees pour preserver l'ordre chronologique
                if b_idx > 0:
                    self._flush_batch(b_idx, batch_crops, batch_mels, batch_meta, ffmpeg_proc)
                    batch_meta.clear()
                    b_idx = 0
                # Purge de la memoire temporelle de bouche
                self.prev_pred_96 = None
                # Ecriture de la frame originale intacte (ZERO bouche fantome)
                ffmpeg_proc.stdin.write(frame.tobytes())

                # Telemetry
                if (f_idx + 1) % 15 == 0 or f_idx == num_frames - 1:
                    elapsed = time.time() - start_time
                    fps_proc = (f_idx + 1) / max(0.001, elapsed)
                    eta = (num_frames - (f_idx + 1)) / max(0.001, fps_proc)
                    if progress_callback:
                        try:
                            progress_callback("sync", f_idx + 1, num_frames, fps_proc, eta)
                        except TypeError:
                            try:
                                progress_callback(f_idx + 1, num_frames, fps_proc, eta)
                            except Exception:
                                pass
                continue

            # CAS 2 : Visage detecte et aligne anatomiquement (Ratio 1:1 + Roll warpAffine)
            aligned_96, M_inv, roi_box = tracked

            # Conversion en RGB pour le reseau Wav2Lip
            aligned_96_rgb = cv2.cvtColor(aligned_96, cv2.COLOR_BGR2RGB)
            masked_96_rgb = aligned_96_rgb.copy()
            masked_96_rgb[self.mouth_size // 2:, :] = 0

            concat_crop = np.concatenate((masked_96_rgb, aligned_96_rgb), axis=2).astype(np.float32) / 255.0
            batch_crops[b_idx] = np.transpose(concat_crop, (2, 0, 1))
            batch_mels[b_idx, 0] = mel_chunks[f_idx]
            batch_meta.append((frame, aligned_96, M_inv, roi_box, speech_weights[f_idx]))
            b_idx += 1

            if b_idx == self.batch_size or f_idx == num_frames - 1:
                self._flush_batch(b_idx, batch_crops, batch_mels, batch_meta, ffmpeg_proc)
                batch_meta.clear()
                b_idx = 0

            # Telemetrie de progression
            if (f_idx + 1) % 15 == 0 or f_idx == num_frames - 1:
                elapsed = time.time() - start_time
                fps_proc = (f_idx + 1) / max(0.001, elapsed)
                eta = (num_frames - (f_idx + 1)) / max(0.001, fps_proc)
                sys.stdout.write(f"\r[PROGRESS] {f_idx + 1}/{num_frames} frames ({fps_proc:.1f} fps) | Elapsed: {elapsed:.1f}s")
                sys.stdout.flush()
                if progress_callback:
                    try:
                        progress_callback("sync", f_idx + 1, num_frames, fps_proc, eta)
                    except TypeError:
                        try:
                            progress_callback(f_idx + 1, num_frames, fps_proc, eta)
                        except Exception:
                            pass

        # Vidange finale si necessaire
        if b_idx > 0:
            self._flush_batch(b_idx, batch_crops, batch_mels, batch_meta, ffmpeg_proc)
            batch_meta.clear()
            b_idx = 0

        cap.release()
        tracker.close()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

        total_elapsed = time.time() - start_time
        print(f"\n[DONE] Lip-sync V2 complete in {total_elapsed:.2f}s. Saved to: {output_path}")
        return output_path


def run_lipsync_process(
    video_path: str,
    audio_path: str,
    output_path: str,
    model_path: str = None,
    batch_size: int = None,
    dlf_cache_path: str = None,
    progress_callback = None,
    **kwargs,
):
    """Entry point for executing lip-sync from UI or external scripts."""
    m_path = model_path or CONFIG.model_path
    b_size = batch_size or CONFIG.hw.batch_size
    pipe = DubbInLipSyncPipeline(model_path=m_path, batch_size=b_size)
    return pipe.process(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        dlf_cache_path=dlf_cache_path,
        progress_callback=progress_callback,
    )


def main():
    parser = argparse.ArgumentParser(description="Dubb-in LipSync Studio V2 — Asymmetric Engine")
    parser.add_argument("--video", "-v", required=True, help="Path to input source video (MP4/MKV)")
    parser.add_argument("--audio", "-a", required=True, help="Path to target dubbed audio (WAV/MP3)")
    parser.add_argument("--output", "-o", default="output_lipsync_v2.mp4", help="Path to output synchronized video")
    parser.add_argument("--model", "-m", default=CONFIG.model_path, help="Path to wav2lip_gan.onnx model")
    parser.add_argument("--batch", "-b", type=int, default=CONFIG.hw.batch_size, help="Micro-batch size (1 to 8)")

    args = parser.parse_args()

    pipeline = DubbInLipSyncPipeline(model_path=args.model, batch_size=args.batch)
    pipeline.process(
        video_path=args.video,
        audio_path=args.audio,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
