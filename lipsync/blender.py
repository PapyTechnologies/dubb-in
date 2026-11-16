"""
Dubb-in LipSync — High-Speed Blending & Frequency Separation Kernel
Hardware: Intel Core i5-3470 (AVX) & Multi-core CPU
Dual-Engine: Uses Numba AVX JIT with No-GIL execution, and pure NumPy fallback.
Features:
  - Full-Face 96x96 anatomical mouth mask covering y=0.50*h to y=0.96*h.
  - Motion-guided temporal EMA anti-flickering stabilizer in 96x96.
  - VFX Frequency Separation (local skin tone transfer + unsharp sharpening).
  - Inverse Affine Warping directly onto original HD frame without transparency leakage.
"""

import numpy as np
import cv2

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG

# Attempt Numba import
HAVE_NUMBA = False
try:
    from numba import njit, prange

    @njit(fastmath=True, nogil=True, cache=True)
    def _numba_anatomical_mask(h: int, w: int) -> np.ndarray:
        """
        Dans le repere Full-Face 96x96 de Wav2Lip :
        - Moitie haute (y < 0.50 * h) : Yeux et nez -> Masque = 0.0 (100% HD original intact).
        - Moitie basse (y de 0.50 a 0.96 * h) : Bouche et machoire -> Remplacement complet de l'ancienne bouche !
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
                    mask[y, x] = 1.0  # Ecrase a 100% l'ancienne bouche originale
                elif dist < 1.0:
                    t = (dist - 0.62) / 0.38
                    mask[y, x] = 0.5 * (1.0 + np.cos(t * np.pi))
                else:
                    mask[y, x] = 0.0
        return mask

    @njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def _numba_temporal_ema(curr_pred: np.ndarray, prev_pred: np.ndarray) -> np.ndarray:
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
    def _numba_vfx_freq_blend(orig_crop: np.ndarray, pred_crop: np.ndarray, mask_96: np.ndarray, speech_weight: float = 1.0) -> np.ndarray:
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

    # Test compilation warm-up
    _test_mask = _numba_anatomical_mask(96, 96)
    _test_roi = np.zeros((96, 96, 3), dtype=np.uint8)
    _ = _numba_temporal_ema(_test_roi, _test_roi)
    _ = _numba_vfx_freq_blend(_test_roi, _test_roi, _test_mask, 1.0)
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


def _numpy_anatomical_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cy = h * 0.73
    cx = w * 0.50
    ry = h * 0.23
    rx = w * 0.38

    Y, X = np.ogrid[:h, :w]
    dy = (Y - cy) / ry
    dx = (X - cx) / rx
    dist = np.sqrt(dx * dx + dy * dy)

    valid_y = (Y >= int(h * 0.50)) & (Y <= int(h * 0.96))
    inner = (dist < 0.62) & valid_y
    mask[inner] = 1.0

    trans = (dist >= 0.62) & (dist < 1.0) & valid_y
    t = (dist[trans] - 0.62) / 0.38
    mask[trans] = 0.5 * (1.0 + np.cos(t * np.pi))
    return mask


def _numpy_temporal_ema(curr_pred: np.ndarray, prev_pred: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(curr_pred.astype(np.float32) - prev_pred.astype(np.float32)), axis=2, keepdims=True)
    w_curr = np.where(diff < 10.0, 0.62, np.where(diff < 26.0, 0.62 + 0.38 * ((diff - 10.0) / 16.0), 1.0))
    blended = w_curr * curr_pred.astype(np.float32) + (1.0 - w_curr) * prev_pred.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _numpy_vfx_freq_blend(orig_crop: np.ndarray, pred_crop: np.ndarray, mask_96: np.ndarray, speech_weight: float = 1.0) -> np.ndarray:
    h, w, _ = orig_crop.shape
    y0, y1 = int(h * 0.48), int(h * 0.60)
    x0, x1 = int(w * 0.20), int(w * 0.80)

    orig_sub = orig_crop[y0:y1, x0:x1]
    pred_sub = pred_crop[y0:y1, x0:x1]

    mean_o = orig_sub.mean(axis=(0, 1))
    mean_p = pred_sub.mean(axis=(0, 1))

    pred_f = pred_crop.astype(np.float32)
    orig_f = orig_crop.astype(np.float32)
    blurred_p = cv2.blur(pred_f, (3, 3))
    sharpened_p = pred_f + 0.45 * (pred_f - blurred_p)
    corrected_p = sharpened_p + (mean_o - mean_p) * 0.60

    a = mask_96[:, :, np.newaxis]
    blended = a * corrected_p + (1.0 - a) * orig_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_anatomical_mouth_mask(h: int, w: int) -> np.ndarray:
    if HAVE_NUMBA:
        return _numba_anatomical_mask(h, w)
    return _numpy_anatomical_mask(h, w)


def temporal_ema_stabilizer(curr_pred: np.ndarray, prev_pred: np.ndarray) -> np.ndarray:
    if HAVE_NUMBA:
        return _numba_temporal_ema(curr_pred, prev_pred)
    return _numpy_temporal_ema(curr_pred, prev_pred)


def vfx_frequency_separation_blend(orig_crop: np.ndarray, pred_crop: np.ndarray, mask_96: np.ndarray, speech_weight: float = 1.0) -> np.ndarray:
    if HAVE_NUMBA:
        return _numba_vfx_freq_blend(orig_crop, pred_crop, mask_96, speech_weight)
    return _numpy_vfx_freq_blend(orig_crop, pred_crop, mask_96, speech_weight)


class LipBlender:
    def __init__(self, cfg=CONFIG):
        self.mouth_size = cfg.mouth_size
        self.feather_ratio = cfg.feather_ratio
        self.engine_type = "Numba AVX" if HAVE_NUMBA else "NumPy SIMD"
        self.base_mask = generate_anatomical_mouth_mask(self.mouth_size, self.mouth_size)

    def blend_affine(self, orig_frame: np.ndarray, pred_96: np.ndarray, aligned_96: np.ndarray,
                     M_inv: np.ndarray, roi_box: tuple, speech_weight: float = 1.0) -> np.ndarray:
        rx1, ry1, rx2, ry2 = roi_box
        roi_w, roi_h = rx2 - rx1, ry2 - ry1
        if roi_w <= 0 or roi_h <= 0:
            return orig_frame

        # Blend dans le repere 96x96 avec masque opaque (zero transparence de l'ancienne bouche)
        blended_96 = vfx_frequency_separation_blend(aligned_96, pred_96, self.base_mask, 1.0)

        M_inv_roi = M_inv.copy()
        M_inv_roi[0, 2] -= rx1
        M_inv_roi[1, 2] -= ry1

        orig_roi = orig_frame[ry1:ry2, rx1:rx2]
        warped_patch = cv2.warpAffine(
            blended_96, M_inv_roi, (roi_w, roi_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
        )
        warped_mask = cv2.warpAffine(
            self.base_mask, M_inv_roi, (roi_w, roi_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        m3 = warped_mask[:, :, np.newaxis]
        orig_frame[ry1:ry2, rx1:rx2] = (
            m3 * warped_patch.astype(np.float32) + (1.0 - m3) * orig_roi.astype(np.float32)
        ).astype(np.uint8)
        return orig_frame

    def blend(self, orig_frame: np.ndarray, pred_mouth: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Legacy axis-aligned blend fallback."""
        x0, y0, x1, y1 = bbox
        bw = x1 - x0
        bh = y1 - y0

        if bw <= 0 or bh <= 0:
            return orig_frame

        if bw != self.mouth_size or bh != self.mouth_size:
            pred_resized = cv2.resize(pred_mouth, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(self.base_mask, (bw, bh), interpolation=cv2.INTER_LINEAR)
        else:
            pred_resized = pred_mouth
            mask_resized = self.base_mask

        orig_roi = orig_frame[y0:y1, x0:x1]
        orig_frame[y0:y1, x0:x1] = vfx_frequency_separation_blend(orig_roi, pred_resized, mask_resized, 1.0)
        return orig_frame
