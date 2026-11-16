"""
Dubb-in LipSync — Lightweight Mel Spectrogram Extractor
Pure NumPy & SciPy implementation. Zero Librosa, Zero PyTorch.
"""

import subprocess
import numpy as np
import scipy.signal
try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class LightweightMelExtractor:
    """Extracts 80-bin Mel-spectrogram chunks exactly formatted for Wav2Lip."""

    def __init__(self, cfg=CONFIG):
        self.sr = cfg.sample_rate
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.n_mels = cfg.n_mels
        self.mel_step_size = cfg.mel_step_size
        self.mel_basis = self._build_mel_basis(self.sr, self.n_fft, self.n_mels, 55.0, 7600.0)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_basis(self, sr, n_fft, n_mels, fmin, fmax):
        n_freqs = int(1 + n_fft // 2)
        fftfreqs = np.linspace(0, sr / 2.0, n_freqs)
        mel_f = self._mel_to_hz(np.linspace(self._hz_to_mel(fmin), self._hz_to_mel(fmax), n_mels + 2))

        weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            lower = (fftfreqs - mel_f[i]) / (mel_f[i + 1] - mel_f[i])
            upper = (mel_f[i + 2] - fftfreqs) / (mel_f[i + 2] - mel_f[i + 1])
            weights[i] = np.maximum(0.0, np.minimum(lower, upper))
            enorm = 2.0 / (mel_f[i + 2] - mel_f[i])
            weights[i] *= enorm
        return weights

    def extract_from_file(self, audio_path: str, fps: float, return_weights: bool = False, **kwargs):
        """Converts audio to 16kHz mono via FFmpeg memory pipe, then computes Mel chunks."""
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", audio_path,
            "-ac", "1", "-ar", str(self.sr),
            "-f", "s16le", "-"
        ]
        raw_audio = subprocess.check_output(cmd)
        wav = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Pre-emphasis filter
        wav = scipy.signal.lfilter([1.0, -0.97], [1.0], wav).astype(np.float32)

        # Windowing & STFT
        window = scipy.signal.get_window("hann", self.win_length, fftbins=True)
        pad_len = int(self.n_fft // 2)
        wav_padded = np.pad(wav, (pad_len, pad_len), mode="reflect")

        num_frames = 1 + (len(wav_padded) - self.n_fft) // self.hop_length
        strides = (wav_padded.strides[0] * self.hop_length, wav_padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(wav_padded, shape=(num_frames, self.n_fft), strides=strides)

        spec = np.abs(np.fft.rfft(frames * window, n=self.n_fft, axis=1)).T
        mel_spec = np.dot(self.mel_basis, spec)

        # Convert to dB scale and normalize [-4.0, 4.0]
        min_level = np.exp(-5.0 * np.log(10.0))
        mel_db = 20.0 * np.log10(np.maximum(min_level, mel_spec)) - 20.0
        mel_norm = np.clip((8.0) * ((mel_db + 100.0) / 100.0) - 4.0, -4.0, 4.0).astype(np.float32)

        # Silence Gate (Noise gate below -3.2 -> clamp to -4.0 so mouth closes naturally)
        mel_norm = np.where(mel_norm < -3.2, -4.0, mel_norm).astype(np.float32)

        # 8-step edge padding to center 16-step window exactly on video frame
        mel_padded = np.pad(mel_norm, ((0, 0), (8, 8)), mode="edge")

        # Slice into 16-step windows indexed by video FPS (+2 anticipation steps for plosives P/B/M)
        mel_idx_multiplier = 80.0 / float(fps)
        mel_chunks = []
        speech_weights = []
        i = 0
        total_time_steps = mel_padded.shape[1]
        smooth_w = 1.0

        while True:
            start_idx = int(i * mel_idx_multiplier) + 2
            if start_idx + self.mel_step_size > total_time_steps:
                last_chunk = mel_padded[:, -self.mel_step_size:]
                if last_chunk.shape[1] < self.mel_step_size:
                    pad_w = self.mel_step_size - last_chunk.shape[1]
                    last_chunk = np.pad(last_chunk, ((0, 0), (0, pad_w)), mode="edge")
                mel_chunks.append(last_chunk)
                speech_weights.append(smooth_w)
                break
            chunk = mel_padded[:, start_idx : start_idx + self.mel_step_size]
            mel_chunks.append(chunk)

            # Vocal activity detection (energy of frequency peak in central window)
            center_energy = float(np.percentile(chunk[:, 5:11], 90))
            raw_w = 1.0 if center_energy > -2.8 else 0.70
            smooth_w = raw_w if raw_w > smooth_w else (0.75 * smooth_w + 0.25 * raw_w)
            speech_weights.append(smooth_w)
            i += 1

        if return_weights:
            return mel_chunks, speech_weights
        return mel_chunks

    def extract_mel_chunks(self, audio_path: str, fps: float, return_weights: bool = True, **kwargs):
        """Convenience method returning mel_chunks and optionally speech_weights."""
        return self.extract_from_file(audio_path, fps, return_weights=return_weights, **kwargs)
