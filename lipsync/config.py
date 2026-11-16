"""
Dubb-in LipSync — Hardware & Pipeline Configuration
Target: Intel Core i5-3470 (AVX) | 12 GB DDR3 | NVIDIA GTX 960 (2 GB VRAM)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HardwareProfile:
    # GPU Constraints (GTX 960 Maxwell 2GB)
    max_vram_mb: int = 650  # Hard VRAM arena cap to avoid OOM
    batch_size: int = 4     # Micro-batch (1 to 4)
    use_gpu: bool = True
    arena_strategy: str = "kSameAsRequested"
    cudnn_search: str = "HEURISTIC"

    # CPU Constraints (i5-3470 4C/4T AVX)
    cpu_threads: int = 3    # Leaves 1 core free for OS & I/O
    use_avx: bool = True    # AVX supported, AVX2 not available

    # DDR3 RAM Bounds (Bounded Queues)
    raw_frame_queue_size: int = 24
    crop_queue_size: int = 16
    mel_queue_size: int = 8
    output_queue_size: int = 24


@dataclass
class LipSyncConfig:
    # Dimensions & Model
    mouth_size: int = 96
    model_path: str = str(Path(__file__).parent / "models" / "wav2lip_gan.onnx")

    # Audio specifications (Standard Wav2Lip 80-bin Mel)
    sample_rate: int = 16000
    n_mels: int = 80
    mel_step_size: int = 16
    n_fft: int = 800
    hop_length: int = 200
    win_length: int = 800

    # Face Tracking Optimization (High reactivity + jump protection)
    detect_every_n_frames: int = 2  # Track 1 frame in 2 for 2x faster tracking response
    smooth_factor: float = 0.35     # Reactive EMA filtering (no mouth float)

    # Blending & Visual Quality
    feather_ratio: float = 0.35
    unsharp_strength: float = 0.35
    color_transfer_weight: float = 0.65

    # Hardware & Memory Profile
    hw: HardwareProfile = HardwareProfile()


CONFIG = LipSyncConfig()
