"""
Dubb-in LipSync — Asymmetric ONNX Runtime GPU Inference Engine
Strictly capped at 650 MB VRAM arena for NVIDIA GTX 960 (2 GB Maxwell).
Implements explicit IO Binding for zero-copy PCIe micro-batching.
"""

import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class Wav2LipGPUEngine:
    def __init__(self, model_path: str = CONFIG.model_path, batch_size: int = CONFIG.hw.batch_size):
        resolved_path = model_path
        if not (resolved_path and os.path.exists(resolved_path)):
            candidates = [
                Path(model_path) if model_path else None,
                Path("D:/Dubb-in/lipsync/models/wav2lip_gan.onnx"),
                Path("D:/Dubb-in/models/wav2lip_gan.onnx"),
                Path(__file__).parent / "models" / "wav2lip_gan.onnx",
                Path(__file__).parent.parent / "models" / "wav2lip_gan.onnx",
            ]
            for c in candidates:
                if c and c.exists() and c.stat().st_size > 140 * 1024 * 1024:
                    resolved_path = str(c)
                    break

        self.model_path = resolved_path
        self.batch_size = batch_size
        self.mouth_size = CONFIG.mouth_size

        # Session optimization configuration
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = CONFIG.hw.cpu_threads
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        # GTX 960 2GB Safety Bounds
        cuda_provider_options = {
            "device_id": 0,
            "gpu_mem_limit": CONFIG.hw.max_vram_mb * 1024 * 1024,  # Strict 650 MB VRAM limit
            "arena_extend_strategy": CONFIG.hw.arena_strategy,
            "cudnn_conv_algo_search": CONFIG.hw.cudnn_search,
            "do_copy_in_default_stream": True,
        }

        providers = [
            ("CUDAExecutionProvider", cuda_provider_options),
            ("DmlExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider"
        ]

        print(f"[GPU] Initializing ONNX Runtime Session (Model: {self.model_path}, VRAM Cap: {CONFIG.hw.max_vram_mb} MB)...")
        self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        
        self.active_provider = self.session.get_providers()[0]
        print(f"[GPU] Active Execution Provider: {self.active_provider}")

        # Input & Output resolution
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        if "mel" in self.input_names[0].lower() or "audio" in self.input_names[0].lower():
            self.mel_name = self.input_names[0]
            self.vid_name = self.input_names[1]
        else:
            self.vid_name = self.input_names[0]
            self.mel_name = self.input_names[1]

    def run_micro_batch(self, face_crops: np.ndarray, mel_chunks: np.ndarray) -> np.ndarray:
        """
        Executes inference for a micro-batch (shape: [B, 6, 96, 96] and [B, 1, 80, 16])
        using explicit IO Binding to maximize PCIe throughput.
        """
        batch_size = face_crops.shape[0]

        # Execution via IO Binding or direct run
        try:
            io_binding = self.session.io_binding()
            io_binding.bind_cpu_input(self.vid_name, face_crops)
            io_binding.bind_cpu_input(self.mel_name, mel_chunks)

            for out_name in self.output_names:
                io_binding.bind_output(out_name)

            self.session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()
            preds = outputs[0]  # Shape: [B, 3, 96, 96]
        except Exception:
            outputs = self.session.run(self.output_names, {self.vid_name: face_crops, self.mel_name: mel_chunks})
            preds = outputs[0]

        # Normalize back to uint8 BGR
        # Handle models returning [-1, 1] or [0, 1]
        if preds.min() < 0.0:
            preds = (preds + 1.0) * 127.5
        else:
            preds = preds * 255.0

        preds = np.clip(preds, 0.0, 255.0).astype(np.uint8)

        # Transpose [B, 3, 96, 96] (CHW) -> [B, 96, 96, 3] (HWC)
        results = []
        for i in range(batch_size):
            rgb_patch = np.transpose(preds[i], (1, 2, 0))
            # Convert RGB to BGR
            bgr_patch = rgb_patch[:, :, ::-1].copy()
            results.append(bgr_patch)

        return np.array(results, dtype=np.uint8)


def download_wav2lip_model(target_path: str = None, progress_callback=None) -> str:
    """Downloads the optimized wav2lip_gan.onnx model (~145MB) with progress feedback."""
    import urllib.request
    from pathlib import Path

    dest = Path(target_path or CONFIG.model_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 140 * 1024 * 1024:
        return str(dest)

    url = "https://huggingface.co/bluefoxcreation/Wav2lip-Onnx/resolve/main/wav2lip_gan.onnx"
    temp_dest = dest.with_suffix(".tmp")

    def _hook(blocks, block_size, total_size):
        if progress_callback and total_size > 0:
            downloaded = blocks * block_size
            pct = min(1.0, downloaded / total_size)
            progress_callback(pct, downloaded, total_size)

    urllib.request.urlretrieve(url, str(temp_dest), reporthook=_hook)
    if temp_dest.exists():
        temp_dest.replace(dest)
    return str(dest)
