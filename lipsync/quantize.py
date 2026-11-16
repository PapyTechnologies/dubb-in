"""
Dubb-in LipSync — Dynamic INT8 Quantizer
Quantizes Wav2Lip ONNX model to INT8, cutting disk size to ~38MB and VRAM footprint to <300MB.
"""

import sys
import argparse
from pathlib import Path


def quantize_model(input_onnx: str, output_onnx: str = None):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[ERROR] Please install onnx & onnxruntime: pip install onnx onnxruntime")
        return

    in_p = Path(input_onnx)
    if not in_p.exists():
        print(f"[ERROR] Input model not found: {input_onnx}")
        return

    if output_onnx is None:
        output_onnx = str(in_p.with_name(in_p.stem + "_int8.onnx"))

    print(f"[QUANT] Quantizing {input_onnx} -> {output_onnx} (QuantType.QInt8)...")
    quantize_dynamic(
        model_input=input_onnx,
        model_output=output_onnx,
        weight_type=QuantType.QInt8,
    )
    print(f"[DONE] INT8 model created: {output_onnx} ({Path(output_onnx).stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Wav2Lip ONNX model to dynamic INT8")
    parser.add_argument("--model", "-m", default="models/wav2lip_gan.onnx", help="Input ONNX model")
    parser.add_argument("--output", "-o", default=None, help="Output INT8 ONNX path")
    args = parser.parse_args()
    quantize_model(args.model, args.output)
