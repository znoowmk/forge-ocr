#!/usr/bin/env python3
"""Makes the FP16 copy of the Latin recognizer (about twice as fast on CUDA; not needed with
--trt, which builds FP16 engines itself).  Needs: pip install onnx onnxconverter-common

    python3 scripts/fp16.py models
"""
import sys
from pathlib import Path

import onnx
from onnxconverter_common import float16

d = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
src, dst = d / "latin_PP-OCRv5_rec_mobile.onnx", d / "latin_PP-OCRv5_rec_mobile_fp16.onnx"
onnx.save(float16.convert_float_to_float16(onnx.load(src), keep_io_types=True), dst)
print(f"wrote {dst}")
