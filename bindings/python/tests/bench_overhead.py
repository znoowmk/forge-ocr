"""Time per page: CLI vs Python read_file, interleaved runs, best of N (the machine may be busy).

    ORT_DYLIB_PATH=... python tests/bench_overhead.py MODELS_DIR DEVICE ROUNDS FILE...
"""
import os, subprocess, sys, time

import forge_ocr

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.environ.get("FORGE_OCR_CLI", os.path.join(HERE, "..", "..", "..", "target", "release", "forge-ocr"))
models, device, rounds, files = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
cli_best = py_best = float("inf")
for _ in range(rounds):
    err = subprocess.run([CLI, "--models", models, "--device", device, *files], check=True, capture_output=True, text=True).stderr
    cli_best = min(cli_best, float(err.split(" lines in ")[1].split(" s")[0]))
    ocr = forge_ocr.Engine(models, device=device)  # fresh, like the CLI (load not timed)
    t = time.perf_counter()
    for f in files:
        ocr.read_file(f)
    py_best = min(py_best, time.perf_counter() - t)
n = len(files)
print(f"{device}: CLI {1e3 * cli_best / n:.0f} ms/page, Python {1e3 * py_best / n:.0f} ms/page ({100 * (py_best / cli_best - 1):+.1f} %), best of {rounds}")
