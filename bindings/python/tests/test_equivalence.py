"""Python binding vs the CLI: same JSON for every page (speed: bench_overhead.py).

    ORT_DYLIB_PATH=... python tests/test_equivalence.py MODELS_DIR DEVICE FILE...

Needs the `forge-ocr` binary (FORGE_OCR_CLI, default ../../target/release/forge-ocr), numpy and Pillow.
For images, read_image() gets the pixels decoded by Pillow, so the CLI is run on a lossless PPM of
those same pixels (JPEG decoders differ by a few levels, which moves boxes slightly).
"""
import json, os, subprocess, sys, tempfile

import numpy as np
from PIL import Image

import forge_ocr

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.environ.get("FORGE_OCR_CLI", os.path.join(HERE, "..", "..", "..", "target", "release", "forge-ocr"))


def cli(models, device, files):
    out = subprocess.run([CLI, "--models", models, "--device", device, "--json", *files], check=True, capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def main():
    models, device, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    forge_ocr.Engine(models, device=device)  # an earlier engine must not block the next one
    ocr = forge_ocr.Engine(models, device=device)

    # 1. read_file: byte-identical JSON to the CLI.
    ref = cli(models, device, files)
    got = [ocr.read_file(f) for f in files]
    n_pages = sum(len(p) for p in got)
    for r, pages in zip(ref, got):
        assert len(r["pages"]) == len(pages), r["file"]
        for rp, p in zip(r["pages"], pages):
            assert json.loads(p.json()) == rp, f"{r['file']}: page differs"
    print(f"read_file: {n_pages} pages identical to the CLI")

    # 2. read_image (numpy, RGB and RGBA, contiguous and sliced) vs the CLI on the same pixels.
    imgs = [f for f in files if not f.lower().endswith(".pdf")]
    with tempfile.TemporaryDirectory() as d:
        arrays, ppms = [], []
        for f in imgs:
            a = np.asarray(Image.open(f).convert("RGB"))
            arrays.append(a)
            ppms.append(os.path.join(d, os.path.basename(f) + ".ppm"))
            Image.fromarray(a).save(ppms[-1])
        ref = cli(models, device, ppms)
        for a, r in zip(arrays, ref):
            want = r["pages"][0]
            rgba = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
            padded = np.zeros((a.shape[0], a.shape[1] + 7, 3), np.uint8)
            padded[:, :a.shape[1]] = a
            for name, x in [("rgb", a), ("rgba", rgba), ("strided", padded[:, :a.shape[1]])]:
                assert json.loads(ocr.read_image(x).json()) == want, f"{r['file']}: read_image({name}) differs"
        # A batch pads lines of several pages together in the recognizer, which moves scores in
        # the 3rd decimal; text and boxes stay the same.
        for p, r in zip(ocr.read_images(arrays), ref):
            got = [(l["text"], l["quad"]) for l in json.loads(p.json())["lines"]]
            assert got == [(l["text"], l["quad"]) for l in r["pages"][0]["lines"]], f"{r['file']}: read_images differs"
    print(f"read_image: {len(imgs)} pages identical to the CLI (RGB, RGBA, strided; batch: same text and boxes)")


if __name__ == "__main__":
    main()
