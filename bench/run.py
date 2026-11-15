#!/usr/bin/env python3
"""Runs OCR engines over the pages from make_pages.py and scores them against the known text.

    python3 bench/run.py PAGES_DIR --engines forge,tesseract,rapidocr [--models DIR]

Scores, per engine:
- Word F1: words read right (multiset match against the known text), whatever the line order or
  segmentation. The fairest single number.
- CER: character error rate. Each ground-truth line is matched to the most similar output line
  (so reading order does not count), then Levenshtein distance / ground-truth length.
- IDs exact: share of the 11-digit IDs in the tables read exactly (spaces ignored).
- Pages/s: wall time for the whole set, models loaded once.

Needs rapidfuzz. Engines: forge (forge-ocr on PATH or FORGE_OCR), forge-trt (TensorRT, engines cached in
FORGE_TRT_CACHE), forge-cpu, tesseract (spa+eng,
psm 3, one OpenMP thread, as many processes as cores), rapidocr (stock RapidOCR, CUDA if
available).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


def run_forge(pages: list[Path], out: Path, models: str, cpu: bool, trt: str | None = None) -> None:
    exe = os.environ.get("FORGE_OCR", "forge-ocr")
    cmd = [exe, "--models", models, "--out-dir", str(out)] + (["--cpu"] if cpu else []) + (["--trt", trt] if trt else []) + [str(p) for p in pages]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def run_tesseract(pages: list[Path], out: Path) -> None:
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    procs = []
    for p in pages:
        procs.append(subprocess.Popen(["tesseract", str(p.with_suffix(".jpg")), str(out / p.stem), "-l", "spa+eng", "--psm", "3"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        while sum(x.poll() is None for x in procs) >= os.cpu_count():
            time.sleep(0.01)
    for x in procs:
        x.wait()


def rows(boxes, texts) -> list[str]:
    """Boxes regrouped into printed rows (the rule forge-ocr uses), so every engine is scored on
    rows, not on how it happens to split them."""
    items = []
    for b, t in zip(boxes, texts):
        ys = [p[1] for p in b]
        y0, y1 = min(ys), max(ys)
        items.append(((y0 + y1) / 2, max(y1 - y0, 1.0), min(p[0] for p in b), t))
    items.sort()
    out, cur, cy, ch = [], [], 0.0, 0.0
    for y, h, x, t in items:
        if cur and abs(y - cy) > 0.5 * min(h, ch):
            out.append(" ".join(t for _, t in sorted(cur)))
            cur = []
        if not cur:
            cy, ch = y, h
        cur.append((x, t))
    if cur:
        out.append(" ".join(t for _, t in sorted(cur)))
    return out


def run_rapidocr(pages: list[Path], out: Path) -> None:
    from rapidocr import RapidOCR

    try:
        import onnxruntime

        onnxruntime.preload_dlls()
        ocr = RapidOCR(params={"Global.log_level": "error", "EngineConfig.onnxruntime.use_cuda": True})
    except Exception:
        ocr = RapidOCR(params={"Global.log_level": "error"})
    for p in pages:
        r = ocr(str(p.with_suffix(".jpg")))
        (out / f"{p.stem}.txt").write_text("\n".join(rows(r.boxes, r.txts) if r.txts else []))


def run_rapidocr_latin(pages: list[Path], out: Path) -> None:
    """RapidOCR set up fairly for Latin text: PP-OCRv5 detector and Latin recognizer, CUDA."""
    import onnxruntime
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    onnxruntime.preload_dlls()
    ocr = RapidOCR(params={
        "Global.log_level": "error",
        "Det.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": LangDet.MULTI, "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
        "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.lang_type": LangRec.LATIN, "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
        "EngineConfig.onnxruntime.use_cuda": True,
    })
    for p in pages:
        r = ocr(str(p.with_suffix(".jpg")))
        (out / f"{p.stem}.txt").write_text("\n".join(rows(r.boxes, r.txts) if r.txts else []))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def score(gt_dir: Path, out: Path, pages: list[Path]) -> tuple[float, float, float]:
    errors = chars = 0
    ids_total = ids_ok = 0
    w_hit = w_gt = w_got = 0
    for p in pages:
        gt = [norm(l) for l in (gt_dir / f"{p.stem}.gt.txt").read_text().splitlines() if l.strip()]
        f = out / f"{p.stem}.txt"
        got = [norm(l) for l in (f.read_text(errors="replace").splitlines() if f.exists() else []) if l.strip()]
        for line in gt:
            best = process.extractOne(line, got, scorer=Levenshtein.normalized_similarity) if got else None
            d = Levenshtein.distance(line, best[0]) if best else len(line)
            errors += min(d, len(line))
            chars += len(line)
        # Words, independent of how lines are cut or ordered (multiset match).
        from collections import Counter

        cg, co = Counter(" ".join(gt).split()), Counter(" ".join(got).split())
        w_hit += sum((cg & co).values())
        w_gt += sum(cg.values())
        w_got += sum(co.values())
        text = re.sub(r"[ \-]", "", " ".join(got))
        for nss in (l.split()[0] for l in gt if re.match(r"^\d{11} ", l)):
            ids_total += 1
            ids_ok += nss in text
    rec, prec = w_hit / max(w_gt, 1), w_hit / max(w_got, 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    return errors / max(chars, 1), ids_ok / max(ids_total, 1), f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("--engines", default="forge,tesseract,rapidocr")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    d = Path(a.pages)
    pages = sorted(d.glob("p*.pdf"))
    print(f"{len(pages)} pages\n")
    print(f"| Engine | Pages/s | Word F1 | CER (rows) | IDs exact |\n|---|---|---|---|---|")
    for eng in a.engines.split(","):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            t = time.time()
            if eng == "forge":
                run_forge(pages, out, a.models, cpu=False)
            elif eng == "forge-trt":
                run_forge(pages, out, a.models, cpu=False, trt=os.environ.get("FORGE_TRT_CACHE", "trt-cache"))
            elif eng == "forge-cpu":
                run_forge(pages, out, a.models, cpu=True)
            elif eng == "tesseract":
                run_tesseract(pages, out)
            elif eng == "rapidocr":
                run_rapidocr(pages, out)
            elif eng == "rapidocr-latin":
                run_rapidocr_latin(pages, out)
            else:
                sys.exit(f"unknown engine {eng}")
            dt = time.time() - t
            cer, ids, f1 = score(d, out, pages)
            print(f"| {eng} | {len(pages) / dt:.2f} | {f1 * 100:.1f} % | {cer * 100:.2f} % | {ids * 100:.1f} % |", flush=True)


if __name__ == "__main__":
    main()
