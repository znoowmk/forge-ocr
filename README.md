# ForgeOCR

Fast, accurate OCR for scanned documents, in Rust.

ForgeOCR runs the PP-OCRv5 detector and recognizer on ONNX Runtime (TensorRT, CUDA or CPU).
Everything around those two networks is native Rust and runs in parallel: reading the scan out
of the PDF, preparing the image, finding text boxes, cutting them out, batching, decoding and
putting rows back in reading order. With TensorRT it reads about **10 pages per second** on a
mid-range GPU. It stays accurate on poor scans, where Tesseract falls apart.

```bash
forge-ocr --models models/ scan.pdf                      # text: one printed row per line, pages split by \f
forge-ocr --models models/ --trt ~/.cache/forge-ocr scan.pdf   # TensorRT FP16 (fastest)
forge-ocr --models models/ --json scan.pdf               # every line with its score and corners
forge-ocr --models models/ --out-dir out/ *.pdf          # one .txt per input
forge-ocr --models models/ --check luhn:16 receipts.pdf  # check-digit correction for 16-digit numbers
forge-ocr --models models/ --cpu photo.jpg               # no GPU
```

## Benchmark

`bench/` makes synthetic scanned pages with known text and scores every engine against it, so
anyone can reproduce the numbers:

```bash
python3 bench/make_pages.py /tmp/pages --pages 60            # clean scans
python3 bench/make_pages.py /tmp/hard --pages 60 --hard      # poor scans: skew, blur, low resolution, noise, uneven light
python3 bench/run.py /tmp/pages --engines forge-trt,forge,tesseract,rapidocr-latin,rapidocr
```

The pages are US letter at 300 dpi: Spanish prose with accents, a payroll-style table (names,
11-digit IDs with a check digit, amounts, dates), then degraded like a scan. Results on an
RTX 3060 12 GB with 16 CPU threads, 60 pages each:

**Clean scans**

| Engine | Pages/s | Words right (F1) | IDs exact |
|---|---|---|---|
| **ForgeOCR, TensorRT** | **9.92** | **100.0 %** | 100 % |
| ForgeOCR, CUDA | 4.71 | 100.0 % | 100 % |
| Tesseract 5.5 (spa+eng, 16 processes) | 7.30 | 97.5 % | 100 % |
| RapidOCR, PP-OCRv5 Latin, CUDA (same models) | 2.65 | 96.9 % | 100 % |
| RapidOCR, stock | 2.29 | 95.4 % | 100 % |

**Poor scans** (`--hard`)

| Engine | Pages/s | Words right (F1) | IDs exact |
|---|---|---|---|
| **ForgeOCR, TensorRT** | **9.94** | **99.2 %** | 99.8 % |
| ForgeOCR, CUDA | 4.70 | 99.2 % | 100 % |
| Tesseract 5.5 | 8.06 | 42.4 % | 38.1 % |
| RapidOCR, PP-OCRv5 Latin, CUDA | 2.70 | 98.0 % | 100 % |
| RapidOCR, stock | 2.36 | 97.4 % | 100 % |

"Words right" matches words regardless of line order or how an engine splits rows. Every engine's
output is grouped into rows the same way before scoring. On 201 real scanned pages (Mexican
social-security and training records, not published), ForgeOCR read 4.5 pages/s on CUDA against
2.9 for RapidOCR tuned the same way, with as many valid IDs.

## Why it is faster

Profiling PP-OCR's Python pipeline on real scans showed the two networks were a small part of
the time. The rest was Python around them:
- normalising the whole page;
- cutting and resizing some 150 crops a page one by one;
- the detector's post-processing in OpenCV and pyclipper;
- small recognizer batches.

ForgeOCR does all of that in parallel Rust (per page: 9 ms to prepare the detector input, 2 ms
to find boxes, 13 ms to cut crops, 1 ms to decode). It reads a scan's JPEG as stored instead of
re-rendering it, runs the recognizer in FP16 on batches sorted by width, and with TensorRT
compiles both networks for your GPU.

```
PDF page ─► embedded JPEG decoded as stored, or rendered (pdftoppm)
         ─► detector input: long side ≤ 1600, sides multiples of 32, normalised (parallel)
         ─► DB detector ─► probability map
         ─► boxes: threshold, 2×2 dilation, 8-connected components, convex hull,
            minimum-area rectangle, mean score, unclip (grow by area×1.6/perimeter)
         ─► crops: each box sampled straight from the full page to 48 px high (one bilinear pass)
         ─► recognizer: crops sorted by shape, batches padded to multiples of 64 px, FP16
         ─► CTC (greedy, keeping each character's probability)
         ─► optional check-digit correction ─► rows by position
```

**Check-digit correction** (`--check luhn:N`). When a number of N digits fails its check digit,
ForgeOCR looks at the digits the network doubted (probability below 0.6). If changing one of
them to the alternative it also considered (0.2 or more) makes the check pass, it applies the
most likely such change. Clear readings are never changed. On the real pages above it fixed 3
digits (valid IDs 234 → 236). Most wrong IDs there were read confidently wrong or missed a digit,
which this rule does not try to fix.

## Install

1. **ONNX Runtime 1.31 or newer.** The simplest source is the Python wheel: `pip install
   "onnxruntime-gpu[cuda,cudnn]"` (or `onnxruntime` for CPU only). Point `ORT_DYLIB_PATH` to its
   `libonnxruntime.so`. For the GPU, put the wheel's `capi/` folder and the `nvidia/*/lib`
   folders on `LD_LIBRARY_PATH`. For TensorRT, also `pip install tensorrt-cu13-libs==10.*` and add
   its `tensorrt_libs` folder.
2. **Models:** `scripts/fetch-models.sh models`. It downloads and checks the PP-OCRv5 detector
   and Latin recognizer. Optionally run `python3 scripts/fp16.py models` for the FP16 recognizer
   used on CUDA.
3. **Build:** `cargo build --release`. The binary is `target/release/forge-ocr`. PDFs whose pages
   aren't a single scan need `pdftoppm` (poppler) to render them.

The first `--trt` run builds TensorRT engines for your GPU (about 10 minutes). They are cached in
the folder you give and reused afterwards.

## Any GPU, or none

`--device` (or `FORGE_OCR_DEVICE`) chooses where the networks run. Each choice is an ONNX
Runtime execution provider, so the ONNX Runtime library you point `ORT_DYLIB_PATH` at must
include it. The official packages each carry a different set:

| Hardware | `--device` | ONNX Runtime package |
|---|---|---|
| NVIDIA | `tensorrt` (fastest, with `--trt CACHE`), `cuda` | `onnxruntime-gpu` (+ `tensorrt-cu13-libs`) |
| Any GPU on Windows (NVIDIA, AMD, Intel) | `directml` | `onnxruntime-directml` |
| Intel CPU, Arc GPU or NPU | `openvino:CPU`, `openvino:GPU`, `openvino:NPU` | `onnxruntime-openvino` |
| AMD on Linux | `rocm`, `migraphx` | ONNX Runtime ROCm build |
| Apple | `coreml` | `onnxruntime` for macOS |
| Anything | `cpu` | `onnxruntime` |

The default, `auto`, tries them in that order and skips the ones the library lacks. Asking for
one by name fails with a clear message when the library doesn't have it.

On the CPU, ForgeOCR switches to the FP32 recognizer (FP16 has no fast CPU kernels), batches
of 8 lines (they stay in cache: about 2.5 times faster per line than 32) and one thread per
physical core. INT8 quantization was measured too and is not used: dynamic INT8 ran 6 to 12
times slower on this recognizer.

Measured with the same text out of every path:

| Machine | Path | Pages/s |
|---|---|---|
| RTX 3060 12 GB | `tensorrt` | 6.9 |
| RTX 3060 12 GB | `cuda` | 3.4 |
| Ryzen 7 5800X | `cpu`, before / after the CPU settings | 0.2 / 0.9 |
| i7-12800HX | `cpu` | 0.8 |
| i7-12800HX | `openvino:CPU` | 1.8 |
| i7-12800HX, UHD 770 integrated | `openvino:GPU` | 0.4 |

DirectML, ROCm, MIGraphX and CoreML are wired in and compile, but haven't been run on that
hardware yet. Reports are welcome.

## As a library

```rust
let mut cfg = forge_ocr::Config::from_dir("models")?;
cfg.trt_cache = Some("/var/cache/forge-ocr".into());
let engine = forge_ocr::Engine::new(cfg)?;
for page in engine.read_pdf("scan.pdf".as_ref())? {
    println!("{}", page.text());
    for line in &page.lines {
        println!("{:.2} {:?} {}", line.score, line.quad, line.text);
    }
}
```

### From Python

```bash
cd bindings/python && maturin build --release && pip install target/wheels/forge_ocr-*.whl
```

```python
import forge_ocr, numpy as np
from PIL import Image

ocr = forge_ocr.Engine("models", device="cuda")       # same devices as --device
for page in ocr.read_file("scan.pdf"):                # PDF or image file
    print(page.text())
page = ocr.read_image(np.asarray(Image.open("photo.png")))  # HxWx3 or HxWx4 uint8, any strides
for line in page.lines:
    print(line.score, line.quad, line.text)
```

`read_images([...])` reads several arrays in one batch. The GIL is released while the networks
run. `page.json()` is the same JSON as `forge-ocr --json`.

### From C (or anything with a C FFI)

`cargo build --release -p forge-ocr-c` builds `target/release/libforge_ocr.so` (and `.a`); the
header is `bindings/c/include/forge_ocr.h`.

```c
ForgeOcr *ocr = forge_ocr_new("models", "cuda");        /* NULL: see forge_ocr_last_error() */
ForgeOcrResult *r = forge_ocr_read_file(ocr, "scan.pdf");
for (size_t p = 0; p < forge_ocr_result_pages(r); p++) {
    char *text = forge_ocr_page_text(r, p);
    puts(text);
    forge_ocr_string_free(text);
}
forge_ocr_result_free(r);
forge_ocr_free(ocr);
```

`forge_ocr_read_pixels` takes RGB/RGBA buffers with any row stride, and the line accessors give
text, score and corners. `bindings/c/examples/read.c` prints the same JSON as the CLI.

### Tests of the bindings

Both bindings give the CLI's exact output: `bindings/c/examples/read.c` and
`bindings/python/tests/test_equivalence.py` compare their JSON with `forge-ocr --json` byte for
byte (images, PDFs, and NumPy arrays as RGB, RGBA and strided views). `read_images` on several
arrays gives the same text and boxes; scores move in the third decimal because the recognizer pads
lines of all the pages together. `tests/bench_overhead.py` times Python against the CLI: on the six
test pages (CPU, i7-12800HX) 1618 vs 1667 ms per page, so the binding adds nothing measurable.
Under `valgrind`, `examples/read.c` (image, PDF, missing file) and `tests/pixels.c` (RGBA with
stride, bad arguments, two engines) lose nothing; the only "possibly lost" blocks are the
thread-local storage of ONNX Runtime's and rayon's worker threads, which live until the process ends.

## Roadmap

- Overlap the CPU stages with the GPU (the GPU waits while pages are prepared).
- Prebuilt binaries and wheels with ONNX Runtime bundled.
- Searchable PDF output (`--pdf`), Node bindings and a browser build.
- Test runs on AMD (ROCm, DirectML), Intel Arc and Apple hardware.
- Layout: columns and tables as structure, not only rows.
- More languages: PP-OCRv5 recognizers exist for other scripts.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributors accept the [CLA](CLA.md) once.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`), © Znoow. The PP-OCRv5 models are PaddleOCR's, also
Apache-2.0, and are downloaded separately.
