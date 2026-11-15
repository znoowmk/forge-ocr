# ForgeOCR

OCR for scanned documents. It runs the PP-OCRv5 detector and recognizer (ONNX) on ONNX Runtime;
everything around the two networks is native Rust and runs in parallel. That covers reading the
scan out of the PDF, resizing, finding the text boxes, cutting them out, batching, CTC decoding
and reading order.

```bash
forge-ocr --models models/ scan.pdf            # text, pages separated by \f, one printed row per line
forge-ocr --models models/ --json scan.pdf     # every line with its score and corners
forge-ocr --models models/ --cpu photo.jpg     # without a GPU
```

## Why it is faster

PP-OCR's own pipeline is Python around two neural networks. Profiled on real scans, the networks
were a small part of the time. The rest was Python:
- normalising the full page;
- cutting and resizing about 150 crops a page one by one;
- the detector's post-processing with OpenCV and pyclipper;
- small recognizer batches.

ForgeOCR does those in Rust:

| Stage (per page) | RapidOCR (Python) | ForgeOCR |
|---|---|---|
| Detector input (resize + normalise) | ~49 ms | 9 ms |
| Detector post-processing (boxes) | part of 125 ms | 2 ms |
| Crops (cut, straighten, resize) | ~40 ms | 13 ms |
| CTC decoding | ~5 ms | 1 ms |

Measured on 201 real scanned pages (16 PDFs of Mexican social-security and training records),
RTX 3060 12 GB, 16 threads:

| | Pages/s | Valid IDs read* |
|---|---|---|
| RapidOCR, stock settings | 1.7 | 214 |
| RapidOCR, tuned (FP16, no orientation net, detection at 1600 px) | 2.9 | 232 |
| **ForgeOCR 0.1** | **4.5** | **234** |

\* NSS numbers whose check digit is right. The text of the two engines agrees 89–98 % per
character.

The recognizer on the GPU is now most of the time, so the next steps are TensorRT FP16 and running
the CPU stages while the GPU works.

## Design

```
PDF page ─► embedded JPEG decoded as stored, or rendered (pdftoppm)
         ─► detector input: resize so the long side ≤ 1600, sides multiples of 32,
            normalise; parallel over rows
         ─► DB detector (ONNX Runtime, CUDA) ─► probability map
         ─► boxes: threshold, 2×2 dilation, 8-connected components, convex hull,
            minimum-area rectangle, mean score, unclip (grow by area×1.6/perimeter)
         ─► crops: each box sampled straight from the full page to 48 px high
            (one bilinear pass)
         ─► recognizer: crops of a chunk of pages sorted by shape, batches
            padded to multiples of 64 px, FP16
         ─► CTC (greedy, keeps each character's probability) ─► rows by position
```

## Models

PP-OCRv5 mobile detector and the Latin recognizer, from PaddleOCR (Apache-2.0), as ONNX. Put them
in the models folder as `det.onnx` and `rec.onnx`, or under their original names. An FP16 copy of
the recognizer is picked up when present (`latin_PP-OCRv5_rec_mobile_fp16.onnx`).

## Running it

ForgeOCR loads ONNX Runtime at start. Point `ORT_DYLIB_PATH` to `libonnxruntime` (1.31 or
newer). For the GPU, ONNX Runtime's CUDA provider and the CUDA and cuDNN libraries must be on the
library path.
