/* ForgeOCR C API. Link with -lforge_ocr. ONNX Runtime is loaded at run time from ORT_DYLIB_PATH.
 *
 * Ownership: ForgeOcr*, ForgeOcrResult* and char* returned to you are freed with their own *_free
 * function. const char* from accessors belong to the result (valid until forge_ocr_result_free).
 * On failure, functions return NULL (or -1) and forge_ocr_last_error() gives the reason for the
 * calling thread. An engine may be shared between threads.
 *
 * Copyright 2026 Znoow. Licensed under the Apache License, Version 2.0. */
#ifndef FORGE_OCR_H
#define FORGE_OCR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ForgeOcr ForgeOcr;
typedef struct ForgeOcrResult ForgeOcrResult;

const char *forge_ocr_version(void);
const char *forge_ocr_last_error(void);

/* device: NULL or "auto", "tensorrt", "cuda", "rocm", "migraphx", "directml",
 * "openvino[:GPU|NPU|CPU]", "coreml", "cpu". */
ForgeOcr *forge_ocr_new(const char *models_dir, const char *device);
void forge_ocr_free(ForgeOcr *ocr);

/* A PDF (every page) or an image file (one page). */
ForgeOcrResult *forge_ocr_read_file(const ForgeOcr *ocr, const char *path);
/* 8-bit pixels; channels 3 (RGB) or 4 (RGBA); stride in bytes, 0 = width * channels. */
ForgeOcrResult *forge_ocr_read_pixels(const ForgeOcr *ocr, const uint8_t *pixels, size_t width,
                                      size_t height, size_t channels, size_t stride);
void forge_ocr_result_free(ForgeOcrResult *r);

size_t forge_ocr_result_pages(const ForgeOcrResult *r);
int forge_ocr_page_size(const ForgeOcrResult *r, size_t page, size_t *width, size_t *height);
size_t forge_ocr_page_lines(const ForgeOcrResult *r, size_t page);
const char *forge_ocr_line_text(const ForgeOcrResult *r, size_t page, size_t line);
float forge_ocr_line_score(const ForgeOcrResult *r, size_t page, size_t line);
/* out[8] = x0,y0,x1,y1,x2,y2,x3,y3, clockwise from the top-left corner. */
int forge_ocr_line_quad(const ForgeOcrResult *r, size_t page, size_t line, float *out);

/* Free these two with forge_ocr_string_free. */
char *forge_ocr_page_text(const ForgeOcrResult *r, size_t page);
char *forge_ocr_result_json(const ForgeOcrResult *r);
void forge_ocr_string_free(char *s);

#ifdef __cplusplus
}
#endif
#endif
