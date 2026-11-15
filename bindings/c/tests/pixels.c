/* pixels MODELS_DIR DEVICE FILE.ppm : reads a binary PPM (P6) as RGBA with padded rows through
 * forge_ocr_read_pixels and prints each page's text. Checks the stride and alpha paths. */
#include <stdio.h>
#include <stdlib.h>
#include "forge_ocr.h"

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *f = fopen(argv[3], "rb");
    size_t w, h; int max;
    if (!f || fscanf(f, "P6 %zu %zu %d", &w, &h, &max) != 3 || max != 255) return 2;
    fgetc(f);
    size_t stride = w * 4 + 16; /* padded rows */
    uint8_t *px = calloc(stride * h, 1);
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            uint8_t *p = px + y * stride + x * 4;
            if (fread(p, 1, 3, f) != 3) return 2;
            p[3] = 255;
        }
    fclose(f);
    ForgeOcr *ocr = forge_ocr_new(argv[1], argv[2]);
    if (!ocr) { fprintf(stderr, "%s\n", forge_ocr_last_error()); return 1; }
    ForgeOcr *second = forge_ocr_new(argv[1], argv[2]); /* several engines per process */
    if (!second) { fprintf(stderr, "second engine: %s\n", forge_ocr_last_error()); return 1; }
    forge_ocr_free(second);
    ForgeOcrResult *bad = forge_ocr_read_pixels(ocr, px, w, h, 2, 0);
    if (bad || !*forge_ocr_last_error()) { fprintf(stderr, "channels=2 should fail\n"); return 1; }
    ForgeOcrResult *r = forge_ocr_read_pixels(ocr, px, w, h, 4, stride);
    if (!r) { fprintf(stderr, "%s\n", forge_ocr_last_error()); return 1; }
    char *t = forge_ocr_page_text(r, 0);
    printf("%s\n", t);
    forge_ocr_string_free(t);
    float q[8];
    if (forge_ocr_line_quad(r, 0, 0, q) != 0 || forge_ocr_line_text(r, 9, 0) != NULL) return 1;
    forge_ocr_result_free(r);
    forge_ocr_free(ocr);
    free(px);
    return 0;
}
