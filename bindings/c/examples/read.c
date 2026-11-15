/* forge-ocr-c MODELS_DIR DEVICE FILE... : prints {"file":..., "pages":[...]} per file, the same
 * output as `forge-ocr --json`. */
#include <stdio.h>
#include <string.h>
#include "forge_ocr.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "use: %s MODELS_DIR DEVICE FILE...\n", argv[0]);
        return 2;
    }
    ForgeOcr *ocr = forge_ocr_new(argv[1], argv[2]);
    if (!ocr) {
        fprintf(stderr, "forge_ocr_new: %s\n", forge_ocr_last_error());
        return 1;
    }
    int status = 0;
    for (int i = 3; i < argc; i++) {
        ForgeOcrResult *r = forge_ocr_read_file(ocr, argv[i]);
        if (!r) {
            fprintf(stderr, "%s: %s\n", argv[i], forge_ocr_last_error());
            status = 1;
            continue;
        }
        char *json = forge_ocr_result_json(r);
        /* {"pages":[...]} -> {"file":"...","pages":[...]} (file names here have no quotes) */
        printf("{\"file\":\"%s\",%s\n", argv[i], json + 1);
        forge_ocr_string_free(json);
        size_t lines = 0;
        for (size_t p = 0; p < forge_ocr_result_pages(r); p++) lines += forge_ocr_page_lines(r, p);
        fprintf(stderr, "%s: %zu pages, %zu lines\n", argv[i], forge_ocr_result_pages(r), lines);
        forge_ocr_result_free(r);
    }
    forge_ocr_free(ocr);
    return status;
}
