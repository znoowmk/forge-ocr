//! C API for ForgeOCR. The header is `include/forge_ocr.h`; see `examples/read.c`.
//!
//! Rules for callers:
//! - Functions that can fail return NULL (or a negative number); `forge_ocr_last_error()` then
//!   gives the message for the calling thread.
//! - Every `ForgeOcr*`, `ForgeOcrResult*` and `char*` returned to you is freed with its own
//!   `*_free` function. `const char*` returned by accessors belong to the result and live until
//!   `forge_ocr_result_free`.
//! - One engine can be shared between threads; reads on it run one network at a time.

// Every extern fn has the same contract (valid pointers from this library, NUL-terminated UTF-8
// strings); it is written once above and in forge_ocr.h instead of on each function.
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;

use forge_ocr::{Config, Engine, Page, Rgb};

pub struct ForgeOcr {
    engine: Engine,
}

pub struct ForgeOcrResult {
    pages: Vec<Page>,
    /// The text of each line as a C string, built once so accessors can hand out pointers.
    texts: Vec<Vec<CString>>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl std::fmt::Display) {
    let s = CString::new(msg.to_string().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = s);
}

/// Runs `f`, turning errors and panics into NULL plus a message for `forge_ocr_last_error`.
fn guard<T>(f: impl FnOnce() -> anyhow::Result<*mut T>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_error(format!("{e:#}"));
            ptr::null_mut()
        }
        Err(_) => {
            set_error("internal error (panic) in forge-ocr");
            ptr::null_mut()
        }
    }
}

unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> anyhow::Result<&'a str> {
    if p.is_null() {
        anyhow::bail!("{what} is NULL");
    }
    unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| anyhow::anyhow!("{what} is not UTF-8"))
}

fn into_result(pages: Vec<Page>) -> *mut ForgeOcrResult {
    let texts = pages
        .iter()
        .map(|p| p.lines.iter().map(|l| CString::new(l.text.replace('\0', " ")).unwrap_or_default()).collect())
        .collect();
    Box::into_raw(Box::new(ForgeOcrResult { pages, texts }))
}

fn line<'a>(r: *const ForgeOcrResult, page: usize, line: usize) -> Option<(&'a ForgeOcrResult, usize, usize)> {
    let r = unsafe { r.as_ref()? };
    (page < r.pages.len() && line < r.pages[page].lines.len()).then_some((r, page, line))
}

/// Version of the library, e.g. "0.1.0". Static; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn forge_ocr_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Message of the last failed call on this thread ("" if none). Valid until the next failing call.
#[unsafe(no_mangle)]
pub extern "C" fn forge_ocr_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Loads the models from `models_dir` (det.onnx/rec.onnx or the PP-OCRv5 names). `device` may be
/// NULL (auto) or "auto", "tensorrt", "cuda", "rocm", "migraphx", "directml", "openvino[:GPU|NPU|CPU]",
/// "coreml", "cpu". ONNX Runtime is found through `ORT_DYLIB_PATH`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_new(models_dir: *const c_char, device: *const c_char) -> *mut ForgeOcr {
    guard(|| {
        let dir = unsafe { str_arg(models_dir, "models_dir") }?;
        let mut cfg = Config::from_dir(dir)?;
        if !device.is_null() {
            cfg.device = unsafe { str_arg(device, "device") }?.parse()?;
        }
        Ok(Box::into_raw(Box::new(ForgeOcr { engine: Engine::new(cfg)? })))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_free(ocr: *mut ForgeOcr) {
    if !ocr.is_null() {
        drop(unsafe { Box::from_raw(ocr) });
    }
}

/// Reads a PDF (every page) or an image file (one page).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_read_file(ocr: *const ForgeOcr, path: *const c_char) -> *mut ForgeOcrResult {
    guard(|| {
        let ocr = unsafe { ocr.as_ref() }.ok_or_else(|| anyhow::anyhow!("ocr is NULL"))?;
        let path = unsafe { str_arg(path, "path") }?;
        Ok(into_result(ocr.engine.read_file(Path::new(path))?))
    })
}

/// Reads one page from 8-bit pixels. `channels` is 3 (RGB) or 4 (RGBA, alpha ignored); `stride` is
/// the bytes from one row to the next (0 = width * channels). The pixels are copied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_read_pixels(
    ocr: *const ForgeOcr,
    pixels: *const u8,
    width: usize,
    height: usize,
    channels: usize,
    stride: usize,
) -> *mut ForgeOcrResult {
    guard(|| {
        let ocr = unsafe { ocr.as_ref() }.ok_or_else(|| anyhow::anyhow!("ocr is NULL"))?;
        anyhow::ensure!(!pixels.is_null(), "pixels is NULL");
        anyhow::ensure!(width > 0 && height > 0, "empty image ({width}x{height})");
        anyhow::ensure!(channels == 3 || channels == 4, "channels must be 3 or 4, not {channels}");
        let stride = if stride == 0 { width * channels } else { stride };
        anyhow::ensure!(stride >= width * channels, "stride {stride} is shorter than a row");
        let src = unsafe { std::slice::from_raw_parts(pixels, stride * (height - 1) + width * channels) };
        let img = Rgb::from_rows(src, width, height, channels, stride).capped(forge_ocr::pdf::MAX_SIDE);
        Ok(into_result(ocr.engine.read_images(vec![img])?))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_result_free(r: *mut ForgeOcrResult) {
    if !r.is_null() {
        drop(unsafe { Box::from_raw(r) });
    }
}

/// Number of pages (0 if `r` is NULL).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_result_pages(r: *const ForgeOcrResult) -> usize {
    unsafe { r.as_ref() }.map_or(0, |r| r.pages.len())
}

/// Page size in pixels as read (after scaling very large images down). Returns -1 if out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_page_size(r: *const ForgeOcrResult, page: usize, width: *mut usize, height: *mut usize) -> i32 {
    let Some(p) = (unsafe { r.as_ref() }).and_then(|r| r.pages.get(page)) else { return -1 };
    unsafe {
        if let Some(w) = width.as_mut() {
            *w = p.width;
        }
        if let Some(h) = height.as_mut() {
            *h = p.height;
        }
    }
    0
}

/// Number of lines on a page (0 if out of range).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_page_lines(r: *const ForgeOcrResult, page: usize) -> usize {
    unsafe { r.as_ref() }.and_then(|r| r.pages.get(page)).map_or(0, |p| p.lines.len())
}

/// Text of a line (UTF-8), owned by the result. NULL if out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_line_text(r: *const ForgeOcrResult, page: usize, line_no: usize) -> *const c_char {
    line(r, page, line_no).map_or(ptr::null(), |(r, p, l)| r.texts[p][l].as_ptr())
}

/// Mean character probability of a line (0..1), or -1 if out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_line_score(r: *const ForgeOcrResult, page: usize, line_no: usize) -> f32 {
    line(r, page, line_no).map_or(-1.0, |(r, p, l)| r.pages[p].lines[l].score)
}

/// Writes the line's four corners (x0,y0,x1,y1,x2,y2,x3,y3; clockwise from top-left) into `out[8]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_line_quad(r: *const ForgeOcrResult, page: usize, line_no: usize, out: *mut f32) -> i32 {
    let Some((r, p, l)) = line(r, page, line_no) else { return -1 };
    if out.is_null() {
        return -1;
    }
    let q = r.pages[p].lines[l].quad;
    let out = unsafe { std::slice::from_raw_parts_mut(out, 8) };
    for (i, c) in q.iter().enumerate() {
        out[2 * i] = c[0];
        out[2 * i + 1] = c[1];
    }
    0
}

fn owned(s: String) -> *mut c_char {
    CString::new(s.replace('\0', " ")).map_or(ptr::null_mut(), CString::into_raw)
}

/// The page's text, one printed row per line. Free with `forge_ocr_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_page_text(r: *const ForgeOcrResult, page: usize) -> *mut c_char {
    match unsafe { r.as_ref() }.and_then(|r| r.pages.get(page)) {
        Some(p) => owned(p.text()),
        None => {
            set_error(format!("no page {page}"));
            ptr::null_mut()
        }
    }
}

/// All pages as JSON, the same shape as `forge-ocr --json` ("pages": [...]). Free with `forge_ocr_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_result_json(r: *const ForgeOcrResult) -> *mut c_char {
    match unsafe { r.as_ref() } {
        Some(r) => owned(serde_json::json!({ "pages": r.pages }).to_string()),
        None => {
            set_error("result is NULL");
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn forge_ocr_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}
