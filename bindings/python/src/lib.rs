//! Python bindings for ForgeOCR.
//!
//! ```python
//! import forge_ocr, numpy as np
//! ocr = forge_ocr.Engine("models", device="cuda")
//! for page in ocr.read_file("scan.pdf"):
//!     print(page.text())
//! page = ocr.read_image(np.asarray(pil_image))   # HxWx3 or HxWx4 uint8
//! ```
//! The GIL is released while the networks run, so other Python threads keep going.

use std::path::PathBuf;

use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

/// One recognised line.
#[pyclass(frozen, module = "forge_ocr")]
struct Line {
    #[pyo3(get)]
    text: String,
    /// Mean character probability (0..1).
    #[pyo3(get)]
    score: f32,
    /// Four (x, y) corners in page pixels, clockwise from the top-left.
    #[pyo3(get)]
    quad: [(f32, f32); 4],
}

#[pymethods]
impl Line {
    fn __repr__(&self) -> String {
        format!("Line({:?}, score={:.3})", self.text, self.score)
    }
}

/// One page: its size in pixels (as read) and its lines.
#[pyclass(frozen, module = "forge_ocr")]
struct Page {
    inner: forge_ocr::Page,
}

#[pymethods]
impl Page {
    #[getter]
    fn width(&self) -> usize {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> usize {
        self.inner.height
    }
    #[getter]
    fn lines(&self) -> Vec<Line> {
        self.inner.lines.iter().map(|l| Line { text: l.text.clone(), score: l.score, quad: l.quad.map(|p| (p[0], p[1])) }).collect()
    }
    /// The page's text, one printed row per line (table rows stay together).
    fn text(&self) -> String {
        self.inner.text()
    }
    /// The page as JSON, the same shape as one entry of `forge-ocr --json`'s "pages".
    fn json(&self) -> String {
        serde_json::to_value(&self.inner).map(|v| v.to_string()).unwrap_or_default()
    }
    fn __len__(&self) -> usize {
        self.inner.lines.len()
    }
    fn __repr__(&self) -> String {
        format!("Page({}x{}, {} lines)", self.inner.width, self.inner.height, self.inner.lines.len())
    }
}

fn pages(v: Vec<forge_ocr::Page>) -> Vec<Page> {
    v.into_iter().map(|inner| Page { inner }).collect()
}

/// Reads an HxWx3 (RGB) or HxWx4 (RGBA) uint8 array. Any strides are accepted; the pixels are
/// packed once into the buffer the pipeline works on (no Python objects per pixel).
fn rgb(a: &PyReadonlyArray3<'_, u8>) -> PyResult<forge_ocr::Rgb> {
    let s = a.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    if c != 3 && c != 4 {
        return Err(PyValueError::new_err(format!("expected HxWx3 or HxWx4, got {h}x{w}x{c}")));
    }
    if h == 0 || w == 0 {
        return Err(PyValueError::new_err("empty image"));
    }
    let img = match a.as_slice() {
        Ok(px) => forge_ocr::Rgb::from_rows(px, w, h, c, w * c),
        // Not C-contiguous (a crop, a transposed view): copy element by element.
        Err(_) => {
            let v = a.as_array();
            let mut px = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for x in 0..w {
                    px.extend_from_slice(&[v[[y, x, 0]], v[[y, x, 1]], v[[y, x, 2]]]);
                }
            }
            forge_ocr::Rgb { w, h, px }
        }
    };
    Ok(img.capped(forge_ocr::pdf::MAX_SIDE))
}

/// Loads the models once; reuse the engine for every file.
#[pyclass(frozen, module = "forge_ocr")]
struct Engine {
    inner: forge_ocr::Engine,
}

#[pymethods]
impl Engine {
    /// `models`: folder with det.onnx/rec.onnx or the PP-OCRv5 names. `device`: "auto", "tensorrt",
    /// "cuda", "rocm", "migraphx", "directml", "openvino[:GPU|NPU|CPU]", "coreml" or "cpu".
    /// `trt_cache`: folder for TensorRT engines (first run builds them). `checks`: digit rules such
    /// as ["luhn:11"] (Mexico's NSS). ONNX Runtime is found through ORT_DYLIB_PATH.
    #[new]
    #[pyo3(signature = (models, device=None, det_side=None, batch=None, trt_cache=None, checks=None))]
    fn new(
        py: Python<'_>,
        models: PathBuf,
        device: Option<&str>,
        det_side: Option<usize>,
        batch: Option<usize>,
        trt_cache: Option<PathBuf>,
        checks: Option<Vec<String>>,
    ) -> PyResult<Engine> {
        let mut cfg = forge_ocr::Config::from_dir(&models).map_err(err)?;
        if let Some(d) = device {
            cfg.device = d.parse().map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;
        }
        if let Some(s) = det_side {
            cfg.det_side = s;
        }
        if let Some(b) = batch {
            cfg.rec_batch = b;
        }
        cfg.trt_cache = trt_cache;
        for c in checks.unwrap_or_default() {
            match c.split_once(':').map(|(k, n)| (k, n.parse::<usize>())) {
                Some(("luhn", Ok(len))) => cfg.digit_checks.push(forge_ocr::ctc::DigitCheck { len, check: forge_ocr::ctc::luhn_ok }),
                _ => return Err(PyValueError::new_err(format!("checks take \"luhn:LENGTH\", not {c:?}"))),
            }
        }
        let inner = py.detach(|| forge_ocr::Engine::new(cfg)).map_err(err)?;
        Ok(Engine { inner })
    }

    /// Reads a PDF (every page) or an image file (one page).
    fn read_file(&self, py: Python<'_>, path: PathBuf) -> PyResult<Vec<Page>> {
        py.detach(|| self.inner.read_file(&path)).map(pages).map_err(err)
    }

    /// Reads one page from an HxWx3 or HxWx4 uint8 array (e.g. `np.asarray(pil_image)`).
    fn read_image(&self, py: Python<'_>, image: PyReadonlyArray3<'_, u8>) -> PyResult<Page> {
        let img = rgb(&image)?;
        let mut v = py.detach(|| self.inner.read_images(vec![img])).map_err(err)?;
        Ok(Page { inner: v.remove(0) })
    }

    /// Reads several pages together (fuller GPU batches than one at a time).
    fn read_images(&self, py: Python<'_>, images: Vec<PyReadonlyArray3<'_, u8>>) -> PyResult<Vec<Page>> {
        let imgs = images.iter().map(rgb).collect::<PyResult<Vec<_>>>()?;
        py.detach(|| self.inner.read_images(imgs)).map(pages).map_err(err)
    }

    /// The device the engine was asked to use.
    #[getter]
    fn device(&self) -> String {
        format!("{:?}", self.inner.config().device).to_lowercase()
    }
}

#[pymodule(name = "forge_ocr")]
fn py_forge_ocr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Engine>()?;
    m.add_class::<Page>()?;
    m.add_class::<Line>()?;
    Ok(())
}
