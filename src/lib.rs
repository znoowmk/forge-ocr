//! ForgeOCR: OCR for scanned documents. The PP-OCRv5 detector and recognizer (ONNX) run on ONNX
//! Runtime; everything around them (decoding, resizing, box finding, cropping, batching, CTC,
//! reading order) is native Rust and parallel.
//!
//! ```no_run
//! let engine = forge_ocr::Engine::new(forge_ocr::Config::from_dir("models")?)?;
//! let pages = engine.read_pdf("scan.pdf".as_ref())?;
//! println!("{}", pages[0].text());
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod ctc;
pub mod db;
pub mod geom;
pub mod imaging;
pub mod pdf;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde::Serialize;

pub use db::DbParams;
pub use imaging::Rgb;

#[derive(Clone, Debug)]
pub struct Config {
    pub det_model: PathBuf,
    pub rec_model: PathBuf,
    /// Long side the detector sees (crops always come from the full page).
    pub det_side: usize,
    /// Crops per recognizer run.
    pub rec_batch: usize,
    /// Lines below this mean character probability are dropped.
    pub min_score: f32,
    pub db: DbParams,
    /// Where the networks run (see [`Device`]).
    pub device: Device,
    /// Pages prepared and recognised together (bigger = fuller GPU batches).
    pub chunk: usize,
    /// Check-digit rules for digit runs (see [`ctc::fix_check_digits`]); empty = off.
    pub digit_checks: Vec<ctc::DigitCheck>,
    /// Run the networks with TensorRT (FP16), keeping the built engines in this folder. The first
    /// run builds them (minutes); later runs load them. Falls back to CUDA, then the CPU.
    pub trt_cache: Option<PathBuf>,
}

impl Config {
    /// Models from a folder: `det.onnx` and `rec.onnx`, or the PP-OCRv5 file names
    /// (the FP16 recognizer is preferred when present).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Config> {
        let d = dir.as_ref();
        let pick = |names: &[&str]| names.iter().map(|n| d.join(n)).find(|p| p.exists());
        let det = pick(&["det.onnx", "ch_PP-OCRv5_det_mobile.onnx"]).ok_or_else(|| anyhow!("no detector model in {}", d.display()))?;
        let rec = pick(&["rec.onnx", "latin_PP-OCRv5_rec_mobile_fp16.onnx", "latin_PP-OCRv5_rec_mobile.onnx"])
            .ok_or_else(|| anyhow!("no recognizer model in {}", d.display()))?;
        Ok(Config { det_model: det, rec_model: rec, det_side: 1600, rec_batch: 32, min_score: 0.5, db: DbParams::default(), device: Device::Auto, chunk: 8, digit_checks: Vec::new(), trt_cache: None })
    }
}

/// One recognised line: text, mean character probability, corners in page pixels.
#[derive(Clone, Debug, Serialize)]
pub struct Line {
    pub text: String,
    pub score: f32,
    pub quad: [[f32; 2]; 4],
    /// Probability of each character, for callers that check fields (IDs, check digits).
    #[serde(skip)]
    pub char_probs: Vec<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Page {
    pub width: usize,
    pub height: usize,
    pub lines: Vec<Line>,
}

impl Page {
    /// The page's text with each printed row on one line: boxes whose vertical centres lie
    /// within half a text height of each other share a row, left to right (keeps table rows
    /// such as "NSS  NAME  DAYS" together).
    pub fn text(&self) -> String {
        let mut items: Vec<(f32, f32, f32, &str)> = self
            .lines
            .iter()
            .map(|l| {
                let ys = l.quad.iter().map(|p| p[1]);
                let (y0, y1) = (ys.clone().fold(f32::MAX, f32::min), ys.fold(f32::MIN, f32::max));
                let x0 = l.quad.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
                ((y0 + y1) / 2.0, (y1 - y0).max(1.0), x0, l.text.as_str())
            })
            .collect();
        items.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut rows: Vec<String> = Vec::new();
        let mut cur: Vec<(f32, &str)> = Vec::new();
        let (mut cy, mut ch) = (0.0f32, 0.0f32);
        for (y, h, x, t) in items {
            if !cur.is_empty() && (y - cy).abs() > 0.5 * h.min(ch) {
                cur.sort_by(|a, b| a.0.total_cmp(&b.0));
                rows.push(cur.iter().map(|c| c.1).collect::<Vec<_>>().join(" "));
                cur.clear();
            }
            if cur.is_empty() {
                (cy, ch) = (y, h);
            }
            cur.push((x, t));
        }
        if !cur.is_empty() {
            cur.sort_by(|a, b| a.0.total_cmp(&b.0));
            rows.push(cur.iter().map(|c| c.1).collect::<Vec<_>>().join(" "));
        }
        rows.join("\n")
    }
}

/// Where the time went, summed over pages.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Timings {
    pub pages: usize,
    pub crops: usize,
    pub decode_ms: f64,
    /// Pages whose scan was decoded as stored, and pages that had to be rendered.
    pub pages_jpeg: usize,
    pub pages_rendered: usize,
    pub det_prep_ms: f64,
    pub det_run_ms: f64,
    pub det_post_ms: f64,
    pub crop_ms: f64,
    pub rec_run_ms: f64,
    pub ctc_ms: f64,
    /// Digits changed by check-digit rules.
    pub digits_fixed: usize,
}

pub struct Engine {
    _env: ort::environment::Environment,
    det: Mutex<Session>,
    rec: Mutex<Session>,
    charset: Vec<String>,
    digit_class: Option<[usize; 10]>,
    cfg: Config,
    pub timings: Mutex<Timings>,
}

/// TensorRT input-shape ranges: (min, opt, max) for the input `x`.
type Profile = (&'static str, &'static str, &'static str);
const DET_PROFILE: Profile = ("x:1x3x32x32", "x:1x3x1600x1248", "x:1x3x1600x1600");
const REC_PROFILE: Profile = ("x:1x3x48x64", "x:32x3x48x640", "x:32x3x48x3072");

fn cuda() -> ort::ep::ExecutionProviderDispatch {
    // Heuristic cuDNN algorithm choice: exhaustive search re-benchmarks every new input shape,
    // and line crops come in many widths.
    ort::ep::CUDA::default().with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic).build()
}

/// Where the networks run. ONNX Runtime does the work; each choice is one of its execution
/// providers, and the ONNX Runtime library in use must include it (the official packages each
/// carry a different set: `onnxruntime-gpu` has CUDA and TensorRT, `onnxruntime-directml` has
/// DirectML, `onnxruntime-openvino` has OpenVINO, the ROCm build has ROCm and MIGraphX).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Device {
    /// The fastest one present, in this order: TensorRT (with `trt_cache`), CUDA (NVIDIA),
    /// ROCm then MIGraphX (AMD on Linux), DirectML (any GPU on Windows: NVIDIA, AMD, Intel),
    /// OpenVINO on the CPU (Intel; faster than plain ONNX Runtime there), CoreML (Apple), and
    /// the CPU at the end.
    #[default]
    Auto,
    TensorRt,
    Cuda,
    Rocm,
    MiGraphX,
    DirectMl,
    /// OpenVINO on a device: `GPU`, `NPU`, `CPU` or `AUTO`.
    OpenVino(String),
    CoreMl,
    Cpu,
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Device> {
        let l = s.to_ascii_lowercase();
        Ok(match l.as_str() {
            "auto" => Device::Auto,
            "tensorrt" | "trt" => Device::TensorRt,
            "cuda" | "nvidia" => Device::Cuda,
            "rocm" | "amd" => Device::Rocm,
            "migraphx" => Device::MiGraphX,
            "directml" | "dml" => Device::DirectMl,
            "openvino" | "intel" => Device::OpenVino("GPU".into()),
            "coreml" | "apple" => Device::CoreMl,
            "cpu" => Device::Cpu,
            _ => match l.strip_prefix("openvino:") {
                Some(d) => Device::OpenVino(d.to_ascii_uppercase()),
                None => return Err(anyhow!("unknown device `{s}`: auto, tensorrt, cuda, rocm, migraphx, directml, openvino[:GPU|NPU|CPU], coreml or cpu")),
            },
        })
    }
}

fn tensorrt(cache: &Path, profile: Profile) -> Result<ort::ep::ExecutionProviderDispatch> {
    std::fs::create_dir_all(cache)?;
    let dir = cache.to_string_lossy().into_owned();
    Ok(ort::ep::TensorRT::default()
        .with_fp16(true)
        .with_engine_cache(true)
        .with_engine_cache_path(&dir)
        .with_timing_cache(true)
        .with_timing_cache_path(&dir)
        .with_profile_min_shapes(profile.0)
        .with_profile_opt_shapes(profile.1)
        .with_profile_max_shapes(profile.2)
        .build())
}

/// The execution providers to try for `device`, best first.
fn providers(cfg: &Config, profile: Profile) -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
    let openvino = |dev: &str| ort::ep::OpenVINO::default().with_device_type(dev).with_dynamic_shapes(true).build();
    Ok(match &cfg.device {
        Device::Cpu => Vec::new(),
        Device::TensorRt => {
            let cache = cfg.trt_cache.clone().unwrap_or_else(|| std::env::temp_dir().join("forge-ocr-trt"));
            vec![tensorrt(&cache, profile)?.error_on_failure()]
        }
        Device::Cuda => vec![cuda().error_on_failure()],
        Device::Rocm => vec![ort::ep::ROCm::default().build().error_on_failure()],
        Device::MiGraphX => vec![ort::ep::MIGraphX::default().build().error_on_failure()],
        Device::DirectMl => vec![ort::ep::DirectML::default().build().error_on_failure()],
        Device::OpenVino(d) => vec![openvino(d).error_on_failure()],
        Device::CoreMl => vec![ort::ep::CoreML::default().build().error_on_failure()],
        Device::Auto => {
            let mut eps = Vec::new();
            if let Some(cache) = &cfg.trt_cache {
                eps.push(tensorrt(cache, profile)?);
            }
            eps.push(cuda());
            eps.push(ort::ep::ROCm::default().build());
            eps.push(ort::ep::MIGraphX::default().build());
            if cfg!(windows) {
                eps.push(ort::ep::DirectML::default().build());
            }
            // OpenVINO on the CPU: on an i7-12800HX it reads 1.8 pages/s against 0.8 for plain
            // ONNX Runtime and 0.4 on the UHD 770 integrated GPU (shapes vary per line, and
            // each new shape recompiles there). An Intel Arc card is worth `openvino:GPU`.
            eps.push(openvino("CPU"));
            if cfg!(target_vendor = "apple") {
                eps.push(ort::ep::CoreML::default().build());
            }
            // Unavailable ones are skipped quietly.
            eps.into_iter().map(|e| e.fail_silently()).collect()
        }
    })
}

/// Whether the networks will run on the CPU: asked for, or `Auto` with no accelerator in the
/// ONNX Runtime library at hand.
fn on_cpu(device: &Device) -> bool {
    use ort::ep::ExecutionProvider;
    match device {
        Device::Cpu => true,
        Device::Auto => {
            let has = |r: ort::Result<bool>| r.unwrap_or(false);
            !(has(ort::ep::CUDA::is_available())
                || has(ort::ep::ROCm::is_available())
                || has(ort::ep::MIGraphX::is_available())
                || has(ort::ep::DirectML::is_available())
                || has(ort::ep::OpenVINO::is_available())
                || has(ort::ep::CoreML::is_available()))
        }
        _ => false,
    }
}

/// Physical cores, roughly: hyper-threads slow ONNX Runtime's CPU kernels down.
fn cpu_threads() -> usize {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    (logical / 2).max(1)
}

fn session(env: &ort::environment::Environment, path: &Path, cfg: &Config, profile: Profile) -> Result<Session> {
    let mut b = Session::builder(env).map_err(|e| anyhow!("{e}"))?;
    if on_cpu(&cfg.device) {
        b = b.with_intra_threads(cpu_threads()).map_err(|e| anyhow!("{e}"))?;
    }
    if matches!(cfg.device, Device::DirectMl | Device::Auto) && cfg!(windows) {
        // DirectML needs these two off.
        b = b.with_memory_pattern(false).map_err(|e| anyhow!("{e}"))?.with_parallel_execution(false).map_err(|e| anyhow!("{e}"))?;
    }
    let eps = providers(cfg, profile)?;
    if !eps.is_empty() {
        b = b.with_execution_providers(eps).map_err(|e| anyhow!("{e}"))?;
    }
    b.commit_from_file(path).map_err(|e| anyhow!("{e}")).with_context(|| format!("loading {}", path.display()))
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

impl Engine {
    pub fn new(cfg: Config) -> Result<Engine> {
        // One ONNX Runtime environment per process. Each session tries the providers of
        // `cfg.device` in order; ONNX Runtime falls back to the CPU for anything they can't run.
        let mut cfg = cfg;
        let cpu = on_cpu(&cfg.device);
        if cpu {
            // On a CPU: FP16 has no fast kernels (FP32 is faster), and small batches stay in
            // cache (8 lines run about 2.5x faster per line than 32).
            let name = cfg.rec_model.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
            if let Some(fp32) = name.strip_suffix("_fp16.onnx").map(|stem| cfg.rec_model.with_file_name(format!("{stem}.onnx"))).filter(|p| p.exists()) {
                cfg.rec_model = fp32;
            }
            cfg.rec_batch = cfg.rec_batch.min(8);
        }
        let env = ort::init().build().map_err(|e| anyhow!("{e}"))?;
        let det = session(&env, &cfg.det_model, &cfg, DET_PROFILE)?;
        let rec = session(&env, &cfg.rec_model, &cfg, REC_PROFILE)?;
        let chars = rec.metadata().map_err(|e| anyhow!("{e}"))?.custom("character").ok_or_else(|| anyhow!("the recognizer has no `character` metadata"))?;
        let charset = ctc::charset_from_metadata(&chars);
        let digit_class = ctc::digit_classes(&charset);
        Ok(Engine { _env: env, det: Mutex::new(det), rec: Mutex::new(rec), charset, digit_class, cfg, timings: Mutex::new(Timings::default()) })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Reads every page of a PDF (embedded scans are decoded as they are; other pages are
    /// rendered with `pdftoppm`).
    pub fn read_pdf(&self, path: &Path) -> Result<Vec<Page>> {
        let t = Instant::now();
        let (imgs, jpeg, rendered) = pdf::page_images(path)?;
        let mut tm = self.timings.lock().unwrap();
        tm.decode_ms += ms(t);
        tm.pages_jpeg += jpeg;
        tm.pages_rendered += rendered;
        drop(tm);
        self.read_images(imgs)
    }

    /// Reads page images, `chunk` at a time.
    pub fn read_images(&self, imgs: Vec<Rgb>) -> Result<Vec<Page>> {
        let mut out = Vec::with_capacity(imgs.len());
        let mut it = imgs.into_iter().peekable();
        while it.peek().is_some() {
            let chunk: Vec<Rgb> = it.by_ref().take(self.cfg.chunk.max(1)).collect();
            out.extend(self.read_chunk(chunk)?);
        }
        Ok(out)
    }

    fn read_chunk(&self, imgs: Vec<Rgb>) -> Result<Vec<Page>> {
        let cfg = &self.cfg;
        // 1. Detector input for every page, in parallel.
        let t = Instant::now();
        let inputs: Vec<(usize, usize, Vec<f32>)> = imgs
            .par_iter()
            .map(|img| {
                let (dw, dh) = imaging::det_size(img.w, img.h, cfg.det_side);
                (dw, dh, imaging::det_tensor(img, dw, dh))
            })
            .collect();
        let det_prep = ms(t);
        // 2. Detector, one page at a time (pages differ in size).
        let t = Instant::now();
        let mut maps = Vec::with_capacity(inputs.len());
        {
            let mut det = self.det.lock().unwrap();
            for (dw, dh, data) in inputs {
                let x = Tensor::from_array(([1usize, 3, dh, dw], data))?;
                let outputs = det.run(ort::inputs![x])?;
                let (_, p) = outputs[0].try_extract_tensor::<f32>()?;
                maps.push((dw, dh, p.to_vec()));
            }
        }
        let det_run = ms(t);
        // 3. Boxes per page, in parallel.
        let t = Instant::now();
        let boxes: Vec<Vec<db::TextBox>> = maps.par_iter().zip(imgs.par_iter()).map(|((dw, dh, p), img)| db::boxes(p, *dw, *dh, img.w, img.h, cfg.db)).collect();
        let det_post = ms(t);
        // 4. Crops of every box of every page, in parallel.
        let t = Instant::now();
        let max_w = 48 * 64; // a 64:1 line is already very long
        let mut crops: Vec<(usize, usize, imaging::Crop)> = boxes
            .par_iter()
            .enumerate()
            .flat_map_iter(|(pi, bs)| bs.iter().enumerate().map(move |(bi, b)| (pi, bi, b)))
            .map(|(pi, bi, b)| (pi, bi, imaging::crop(&imgs[pi], &b.quad, max_w)))
            .collect();
        // Sorted by shape so each batch pads little.
        crops.sort_by(|a, b| a.2.ratio.total_cmp(&b.2.ratio));
        let crop_ms = ms(t);
        // 5. Recognizer in batches; 6. CTC in parallel.
        let mut results: Vec<Vec<Option<ctc::Decoded>>> = boxes.iter().map(|b| vec![None; b.len()]).collect();
        let (mut rec_run, mut ctc_ms, mut digits_fixed) = (0.0, 0.0, 0);
        for batch in crops.chunks(cfg.rec_batch.max(1)) {
            let max_ratio = batch.iter().map(|c| c.2.ratio).fold(320.0 / 48.0, f32::max);
            // Widths rounded up to 64 so batches repeat a few tensor shapes.
            let bw = ((48.0 * max_ratio) as usize).min(max_w).max(batch.iter().map(|c| c.2.w).max().unwrap_or(1)).div_ceil(64) * 64;
            let n = batch.len();
            let mut data = vec![0f32; n * 3 * 48 * bw];
            data.par_chunks_mut(3 * 48 * bw).zip(batch.par_iter()).for_each(|(dst, (_, _, c))| {
                for ch in 0..3 {
                    for y in 0..48 {
                        let s = &c.data[ch * 48 * c.w + y * c.w..ch * 48 * c.w + (y + 1) * c.w];
                        dst[ch * 48 * bw + y * bw..ch * 48 * bw + y * bw + c.w].copy_from_slice(s);
                    }
                }
            });
            let t = Instant::now();
            let (shape, probs) = {
                let mut rec = self.rec.lock().unwrap();
                let x = Tensor::from_array(([n, 3usize, 48, bw], data))?;
                let outputs = rec.run(ort::inputs![x])?;
                let (shape, p) = outputs[0].try_extract_tensor::<f32>()?;
                (shape.to_vec(), p.to_vec())
            };
            rec_run += ms(t);
            let t = Instant::now();
            let (tt, cc) = (shape[1] as usize, shape[2] as usize);
            let decoded: Vec<(ctc::Decoded, usize)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let p = &probs[i * tt * cc..(i + 1) * tt * cc];
                    let mut d = ctc::greedy(p, tt, cc, &self.charset);
                    let mut fixed = 0;
                    if let Some(classes) = &self.digit_class {
                        for rule in &cfg.digit_checks {
                            fixed += ctc::fix_check_digits(&mut d, p, cc, classes, *rule);
                        }
                    }
                    (d, fixed)
                })
                .collect();
            for ((pi, bi, _), (d, fixed)) in batch.iter().zip(decoded) {
                results[*pi][*bi] = Some(d);
                digits_fixed += fixed;
            }
            ctc_ms += ms(t);
        }
        // 7. Pages.
        let pages: Vec<Page> = imgs
            .iter()
            .zip(boxes.iter().zip(results))
            .map(|(img, (bs, rs))| Page {
                width: img.w,
                height: img.h,
                lines: bs
                    .iter()
                    .zip(rs)
                    .filter_map(|(b, r)| r.filter(|d| d.score >= cfg.min_score && !d.text.trim().is_empty()).map(|d| Line { text: d.text, score: d.score, quad: b.quad, char_probs: d.char_probs }))
                    .collect(),
            })
            .collect();
        let mut tm = self.timings.lock().unwrap();
        tm.pages += pages.len();
        tm.crops += crops.len();
        tm.det_prep_ms += det_prep;
        tm.det_run_ms += det_run;
        tm.det_post_ms += det_post;
        tm.crop_ms += crop_ms;
        tm.rec_run_ms += rec_run;
        tm.ctc_ms += ctc_ms;
        tm.digits_fixed += digits_fixed;
        Ok(pages)
    }
}
