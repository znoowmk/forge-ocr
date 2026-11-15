//! `forge-ocr`: read scanned PDFs and images.
//!
//! ```text
//! forge-ocr [--models DIR] [--cpu] [--json] [--det-side N] [--batch N] FILE...
//! ```
//! Text goes to stdout, pages separated by form feeds (\f), one printed row per line. With
//! `--json`, one JSON object per file with every line's text, score and corners. Timings go to
//! stderr.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use forge_ocr::{Config, Engine, Rgb};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut models = std::env::var_os("FORGE_OCR_MODELS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("models"));
    let (mut cpu, mut json, mut det_side, mut batch) = (false, false, None, None);
    let mut files = Vec::new();
    let mut checks: Vec<String> = Vec::new();
    let mut out_dir: Option<PathBuf> = None;
    let mut trt: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--models" => models = args.next().map(PathBuf::from).unwrap_or(models),
            "--cpu" => cpu = true,
            "--trt" => trt = args.next().map(PathBuf::from),
            "--json" => json = true,
            // One <stem>.txt (or .json) per input file in DIR, instead of stdout.
            "--out-dir" => out_dir = args.next().map(PathBuf::from),
            "--det-side" => det_side = args.next().and_then(|v| v.parse().ok()),
            "--batch" => batch = args.next().and_then(|v| v.parse().ok()),
            // Check-digit rules: `luhn:11` (Mexico's NSS), `luhn:16` (cards)... repeatable.
            "--check" => checks.push(args.next().unwrap_or_default()),
            "-h" | "--help" => {
                println!("forge-ocr [--models DIR] [--cpu] [--json] [--det-side N] [--batch N] [--check luhn:LEN]... [--out-dir DIR] [--trt CACHE_DIR] FILE...");
                return Ok(());
            }
            "--version" => {
                println!("forge-ocr {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            _ => files.push(PathBuf::from(a)),
        }
    }
    if files.is_empty() {
        bail!("no input files (forge-ocr --help)");
    }
    let mut cfg = Config::from_dir(&models)?;
    cfg.gpu = !cpu;
    cfg.trt_cache = trt;
    if let Some(s) = det_side {
        cfg.det_side = s;
    }
    if let Some(b) = batch {
        cfg.rec_batch = b;
    }
    for c in &checks {
        match c.split_once(':') {
            Some(("luhn", n)) => cfg.digit_checks.push(forge_ocr::ctc::DigitCheck { len: n.parse()?, check: forge_ocr::ctc::luhn_ok }),
            _ => bail!("--check takes luhn:LENGTH, e.g. luhn:11"),
        }
    }
    let t0 = Instant::now();
    let engine = Engine::new(cfg)?;
    let load = t0.elapsed().as_secs_f64();
    let t = Instant::now();
    for f in &files {
        let pages = if f.extension().is_some_and(|e| e.eq_ignore_ascii_case("pdf")) {
            engine.read_pdf(f)?
        } else {
            engine.read_images(vec![Rgb::from_image(image::open(f)?).capped(forge_ocr::pdf::MAX_SIDE)])?
        };
        let body = if json {
            serde_json::json!({"file": f, "pages": pages}).to_string()
        } else {
            pages.iter().map(|p| p.text()).collect::<Vec<_>>().join("\x0c")
        };
        match &out_dir {
            Some(dir) => {
                let stem = f.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let path = dir.join(format!("{stem}.{}", if json { "json" } else { "txt" }));
                let part = path.with_extension("part");
                std::fs::write(&part, body)?;
                std::fs::rename(part, path)?;
            }
            None => {
                print!("{body}");
                if json || files.len() > 1 {
                    print!("{}", if json { "\n" } else { "\x0c" });
                }
            }
        }
    }
    let tm = engine.timings.lock().unwrap().clone();
    let secs = t.elapsed().as_secs_f64();
    let per = |v: f64| v / tm.pages.max(1) as f64;
    eprintln!(
        "forge-ocr: {} pages ({} scans read as stored, {} rendered), {} lines in {:.1} s ({:.1} pages/s; models {:.1} s). Per page: decode {:.0} ms, det prep {:.0}, det {:.0}, boxes {:.0}, crops {:.0}, rec {:.0}, ctc {:.0}. Digits fixed by check rules: {}",
        tm.pages,
        tm.pages_jpeg,
        tm.pages_rendered,
        tm.crops,
        secs,
        tm.pages as f64 / secs,
        load,
        per(tm.decode_ms),
        per(tm.det_prep_ms),
        per(tm.det_run_ms),
        per(tm.det_post_ms),
        per(tm.crop_ms),
        per(tm.rec_run_ms),
        per(tm.ctc_ms),
        tm.digits_fixed
    );
    Ok(())
}
