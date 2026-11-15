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
    while let Some(a) = args.next() {
        match a.as_str() {
            "--models" => models = args.next().map(PathBuf::from).unwrap_or(models),
            "--cpu" => cpu = true,
            "--json" => json = true,
            "--det-side" => det_side = args.next().and_then(|v| v.parse().ok()),
            "--batch" => batch = args.next().and_then(|v| v.parse().ok()),
            "-h" | "--help" => {
                println!("forge-ocr [--models DIR] [--cpu] [--json] [--det-side N] [--batch N] FILE...");
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
    if let Some(s) = det_side {
        cfg.det_side = s;
    }
    if let Some(b) = batch {
        cfg.rec_batch = b;
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
        if json {
            println!("{}", serde_json::json!({"file": f, "pages": pages}));
        } else {
            let text: Vec<String> = pages.iter().map(|p| p.text()).collect();
            print!("{}", text.join("\x0c"));
            if files.len() > 1 {
                print!("\x0c");
            }
        }
    }
    let tm = engine.timings.lock().unwrap().clone();
    let secs = t.elapsed().as_secs_f64();
    let per = |v: f64| v / tm.pages.max(1) as f64;
    eprintln!(
        "forge-ocr: {} pages ({} scans read as stored, {} rendered), {} lines in {:.1} s ({:.1} pages/s; models {:.1} s). Per page: decode {:.0} ms, det prep {:.0}, det {:.0}, boxes {:.0}, crops {:.0}, rec {:.0}, ctc {:.0}",
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
        per(tm.ctc_ms)
    );
    Ok(())
}
