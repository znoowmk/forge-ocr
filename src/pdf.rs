//! Page images from a PDF. A scanned page is usually one JPEG covering the page: it is decoded as
//! stored, at its own resolution (no re-render). Any other page is rendered with `pdftoppm`.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use lopdf::{Document, ObjectId};
use rayon::prelude::*;

use crate::imaging::Rgb;

/// Long side kept for page images; bigger scans are halved (300 ppi A4 is ~3500 px).
pub const MAX_SIDE: usize = 3600;

enum Source {
    Jpeg(Vec<u8>),
    Render(u32),
}

/// The page images, and how many came from an embedded JPEG and how many were rendered.
pub fn page_images(path: &Path) -> Result<(Vec<Rgb>, usize, usize)> {
    let doc = Document::load(path).with_context(|| format!("reading {}", path.display()))?;
    let pages: Vec<(u32, ObjectId)> = doc.get_pages().into_iter().collect();
    let sources: Vec<Source> = pages.iter().map(|(no, id)| embedded_jpeg(&doc, *id).map(Source::Jpeg).unwrap_or(Source::Render(*no))).collect();
    let jpeg = sources.iter().filter(|s| matches!(s, Source::Jpeg(_))).count();
    let rendered = sources.len() - jpeg;
    // A page that can't be read becomes a blank page with a warning, not a failed document.
    let imgs = sources
        .into_par_iter()
        .enumerate()
        .map(|(i, s)| {
            let r = match s {
                Source::Jpeg(bytes) => decode_jpeg(&bytes),
                Source::Render(no) => render(path, no),
            };
            match r {
                Ok(img) => img.capped(MAX_SIDE),
                Err(e) => {
                    eprintln!("forge-ocr: {} page {}: {e:#}", path.display(), i + 1);
                    Rgb { w: 32, h: 32, px: vec![255; 32 * 32 * 3] }
                }
            }
        })
        .collect();
    Ok((imgs, jpeg, rendered))
}

/// The page's JPEG when the page is exactly one DCT-encoded image, unrotated, with the page's
/// shape (so it covers the page).
fn embedded_jpeg(doc: &Document, page: ObjectId) -> Option<Vec<u8>> {
    let dict = doc.get_dictionary(page).ok()?;
    let rotate = dict.get(b"Rotate").ok().and_then(|o| o.as_i64().ok()).unwrap_or(0);
    if rotate % 360 != 0 {
        return None;
    }
    let images = doc.get_page_images(page).ok()?;
    if images.len() != 1 {
        return None;
    }
    let img = &images[0];
    let filters = img.filters.as_ref()?;
    if filters.len() != 1 || filters[0] != "DCTDecode" {
        return None;
    }
    // Same shape as the page (within 8%), so the image is the page.
    let media = media_box(doc, page)?;
    let (pw, ph) = ((media[2] - media[0]).abs() as f64, (media[3] - media[1]).abs() as f64);
    let (iw, ih) = (img.width as f64, img.height as f64);
    if pw <= 0.0 || ph <= 0.0 || ((iw / ih) / (pw / ph) - 1.0).abs() > 0.08 {
        return None;
    }
    Some(img.content.to_vec())
}

fn decode_jpeg(bytes: &[u8]) -> Result<Rgb> {
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg).context("decoding the page JPEG")?;
    Ok(Rgb::from_image(img))
}

/// Renders one page at 300 dpi with `pdftoppm` into a private temporary folder.
fn render(path: &Path, page: u32) -> Result<Rgb> {
    let dir = std::env::temp_dir().join(format!(
        "forge-ocr-{}-{page}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    ));
    std::fs::create_dir(&dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
    }
    let result = (|| {
        let out = Command::new("pdftoppm")
            // Uncompressed PPM: PNG would spend more time compressing than rendering.
            .args(["-r", "300", "-f", &page.to_string(), "-l", &page.to_string(), "-singlefile"])
            .arg(path)
            .arg(dir.join("page"))
            .output()
            .context("running pdftoppm (poppler) to render a page")?;
        if !out.status.success() {
            bail!("pdftoppm could not render page {page}: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        Ok(Rgb::from_image(image::open(dir.join("page.ppm"))?))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// The page's MediaBox, inherited from its parents when the page has none.
fn media_box(doc: &Document, mut id: ObjectId) -> Option<[f32; 4]> {
    for _ in 0..32 {
        let d = doc.get_dictionary(id).ok()?;
        if let Ok(arr) = d.get(b"MediaBox").and_then(|o| o.as_array()) {
            let v: Vec<f32> = arr.iter().filter_map(|o| o.as_float().ok().or_else(|| o.as_i64().ok().map(|i| i as f32))).collect();
            return (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]]);
        }
        id = d.get(b"Parent").ok()?.as_reference().ok()?;
    }
    None
}
