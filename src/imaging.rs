//! Pixels in and tensors out: page images as 8-bit RGB, the detector's input (resized and
//! normalised in one parallel pass) and the recognizer's crops (sampled straight from the page).

use rayon::prelude::*;

use crate::geom::{Pt, dist};

/// A page image: 8-bit RGB, row-major.
#[derive(Clone)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

impl Rgb {
    pub fn from_image(img: image::DynamicImage) -> Rgb {
        let rgb = img.into_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        Rgb { w, h, px: rgb.into_raw() }
    }

    /// Halves the image (2×2 average) until its long side is at most `max_side`. Scans far above
    /// 300 ppi only cost time.
    pub fn capped(mut self, max_side: usize) -> Rgb {
        while self.w.max(self.h) > max_side {
            let (w, h) = (self.w / 2, self.h / 2);
            let src = &self.px;
            let sw = self.w;
            let mut out = vec![0u8; w * h * 3];
            out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                for x in 0..w {
                    for c in 0..3 {
                        let s = |yy: usize, xx: usize| src[(yy * sw + xx) * 3 + c] as u32;
                        row[x * 3 + c] = ((s(2 * y, 2 * x) + s(2 * y, 2 * x + 1) + s(2 * y + 1, 2 * x) + s(2 * y + 1, 2 * x + 1) + 2) / 4) as u8;
                    }
                }
            });
            self = Rgb { w, h, px: out };
        }
        self
    }

    /// Bilinear sample of channel `c` at a real position, edges repeated.
    #[inline]
    fn sample(&self, x: f32, y: f32, c: usize) -> f32 {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |xx: usize, yy: usize| self.px[(yy * self.w + xx) * 3 + c] as f32;
        let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * fx;
        let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * fx;
        top + (bot - top) * fy
    }
}

/// The detector's input size for a page: long side at most `max_side`, both sides rounded to
/// multiples of 32 (the network halves the image five times).
pub fn det_size(w: usize, h: usize, max_side: usize) -> (usize, usize) {
    let ratio = if w.max(h) > max_side { max_side as f32 / w.max(h) as f32 } else { 1.0 };
    let r32 = |v: f32| ((v / 32.0).round() as usize).max(1) * 32;
    (r32(w as f32 * ratio), r32(h as f32 * ratio))
}

/// The page resized to `dw × dh` and normalised to `(x/255 − 0.5) / 0.5`, as NCHW floats.
pub fn det_tensor(img: &Rgb, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * dw * dh];
    let (sx, sy) = (img.w as f32 / dw as f32, img.h as f32 / dh as f32);
    let plane = dw * dh;
    let (r, rest) = out.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    r.par_chunks_mut(dw).zip(g.par_chunks_mut(dw)).zip(b.par_chunks_mut(dw)).enumerate().for_each(|(y, ((rr, gr), br))| {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        for x in 0..dw {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            rr[x] = img.sample(fx, fy, 0) / 127.5 - 1.0;
            gr[x] = img.sample(fx, fy, 1) / 127.5 - 1.0;
            br[x] = img.sample(fx, fy, 2) / 127.5 - 1.0;
        }
    });
    out
}

/// One text box cut out of the page, upright, 48 px high and as wide as its shape asks, already
/// normalised; `ratio` is width/height for batching.
pub struct Crop {
    pub w: usize,
    pub ratio: f32,
    pub data: Vec<f32>, // 3 × 48 × w
}

pub const REC_H: usize = 48;

/// Cuts the box `q` (top-left, top-right, bottom-right, bottom-left) out of the page straight to
/// recognizer size: one bilinear sample per output pixel, no intermediate crop. Boxes much taller
/// than wide are turned a quarter (vertical text), as PaddleOCR does.
pub fn crop(img: &Rgb, q: &[Pt; 4], max_w: usize) -> Crop {
    let bw = dist(q[0], q[1]).max(dist(q[3], q[2])).max(1.0);
    let bh = dist(q[0], q[3]).max(dist(q[1], q[2])).max(1.0);
    let vertical = bh / bw >= 1.5;
    // Origin and axes in page space for output x (along the text) and y (across it).
    let (o, ax, ay, ratio) = if vertical {
        // np.rot90: the new x runs down the old box, the new y runs right to left.
        (q[1], [q[3][0] - q[0][0], q[3][1] - q[0][1]], [q[0][0] - q[1][0], q[0][1] - q[1][1]], bh / bw)
    } else {
        (q[0], [q[1][0] - q[0][0], q[1][1] - q[0][1]], [q[3][0] - q[0][0], q[3][1] - q[0][1]], bw / bh)
    };
    let w = ((REC_H as f32 * ratio).ceil() as usize).clamp(1, max_w);
    let mut data = vec![0f32; 3 * REC_H * w];
    let plane = REC_H * w;
    for y in 0..REC_H {
        let v = (y as f32 + 0.5) / REC_H as f32;
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let px = o[0] + u * ax[0] + v * ay[0];
            let py = o[1] + u * ax[1] + v * ay[1];
            for c in 0..3 {
                data[c * plane + y * w + x] = img.sample(px, py, c) / 127.5 - 1.0;
            }
        }
    }
    Crop { w, ratio, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_size_rounds_to_32_and_caps() {
        assert_eq!(det_size(2480, 3508, 1600), (1120, 1600));
        assert_eq!(det_size(500, 300, 1600), (512, 288));
    }

    #[test]
    fn capped_halves_until_it_fits() {
        let img = Rgb { w: 4000, h: 2000, px: vec![200; 4000 * 2000 * 3] }.capped(1600);
        assert_eq!((img.w, img.h), (1000, 500));
        assert_eq!(img.px[0], 200);
    }

    #[test]
    fn crop_reads_the_box() {
        // A white page with a black 100×20 bar at (10, 10).
        let (w, h) = (200, 100);
        let mut px = vec![255u8; w * h * 3];
        for y in 10..30 {
            for x in 10..110 {
                for c in 0..3 {
                    px[(y * w + x) * 3 + c] = 0;
                }
            }
        }
        let img = Rgb { w, h, px };
        let c = crop(&img, &[[10.0, 10.0], [110.0, 10.0], [110.0, 30.0], [10.0, 30.0]], 2000);
        assert_eq!(c.w, 240); // 48 × 100/20
        // Inside the bar is black; the outermost samples blend with the white around it.
        let inner = (4..REC_H - 4).flat_map(|y| (4..c.w - 4).map(move |x| y * 240 + x));
        assert!(inner.map(|i| c.data[i]).all(|v| v < -0.9), "black inside");
    }
}
