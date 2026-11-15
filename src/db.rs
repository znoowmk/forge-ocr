//! Differentiable Binarization post-processing: the detector's probability map to text boxes.
//!
//! PaddleOCR does this with OpenCV contours and pyclipper. Here: threshold and 2×2 dilation,
//! 8-connected components with an explicit stack, the hull of each component's boundary pixels,
//! the minimum-area rectangle, the mean probability inside it, then the "unclip" (the network
//! marks the core of each line; the box is grown by area × ratio / perimeter on every side,
//! which is what offsetting a rectangle and taking its bounding rectangle again gives).

use crate::geom::{Pt, RotRect, convex_hull, in_quad, min_area_rect, order_corners};

#[derive(Clone, Copy, Debug)]
pub struct DbParams {
    pub thresh: f32,
    pub box_thresh: f32,
    pub unclip_ratio: f32,
    pub max_candidates: usize,
}

impl Default for DbParams {
    fn default() -> Self {
        DbParams { thresh: 0.3, box_thresh: 0.5, unclip_ratio: 1.6, max_candidates: 1000 }
    }
}

/// A detected text box in page pixels: corners top-left, top-right, bottom-right, bottom-left.
#[derive(Clone, Copy, Debug)]
pub struct TextBox {
    pub quad: [Pt; 4],
    pub score: f32,
}

/// Text boxes from a `w × h` probability map, scaled to a `page_w × page_h` page.
pub fn boxes(prob: &[f32], w: usize, h: usize, page_w: usize, page_h: usize, p: DbParams) -> Vec<TextBox> {
    // Threshold, then dilate with a 2×2 kernel (PaddleOCR's use_dilation).
    let mut bin = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let hit = |xx: usize, yy: usize| prob[yy * w + xx] > p.thresh;
            bin[y * w + x] = hit(x, y) || (x > 0 && hit(x - 1, y)) || (y > 0 && hit(x, y - 1)) || (x > 0 && y > 0 && hit(x - 1, y - 1));
        }
    }
    let (sx, sy) = (page_w as f32 / w as f32, page_h as f32 / h as f32);
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    let mut pixels: Vec<(usize, usize)> = Vec::new();
    for start in 0..w * h {
        if !bin[start] || seen[start] {
            continue;
        }
        if out.len() >= p.max_candidates {
            break;
        }
        // Flood the component (8-connected).
        pixels.clear();
        stack.push(start);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            pixels.push((x, y));
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if bin[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        // Boundary pixels are enough for the hull.
        let mut edge: Vec<Pt> = pixels
            .iter()
            .filter(|&&(x, y)| x == 0 || y == 0 || x + 1 == w || y + 1 == h || !bin[y * w + x - 1] || !bin[y * w + x + 1] || !bin[(y - 1) * w + x] || !bin[(y + 1) * w + x])
            .map(|&(x, y)| [x as f32, y as f32])
            .collect();
        let hull = convex_hull(&mut edge);
        let rect = min_area_rect(&hull);
        if rect.w.min(rect.h) < 3.0 {
            continue;
        }
        let score = mean_inside(prob, w, h, &order_corners(rect.corners()));
        if score < p.box_thresh {
            continue;
        }
        let d = rect.w * rect.h * p.unclip_ratio / (2.0 * (rect.w + rect.h));
        let grown: RotRect = rect.grown(d);
        if grown.w.min(grown.h) < 5.0 {
            continue;
        }
        let mut quad = order_corners(grown.corners());
        for c in quad.iter_mut() {
            c[0] = (c[0] * sx).round().clamp(0.0, page_w as f32);
            c[1] = (c[1] * sy).round().clamp(0.0, page_h as f32);
        }
        out.push(TextBox { quad, score });
    }
    sort_reading_order(&mut out);
    out
}

/// Mean probability inside a quadrilateral (PaddleOCR's "fast" box score).
fn mean_inside(prob: &[f32], w: usize, h: usize, q: &[Pt; 4]) -> f32 {
    let xs = q.iter().map(|p| p[0]);
    let ys = q.iter().map(|p| p[1]);
    let x0 = xs.clone().fold(f32::MAX, f32::min).floor().clamp(0.0, (w - 1) as f32) as usize;
    let x1 = xs.fold(f32::MIN, f32::max).ceil().clamp(0.0, (w - 1) as f32) as usize;
    let y0 = ys.clone().fold(f32::MAX, f32::min).floor().clamp(0.0, (h - 1) as f32) as usize;
    let y1 = ys.fold(f32::MIN, f32::max).ceil().clamp(0.0, (h - 1) as f32) as usize;
    let (mut sum, mut n) = (0.0f64, 0usize);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if in_quad(q, [x as f32, y as f32]) {
                sum += prob[y * w + x] as f64;
                n += 1;
            }
        }
    }
    if n == 0 { 0.0 } else { (sum / n as f64) as f32 }
}

/// Top to bottom, then left to right among boxes on the same line (within 10 px), as
/// PaddleOCR's `sorted_boxes`.
pub fn sort_reading_order(b: &mut [TextBox]) {
    b.sort_by(|a, c| a.quad[0][1].total_cmp(&c.quad[0][1]).then(a.quad[0][0].total_cmp(&c.quad[0][0])));
    for i in 0..b.len().saturating_sub(1) {
        let mut j = i;
        while j < b.len() - 1 && (b[j + 1].quad[0][1] - b[j].quad[0][1]).abs() < 10.0 && b[j + 1].quad[0][0] < b[j].quad[0][0] {
            b.swap(j, j + 1);
            if j == 0 {
                break;
            }
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_two_lines_and_grows_them() {
        // 100×40 map, two bright bars.
        let (w, h) = (100, 40);
        let mut p = vec![0f32; w * h];
        for y in 5..10 {
            for x in 10..80 {
                p[y * w + x] = 0.9;
            }
        }
        for y in 25..30 {
            for x in 20..60 {
                p[y * w + x] = 0.9;
            }
        }
        let b = boxes(&p, w, h, 200, 80, DbParams::default());
        assert_eq!(b.len(), 2);
        // Scaled ×2 and grown: the first bar spans more than its core 140×10 page pixels.
        let q = b[0].quad;
        assert!(q[1][0] - q[0][0] > 140.0 && q[3][1] - q[0][1] > 10.0, "{q:?}");
        assert!(b[0].quad[0][1] < b[1].quad[0][1], "reading order");
    }

    #[test]
    fn faint_blobs_are_dropped() {
        let (w, h) = (50, 20);
        let mut p = vec![0f32; w * h];
        for y in 5..10 {
            for x in 5..40 {
                p[y * w + x] = 0.35; // above thresh, below box_thresh
            }
        }
        assert!(boxes(&p, w, h, w, h, DbParams::default()).is_empty());
    }
}
