//! Plane geometry for text boxes: convex hull, minimum-area rotated rectangle (what OpenCV calls
//! `minAreaRect`) and the corner order the recognizer expects.

pub type Pt = [f32; 2];

/// A rotated rectangle: centre, side lengths (`w` along `angle`, `h` across) and angle in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotRect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
}

impl RotRect {
    /// The four corners, in no particular order.
    pub fn corners(&self) -> [Pt; 4] {
        let (s, c) = self.angle.sin_cos();
        let (hw, hh) = (self.w / 2.0, self.h / 2.0);
        let ux = [c * hw, s * hw];
        let uy = [-s * hh, c * hh];
        [
            [self.cx - ux[0] - uy[0], self.cy - ux[1] - uy[1]],
            [self.cx + ux[0] - uy[0], self.cy + ux[1] - uy[1]],
            [self.cx + ux[0] + uy[0], self.cy + ux[1] + uy[1]],
            [self.cx - ux[0] + uy[0], self.cy - ux[1] + uy[1]],
        ]
    }

    /// The same rectangle grown by `d` on every side.
    pub fn grown(&self, d: f32) -> RotRect {
        RotRect { w: self.w + 2.0 * d, h: self.h + 2.0 * d, ..*self }
    }
}

fn cross(o: Pt, a: Pt, b: Pt) -> f32 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Convex hull (Andrew's monotone chain), counter-clockwise, without collinear points.
pub fn convex_hull(points: &mut [Pt]) -> Vec<Pt> {
    if points.len() < 3 {
        return points.to_vec();
    }
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let mut hull: Vec<Pt> = Vec::with_capacity(points.len() * 2);
    for &p in points.iter() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in points.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// The smallest-area rectangle that contains the convex polygon `hull` (rotating calipers: the
/// best rectangle has a side on one of the hull's edges).
pub fn min_area_rect(hull: &[Pt]) -> RotRect {
    match hull.len() {
        0 => return RotRect { cx: 0.0, cy: 0.0, w: 0.0, h: 0.0, angle: 0.0 },
        1 => return RotRect { cx: hull[0][0], cy: hull[0][1], w: 0.0, h: 0.0, angle: 0.0 },
        _ => {}
    }
    let mut best = RotRect { cx: 0.0, cy: 0.0, w: f32::MAX, h: f32::MAX, angle: 0.0 };
    let mut best_area = f32::MAX;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in hull {
            let (px, py) = (p[0] - a[0], p[1] - a[1]);
            let u = px * ux + py * uy;
            let v = -px * uy + py * ux;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let (w, h) = (max_u - min_u, max_v - min_v);
        let area = w * h;
        if area < best_area {
            best_area = area;
            let (cu, cv) = ((min_u + max_u) / 2.0, (min_v + max_v) / 2.0);
            best = RotRect { cx: a[0] + cu * ux - cv * uy, cy: a[1] + cu * uy + cv * ux, w, h, angle: uy.atan2(ux) };
        }
    }
    best
}

/// Corners as top-left, top-right, bottom-right, bottom-left (PaddleOCR's `get_mini_boxes`
/// order: the two leftmost points split by height, then the two rightmost).
pub fn order_corners(mut c: [Pt; 4]) -> [Pt; 4] {
    c.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let (l, r) = c.split_at_mut(2);
    let (tl, bl) = if l[0][1] <= l[1][1] { (l[0], l[1]) } else { (l[1], l[0]) };
    let (tr, br) = if r[0][1] <= r[1][1] { (r[0], r[1]) } else { (r[1], r[0]) };
    [tl, tr, br, bl]
}

pub fn dist(a: Pt, b: Pt) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Is `p` inside the convex quadrilateral `q` (corners in order, either orientation)?
pub fn in_quad(q: &[Pt; 4], p: Pt) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let c = cross(q[i], q[(i + 1) % 4], p);
        if c != 0.0 {
            if sign == 0.0 {
                sign = c.signum();
            } else if c.signum() != sign {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_and_axis_aligned_rect() {
        let mut pts: Vec<Pt> = (0..=10).flat_map(|x| (0..=4).map(move |y| [x as f32, y as f32])).collect();
        let hull = convex_hull(&mut pts);
        assert_eq!(hull.len(), 4);
        let r = min_area_rect(&hull);
        let (w, h) = if r.w >= r.h { (r.w, r.h) } else { (r.h, r.w) };
        assert!((w - 10.0).abs() < 1e-4 && (h - 4.0).abs() < 1e-4, "{r:?}");
        assert!((r.cx - 5.0).abs() < 1e-4 && (r.cy - 2.0).abs() < 1e-4);
    }

    #[test]
    fn rotated_rect_round_trips() {
        let truth = RotRect { cx: 50.0, cy: 30.0, w: 40.0, h: 10.0, angle: 0.3 };
        let mut pts = truth.corners().to_vec();
        let r = min_area_rect(&convex_hull(&mut pts));
        assert!((r.w * r.h - 400.0).abs() < 0.5, "{r:?}");
        let o = order_corners(r.corners());
        assert!(o[0][0] < o[1][0] && o[3][1] > o[0][1], "{o:?}");
        assert!(in_quad(&o, [50.0, 30.0]) && !in_quad(&o, [0.0, 0.0]));
    }

    #[test]
    fn growing_adds_to_both_sides() {
        let r = RotRect { cx: 0.0, cy: 0.0, w: 10.0, h: 2.0, angle: 0.0 }.grown(1.5);
        assert_eq!((r.w, r.h), (13.0, 5.0));
    }
}
