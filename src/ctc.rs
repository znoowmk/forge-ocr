//! CTC decoding of the recognizer's output: one probability row per image column; repeated
//! classes merge, the blank (class 0) separates them.

/// The decoded text of one crop, with the probability of each character and the mean.
#[derive(Clone, Debug, Default)]
pub struct Decoded {
    pub text: String,
    pub char_probs: Vec<f32>,
    /// Column of each character (for later constrained decoding over the full distribution).
    pub cols: Vec<usize>,
    pub score: f32,
}

/// Greedy CTC over `t × c` probabilities. `charset[k]` is class `k + 1`; the last class is space.
pub fn greedy(probs: &[f32], t: usize, c: usize, charset: &[String]) -> Decoded {
    let mut d = Decoded::default();
    let mut prev = usize::MAX;
    for col in 0..t {
        let row = &probs[col * c..(col + 1) * c];
        let (k, p) = row.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        if k != 0 && k != prev {
            if let Some(ch) = charset.get(k - 1) {
                d.text.push_str(ch);
                d.char_probs.push(p);
                d.cols.push(col);
            }
        }
        prev = k;
    }
    d.score = if d.char_probs.is_empty() { 0.0 } else { d.char_probs.iter().sum::<f32>() / d.char_probs.len() as f32 };
    d
}

/// The recognizer's alphabet from its `character` metadata (one symbol per line), with the
/// space PaddleOCR appends as the last class.
pub fn charset_from_metadata(s: &str) -> Vec<String> {
    let mut v: Vec<String> = s.split('\n').filter(|l| !l.is_empty()).map(str::to_string).collect();
    v.push(" ".to_string());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn onehot(seq: &[usize], c: usize) -> Vec<f32> {
        seq.iter().flat_map(|&k| (0..c).map(move |i| if i == k { 0.9 } else { 0.1 / (c - 1) as f32 })).collect()
    }

    #[test]
    fn merges_repeats_and_drops_blanks() {
        let cs: Vec<String> = ["a", "b", " "].iter().map(|s| s.to_string()).collect();
        // a a _ a b b _ b   ->  "aabb"  (blank splits the doubled letters)
        let p = onehot(&[1, 1, 0, 1, 2, 2, 0, 2], 4);
        let d = greedy(&p, 8, 4, &cs);
        assert_eq!(d.text, "aabb");
        assert_eq!(d.cols, [0, 3, 4, 7]);
        assert!((d.score - 0.9).abs() < 1e-6);
        assert_eq!(greedy(&onehot(&[3], 4), 1, 4, &cs).text, " ");
    }

    #[test]
    fn charset_adds_space() {
        assert_eq!(charset_from_metadata("0\n1\n2\n"), ["0", "1", "2", " "]);
    }
}
