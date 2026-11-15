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

/// The Luhn check (the last digit makes the weighted sum a multiple of 10): credit cards, IMEI,
/// Mexico's NSS and others.
pub fn luhn_ok(digits: &[u8]) -> bool {
    let mut sum = 0u32;
    for (i, &d) in digits.iter().rev().enumerate() {
        let mut v = d as u32;
        if i % 2 == 1 {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
    }
    sum % 10 == 0
}

/// A rule for digit runs of one length: the check their last digit must satisfy.
#[derive(Clone, Copy, Debug)]
pub struct DigitCheck {
    pub len: usize,
    pub check: fn(&[u8]) -> bool,
}

/// How unsure the network must have been about a digit before it may change, and how likely the
/// replacement must have been.
pub const MAX_DOUBTED: f32 = 0.6;
pub const MIN_ALTERNATIVE: f32 = 0.2;

/// Checksum-aware correction. For each run of exactly `rule.len` digits that fails its check,
/// try changing one digit the network doubted (probability < MAX_DOUBTED) to the alternative it
/// gave at least MIN_ALTERNATIVE; keep the most likely change that passes. Clear readings are
/// never touched, so a number that simply doesn't use the check stays as read. Returns how many
/// digits changed.
pub fn fix_check_digits(d: &mut Decoded, probs: &[f32], c: usize, digit_class: &[usize; 10], rule: DigitCheck) -> usize {
    let chars: Vec<char> = d.text.chars().collect();
    if chars.len() != d.cols.len() {
        return 0;
    }
    let mut changed = 0;
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // A number: digits, possibly in groups split by single spaces or dashes ("12 34 56-7").
        let mut pos: Vec<usize> = Vec::new();
        while i < chars.len() {
            if chars[i].is_ascii_digit() {
                pos.push(i);
                i += 1;
            } else if (chars[i] == ' ' || chars[i] == '-') && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                i += 1;
            } else {
                break;
            }
        }
        if pos.len() != rule.len {
            continue;
        }
        let digits: Vec<u8> = pos.iter().map(|&p| chars[p] as u8 - b'0').collect();
        if (rule.check)(&digits) {
            continue;
        }
        let mut best: Option<(usize, u8, f32)> = None;
        for (k, &cur) in digits.iter().enumerate() {
            let at = pos[k];
            if d.char_probs[at] >= MAX_DOUBTED {
                continue;
            }
            let row = &probs[d.cols[at] * c..(d.cols[at] + 1) * c];
            for alt in 0..10u8 {
                if alt == cur {
                    continue;
                }
                let p = row[digit_class[alt as usize]];
                if p < MIN_ALTERNATIVE {
                    continue;
                }
                let mut trial = digits.clone();
                trial[k] = alt;
                if (rule.check)(&trial) && best.is_none_or(|b| p > b.2) {
                    best = Some((at, alt, p));
                }
            }
        }
        if let Some((pos, alt, p)) = best {
            let mut text: Vec<char> = d.text.chars().collect();
            text[pos] = (b'0' + alt) as char;
            d.text = text.into_iter().collect();
            d.char_probs[pos] = p;
            changed += 1;
        }
    }
    if changed > 0 {
        d.score = d.char_probs.iter().sum::<f32>() / d.char_probs.len() as f32;
    }
    changed
}

/// The class index of each digit '0'..'9' in `charset` (class = position + 1).
pub fn digit_classes(charset: &[String]) -> Option<[usize; 10]> {
    let mut out = [0usize; 10];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = charset.iter().position(|s| s == &k.to_string())? + 1;
    }
    Some(out)
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
    fn luhn() {
        assert!(luhn_ok(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3]));
        assert!(!luhn_ok(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 4]));
    }

    /// Classes: 0 blank, 1..=10 digits '0'..'9', 11 '-', 12 space.
    fn digits_charset() -> Vec<String> {
        let mut v: Vec<String> = (0..10).map(|d| d.to_string()).collect();
        v.push("-".into());
        v.push(" ".into());
        v
    }

    /// One column per character; `doubt` gives (position, chosen p, alternative digit, alt p).
    fn columns(text: &str, doubt: Option<(usize, f32, u8, f32)>) -> (Vec<f32>, usize) {
        let c = 13;
        let cls = |ch: char| match ch {
            '0'..='9' => ch as usize - '0' as usize + 1,
            '-' => 11,
            _ => 12,
        };
        let mut probs = Vec::new();
        for (i, ch) in text.chars().enumerate() {
            let mut row = vec![0.0f32; c];
            match doubt {
                Some((pos, p, alt, pa)) if pos == i => {
                    row[cls(ch)] = p;
                    row[alt as usize + 1] = pa;
                }
                _ => row[cls(ch)] = 0.99,
            }
            probs.extend(row);
        }
        (probs, c)
    }

    #[test]
    fn a_doubted_digit_is_fixed_by_the_check_digit() {
        let cs = digits_charset();
        let classes = digit_classes(&cs).unwrap();
        let rule = DigitCheck { len: 11, check: luhn_ok };
        // True number 12345678903; the network read the 5th digit as 8 (0.48) over 5 (0.45).
        let (probs, c) = columns("12348678903", Some((4, 0.48, 5, 0.45)));
        let mut d = greedy(&probs, 11, c, &cs);
        assert_eq!(d.text, "12348678903");
        assert_eq!(fix_check_digits(&mut d, &probs, c, &classes, rule), 1);
        assert_eq!(d.text, "12345678903");
    }

    #[test]
    fn clear_readings_and_other_lengths_are_left_alone() {
        let cs = digits_charset();
        let classes = digit_classes(&cs).unwrap();
        let rule = DigitCheck { len: 11, check: luhn_ok };
        // Fails Luhn but every digit was read clearly: a folio, not an error.
        let (probs, c) = columns("12345678904", None);
        let mut d = greedy(&probs, 11, c, &cs);
        assert_eq!(fix_check_digits(&mut d, &probs, c, &classes, rule), 0);
        assert_eq!(d.text, "12345678904");
        // Ten digits: not this rule's length.
        let (probs, c) = columns("1234867890", Some((4, 0.48, 5, 0.45)));
        let mut d = greedy(&probs, 10, c, &cs);
        assert_eq!(fix_check_digits(&mut d, &probs, c, &classes, rule), 0);
    }

    #[test]
    fn numbers_split_by_spaces_or_dashes_are_checked_whole() {
        let cs = digits_charset();
        let classes = digit_classes(&cs).unwrap();
        let rule = DigitCheck { len: 11, check: luhn_ok };
        // "12 34 86 7890 3": the 5th digit (text position 6) doubted, 5 was the runner-up.
        let (probs, c) = columns("12 34 86-7890 3", Some((6, 0.5, 5, 0.4)));
        let mut d = greedy(&probs, 15, c, &cs);
        assert_eq!(fix_check_digits(&mut d, &probs, c, &classes, rule), 1);
        assert_eq!(d.text, "12 34 56-7890 3");
    }

    #[test]
    fn charset_adds_space() {
        assert_eq!(charset_from_metadata("0\n1\n2\n"), ["0", "1", "2", " "]);
    }
}
