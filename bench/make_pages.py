#!/usr/bin/env python3
"""Synthetic scanned pages with known text, for a benchmark anyone can reproduce.

    python3 bench/make_pages.py OUT_DIR [--pages 60] [--seed 1] [--fonts DIR...]

Each page is a US-letter page at 300 dpi with Spanish prose (accents, ñ, ¿¡), a payroll-style
table (names, 11-digit IDs with a Luhn check digit, amounts, dates) and a short list. It is then
degraded like a scan: slight rotation, blur, noise and JPEG compression. Writes OUT_DIR/pNNN.jpg,
OUT_DIR/pNNN.pdf (the JPEG wrapped in a PDF, as scanners do) and OUT_DIR/pNNN.gt.txt (the
text, one printed line per line).

Needs Pillow. Everything is generated, so the pages carry no real data.
"""
import argparse
import io
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORDS = (
    "el la de que y en un una por con para como más pero sus le ya o fue este ha sí porque esta "
    "entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno "
    "les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él "
    "tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros "
    "trabajo empresa contrato seguridad social obra cuota salario pago registro patronal periodo "
    "capacitación constancia trabajador nómina cálculo días retención aportación vivienda crédito "
    "información además según después también través pública económico número año México título "
    "ingeniería construcción supervisión mantenimiento señal niño año pequeño diseño compañía"
).split()
NAMES = "JUAN PEDRO MARÍA JOSÉ LUIS ANA SOFÍA MIGUEL ÁNGEL CARLOS LUCÍA DIEGO ELENA RAÚL NÉSTOR IÑAKI".split()
SURNAMES = "GARCÍA HERNÁNDEZ LÓPEZ MARTÍNEZ GONZÁLEZ PÉREZ SÁNCHEZ RAMÍREZ CRUZ FLORES GÓMEZ DÍAZ MUÑOZ NÚÑEZ PEÑA ORDÓÑEZ".split()

W, H = 2550, 3300  # US letter at 300 dpi


def luhn_digit(ten: str) -> str:
    for d in "0123456789":
        s = ten + d
        total = 0
        for i, ch in enumerate(reversed(s)):
            v = int(ch)
            if i % 2:
                v = v * 2 - 9 if v > 4 else v * 2
            total += v
        if total % 10 == 0:
            return d
    raise AssertionError


def sentence(rng: random.Random) -> str:
    n = rng.randint(6, 14)
    s = " ".join(rng.choice(WORDS) for _ in range(n))
    s = s[0].upper() + s[1:]
    end = rng.choice([".", ".", ".", "?", "!"])
    if end == "?":
        return "¿" + s + "?"
    if end == "!":
        return "¡" + s + "!"
    return s + "."


def wrap(text: str, font, width: int, draw) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def page(rng: random.Random, fonts: list[str]) -> tuple[Image.Image, list[str]]:
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    face = rng.choice(fonts)
    size = rng.choice([30, 34, 38, 42])
    font = ImageFont.truetype(face, size)
    bold = ImageFont.truetype(face, size + 10)
    gt: list[str] = []
    x0, y = 200, 220
    title = f"CONSTANCIA DE {rng.choice(['CAPACITACIÓN', 'PAGO', 'REGISTRO', 'SEGURIDAD'])} {rng.randint(2019, 2026)}"
    d.text((x0, y), title, font=bold, fill=0)
    gt.append(title)
    y += int(size * 2.4)
    for _ in range(rng.randint(2, 3)):
        para = " ".join(sentence(rng) for _ in range(rng.randint(2, 4)))
        for line in wrap(para, font, W - 2 * x0, d):
            d.text((x0, y), line, font=font, fill=0)
            gt.append(line)
            y += int(size * 1.45)
        y += int(size * 0.8)
    # Table: ID | NAME | DAYS | AMOUNT | DATE
    cols = [x0, x0 + int(size * 8.5), x0 + int(size * 30), x0 + int(size * 34), x0 + int(size * 41)]
    header = ["NSS", "NOMBRE", "DÍAS", "IMPORTE", "FECHA"]
    for c, h in zip(cols, header):
        d.text((c, y), h, font=font, fill=0)
    gt.append(" ".join(header))
    y += int(size * 1.6)
    for _ in range(rng.randint(8, 14)):
        if y > H - 300:
            break
        ten = "".join(rng.choice("0123456789") for _ in range(10))
        nss = ten + luhn_digit(ten)
        name = f"{rng.choice(SURNAMES)} {rng.choice(SURNAMES)} {rng.choice(NAMES)}"
        days = str(rng.randint(1, 31))
        amount = f"${rng.randint(1, 99):,}{rng.randint(0, 999):03d}.{rng.randint(0, 99):02d}"
        date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2019, 2026)}"
        row = [nss, name, days, amount, date]
        for c, v in zip(cols, row):
            d.text((c, y), v, font=font, fill=0)
        gt.append(" ".join(row))
        y += int(size * 1.4)
    return img, gt


def degrade(img: Image.Image, rng: random.Random, hard: bool) -> Image.Image:
    """A scan's flaws. `hard`: a poor scan or a phone photo of a page: more skew and blur, lower
    resolution (scanned at ~150 dpi and blown back up), uneven light and more noise."""
    img = img.rotate(rng.uniform(-3, 3) if hard else rng.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=255, expand=False)
    if hard:
        small = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
        img = small.resize(img.size, Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 1.8) if hard else rng.uniform(0.4, 1.1)))
    noise = Image.effect_noise(img.size, rng.uniform(20, 40) if hard else rng.uniform(8, 22))
    img = Image.blend(img, noise, 0.16 if hard else 0.08)
    if hard:
        # Light falling off towards one corner.
        shade = Image.linear_gradient("L").resize(img.size).rotate(rng.choice([0, 90, 180, 270]))
        img = Image.composite(img, img.point(lambda v: int(v * 0.72)), shade)
    # Paper tint, as a colour scan.
    return Image.merge("RGB", [img.point(lambda v, k=k: min(255, int(v * k))) for k in (1.0, 0.98, 0.93)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hard", action="store_true", help="poor scans: more skew, blur, noise, low resolution, uneven light")
    ap.add_argument("--fonts", nargs="*", default=[
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSerif-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    ])
    a = ap.parse_args()
    fonts = [f for f in a.fonts if Path(f).exists()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    for i in range(a.pages):
        img, gt = page(rng, fonts)
        img = degrade(img, rng, a.hard)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.choice([40, 50]) if a.hard else rng.choice([55, 65, 75]), dpi=(300, 300))
        (out / f"p{i:03d}.jpg").write_bytes(buf.getvalue())
        Image.open(io.BytesIO(buf.getvalue())).save(out / f"p{i:03d}.pdf", "PDF", resolution=300.0)
        (out / f"p{i:03d}.gt.txt").write_text("\n".join(gt) + "\n")
    print(f"{a.pages} pages in {out} (fonts: {len(fonts)})")


if __name__ == "__main__":
    main()
