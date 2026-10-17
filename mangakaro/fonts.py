"""Choosing the English fonts for a manga from the weight of its own lettering.

weight_ratio() measures how heavy lettering is: stroke thickness divided by typical glyph size, using the same code for the
manga's Japanese lettering and for a rendered sample of an English font, so the two can be compared. pick_fonts() then
chooses

  primary    the candidate whose weight is closest to the manga's median lettering, used for normal speech
  secondary  the candidate closest to its heavy (90th percentile) lettering and heavier than the primary, for loud lines

Only weight is matched, not letter shape. Candidates are the installed comic, hand-lettered, rounded and heavy display
faces listed below, plus any font file dropped in the project's fonts/ folder.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .paths import EXTRA_FONTS

FONT_DIRS = [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"), EXTRA_FONTS]
# Faces allowed for normal speech (comic, hand-lettered, rounded) and, on top of those, heavy display faces for loud lines.
TEXT_FAMILIES = ["Comic Sans MS", "Chalkboard SE", "Marker Felt", "Noteworthy", "Arial Rounded MT Bold"]
LOUD_FAMILIES = ["Avenir Next Condensed", "Helvetica Neue", "Optima", "Impact"]
FAMILIES = TEXT_FAMILIES + LOUD_FAMILIES
FILE_HINTS = ("comic", "chalkboard", "marker", "noteworthy", "rounded", "avenirnextcondensed", "avenir next condensed",
              "helveticaneue", "optima", "impact")
SAMPLE = "Hey, who said I'm a rookie? Let's go, you big fool!"
INK = 150  # gray below this is lettering ink when measuring a manga's weight

DEFAULT_FONT = {"name": "Comic Sans MS Bold", "path": "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf", "index": 0}


def candidate_fonts():
    """All installed upright faces of the allowed families, as specs {name, path, index, text}. text = fine for normal speech."""
    found = []
    for d in FONT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                continue
            if d != EXTRA_FONTS and not any(h in p.name.lower() for h in FILE_HINTS):
                continue
            for i in range(32):
                try:
                    family, style = ImageFont.truetype(str(p), 20, index=i).getname()
                except OSError:
                    break
                if "italic" in style.lower() or "oblique" in style.lower():
                    continue
                if d == EXTRA_FONTS or family in FAMILIES:
                    found.append({"name": f"{family} {style}", "path": str(p), "index": i,
                                  "text": d == EXTRA_FONTS or family in TEXT_FAMILIES})
                if p.suffix.lower() != ".ttc":
                    break
    return found


@lru_cache(maxsize=256)
def load(path, index, size):
    return ImageFont.truetype(path, size, index=index)


def font_of(spec, size):
    return load(spec["path"], spec.get("index", 0), size)


def weight_ratio(ink):
    """Stroke thickness / typical glyph size for a boolean ink image. None if there is too little ink to measure.

    Thickness = 2 * ink area / total contour length (exact for a uniform stroke). Glyph size = median over connected
    pieces of their longer side, so a kanji and a Latin letter are measured the same way."""
    ink = ink.astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    sizes = [max(s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT]) for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 15]
    if perimeter == 0 or len(sizes) < 3:
        return None
    return float(2 * ink.sum() / perimeter / np.median(sizes))


def sample_ratio(spec, size=64):
    """weight_ratio of the sample sentence set in this font."""
    font = font_of(spec, size)
    w = int(font.getlength(SAMPLE)) + 20
    im = Image.new("L", (w, size * 2), 255)
    ImageDraw.Draw(im).text((10, size // 2), SAMPLE, font=font, fill=0)
    return weight_ratio(np.array(im) < 128)


def measure_manga(items, text_detector):
    """Weight of the manga's lettering, one value per text box. items: iterable of (BGR page image, list of [x0,y0,x1,y1])."""
    ratios = []
    for page, boxes in items:
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        mask = text_detector.mask(page)
        for x0, y0, x1, y1 in boxes:
            r = weight_ratio((gray[y0:y1, x0:x1] < INK) & mask[y0:y1, x0:x1])
            if r is not None:
                ratios.append(r)
    return np.array(ratios)


def pick_fonts(ratios):
    """Choose primary and secondary fonts for a manga from its measured lettering weights.

    Returns {primary, secondary, measured, candidates} with font specs ready to save as JSON, or None if there was too
    little lettering to measure (the caller then uses DEFAULT_FONT)."""
    if len(ratios) < 3:
        return None
    normal, heavy = float(np.median(ratios)), float(np.percentile(ratios, 90))
    cands = []
    for c in candidate_fonts():
        try:
            c["ratio"] = sample_ratio(c)
        except OSError:
            continue
        if c["ratio"] is not None:
            cands.append(c)
    cands.sort(key=lambda c: c["ratio"])
    text_cands = [c for c in cands if c["text"]]
    if not text_cands:
        return None
    primary = min(text_cands, key=lambda c: abs(c["ratio"] - normal))
    heavier = [c for c in cands if c["ratio"] > primary["ratio"] * 1.1] or [max(cands, key=lambda c: c["ratio"])]
    secondary = min(heavier, key=lambda c: abs(c["ratio"] - heavy))
    spec = lambda c: {k: c[k] for k in ("name", "path", "index")}
    return {"primary": spec(primary), "secondary": spec(secondary),
            "measured": {"boxes": int(len(ratios)), "median": normal, "p90": heavy},
            "candidates": [{"name": c["name"], "weight": round(c["ratio"], 3)} for c in cands], "_cands": cands}


def write_specimen(picks, path):
    """Image of every candidate font with its measured weight and the picks marked, so they can be judged by eye."""
    cands = picks["_cands"]
    row = 78
    im = Image.new("RGB", (1500, row * len(cands) + 20), "white")
    d = ImageDraw.Draw(im)
    label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    for k, c in enumerate(cands):
        y = 10 + k * row
        tag = " PRIMARY" if c["name"] == picks["primary"]["name"] else " SECONDARY" if c["name"] == picks["secondary"]["name"] else ""
        d.text((10, y), f"{c['name']}   weight {c['ratio']:.3f}{tag}", font=label, fill=(200, 0, 0) if tag else (90, 90, 90))
        d.text((10, y + 22), SAMPLE, font=font_of(c, 36), fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
