"""Shared lettering helpers: candidate fonts, loading a font by spec, and a script-agnostic weight measure.

weight_ratio() measures how heavy lettering is as stroke thickness divided by typical glyph size, using the same code for
the manga's Japanese lettering and for a rendered sample of an English font, so the two can be compared.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
             Path(__file__).resolve().parent.parent / "fonts"]
# Edit freely; drop .ttf/.otf files in the project's fonts/ folder to add more (check each font's license first).
# Faces allowed for normal speech (comic, hand-lettered, rounded) and, on top of those, heavy display faces for loud lines.
TEXT_FAMILIES = ["Comic Sans MS", "Chalkboard SE", "Marker Felt", "Noteworthy", "Arial Rounded MT Bold"]
LOUD_FAMILIES = ["Avenir Next Condensed", "Helvetica Neue", "Optima", "Impact"]
FAMILIES = TEXT_FAMILIES + LOUD_FAMILIES
FILE_HINTS = ("comic", "chalkboard", "marker", "noteworthy", "rounded", "avenirnextcondensed", "avenir next condensed",
              "helveticaneue", "optima", "impact")
SAMPLE = "Hey, who said I'm a rookie? Let's go, you big fool!"


def candidate_fonts():
    """All installed upright faces of the allowed families, as specs {name, path, index, text}. text = fine for normal speech."""
    found = []
    for d in FONT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                continue
            if d.name != "fonts" and not any(h in p.name.lower() for h in FILE_HINTS):
                continue
            for i in range(32):
                try:
                    family, style = ImageFont.truetype(str(p), 20, index=i).getname()
                except OSError:
                    break
                if "italic" in style.lower() or "oblique" in style.lower():
                    continue
                if d.name == "fonts" or family in FAMILIES:
                    found.append({"name": f"{family} {style}", "path": str(p), "index": i,
                                  "text": d.name == "fonts" or family in TEXT_FAMILIES})
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
