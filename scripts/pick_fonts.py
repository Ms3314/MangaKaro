"""Choose a primary and a secondary English font for one manga, from the weight of its own lettering.

For every text box in the staged outputs it measures how heavy the original lettering is (stroke thickness / glyph size, on
the pixels the text detector marks as lettering), then picks:
  primary    the candidate font whose weight is closest to the typical (median) lettering, used for normal speech
  secondary  the candidate closest to the heavy lettering (90th percentile) and heavier than the primary, for loud lines
and writes data/fonts/<folder>.json plus a specimen sheet output/fonts/<folder>.png showing every candidate with its
measured weight, so the picks can be judged by eye and overridden by editing the JSON.

Usage:
  .venv/bin/python scripts/pick_fonts.py daiboken-volume-2 --variant S7 --pages 20 21 22
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_detectors import ComicTextDetector
from lettering import SAMPLE, candidate_fonts, font_of, sample_ratio, weight_ratio
from translate_pages import load_pages, shrink

ROOT = Path(__file__).resolve().parent.parent
INK = 150


def measure_manga(folder, variant, pages):
    det = ComicTextDetector()
    ratios = []
    for name, page in load_pages(folder, pages):
        f = ROOT / "data/intermediate/staged" / variant / f"{name}.json"
        if not f.exists():
            continue
        page = shrink(page)
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        mask = det.mask(page)
        for box in json.loads(f.read_text())["boxes"].values():
            x0, y0, x1, y1 = box
            ink = (gray[y0:y1, x0:x1] < INK) & mask[y0:y1, x0:x1]
            r = weight_ratio(ink)
            if r is not None:
                ratios.append(r)
    return np.array(ratios)


def specimen(cands, primary, secondary, path):
    row = 78
    im = Image.new("RGB", (1500, row * len(cands) + 20), "white")
    d = ImageDraw.Draw(im)
    label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    for k, c in enumerate(cands):
        y = 10 + k * row
        tag = " PRIMARY" if c["name"] == primary["name"] else " SECONDARY" if c["name"] == secondary["name"] else ""
        d.text((10, y), f"{c['name']}   weight {c['ratio']:.3f}{tag}", font=label, fill=(200, 0, 0) if tag else (90, 90, 90))
        d.text((10, y + 22), SAMPLE, font=font_of(c, 36), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--pages", nargs="*")
    a = ap.parse_args()

    ratios = measure_manga(a.folder, a.variant, a.pages)
    normal, heavy = float(np.median(ratios)), float(np.percentile(ratios, 90))
    print(f"{len(ratios)} boxes measured: median weight {normal:.3f}, 90th percentile {heavy:.3f}")

    cands = []
    for c in candidate_fonts():
        try:
            c["ratio"] = sample_ratio(c)
        except OSError:
            continue
        if c["ratio"] is not None:
            cands.append(c)
    cands.sort(key=lambda c: c["ratio"])
    primary = min((c for c in cands if c["text"]), key=lambda c: abs(c["ratio"] - normal))
    heavier = [c for c in cands if c["ratio"] > primary["ratio"] * 1.1] or [max(cands, key=lambda c: c["ratio"])]
    secondary = min(heavier, key=lambda c: abs(c["ratio"] - heavy))

    stem = Path(a.folder).name
    out = ROOT / "data/fonts" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    pick = lambda c: {k: c[k] for k in ("name", "path", "index")}
    out.write_text(json.dumps({"manga": stem, "measured": {"boxes": len(ratios), "median": normal, "p90": heavy},
                               "primary": pick(primary), "secondary": pick(secondary),
                               "candidates": [{"name": c["name"], "weight": round(c["ratio"], 3)} for c in cands]},
                              ensure_ascii=False, indent=1))
    specimen(cands, primary, secondary, ROOT / "output/fonts" / f"{stem}.png")
    print(f"primary   {primary['name']} ({primary['ratio']:.3f})")
    print(f"secondary {secondary['name']} ({secondary['ratio']:.3f})")
    print(f"wrote {out.relative_to(ROOT)} and output/fonts/{stem}.png ({len(cands)} candidates)")


if __name__ == "__main__":
    main()
