"""Phase 3 prototype: erase the original text in plain white bubbles and typeset the English translation.

Reads the final per-page JSON written by staged.py (boxes + translations) and the matching page image, and writes
  output/rendered/<variant>/<page>.png          translated page
  output/rendered/<variant>/<page>.compare.jpg  original | translated, side by side
  output/rendered/<variant>/report.json         per box group: rendered / overflow / skipped, and why

Easy case only. A box counts as a plain white bubble if the ring just outside it is mostly white; anything else
(text over art, sound effects) is left untouched and reported. Per box: dark pixels inside the box are painted white,
then the box is grown inside the connected white area around it (the bubble interior) and the English is wrapped and
sized to fit. Each box is typeset on its own; it may not grow into another box's text.

Usage:
  .venv/bin/python scripts/render.py daiboken-volume-2 --variant S7 --pages 20 21
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from translate_pages import load_pages, shrink

ROOT = Path(__file__).resolve().parent.parent
FONT = "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf"
INK = 150          # gray below this is "not background" when judging a box
ERASE_INK = 200    # gray below this is lettering to erase (anti-aliased stroke edges are lighter than the core)
WHITE = 215        # gray above this is paper (photographed pages: paper is about 225-240)
MIN_PAPER = 0.8    # share of the non-ink pixels in a box that must be paper for it to count as a plain white bubble
MAX_SIZE, MIN_SIZE = 38, 10


def clamp(box, shape, pad=0):
    H, W = shape[:2]
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def is_white_bubble(gray, box):
    """True if, ignoring the lettering itself, the inside of the box is paper-white.

    Measured on the S7 test pages: real white bubbles score 0.82-0.97, boxes over art, captions or sound effects 0.00-0.67.
    (A ring around the box does not work: tight boxes sit close to the bubble outline, so a ring crosses it.)"""
    x0, y0, x1, y1 = clamp(box, gray.shape)
    roi = gray[y0:y1, x0:x1]
    background = roi >= INK
    return background.any() and (roi > WHITE).sum() / background.sum() >= MIN_PAPER


def erase(img, gray, box, pad=2):
    """Paint the lettering inside the box with the bubble's own paper colour (median of its paper pixels)."""
    x0, y0, x1, y1 = clamp(box, gray.shape, pad)
    roi = gray[y0:y1, x0:x1]
    paper = roi > WHITE
    colour = np.median(img[y0:y1, x0:x1][paper], axis=0) if paper.any() else np.array([235, 235, 235])
    ink = (roi < ERASE_INK).astype(np.uint8)
    # Lettering is made of small strokes. Large dark shapes (the bubble outline, black background around the bubble,
    # art) are big connected components: leave them alone, or the box becomes a hard-edged white rectangle.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    limit = 0.08 * roi.size
    h, w = roi.shape
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        # also keep a long stroke that runs along the box edge on a big box: that is the bubble outline passing through
        touches = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        outline = touches and max(h, w) > 100 and max(bw, bh) > 0.5 * max(h, w)
        if area > limit or outline:
            ink[labels == k] = 0
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    img[y0:y1, x0:x1][ink > 0] = colour


def grow(mask, rect, limit, step=4):
    """Grow rect side by side while the new strip stays inside mask, never past limit (the bubble interior)."""
    x0, y0, x1, y1 = rect
    lx0, ly0, lx1, ly1 = limit
    moved = True
    while moved:
        moved = False
        if x0 - step >= lx0 and mask[y0:y1, x0 - step:x0].all():
            x0 -= step; moved = True
        if x1 + step <= lx1 and mask[y0:y1, x1:x1 + step].all():
            x1 += step; moved = True
        if y0 - step >= ly0 and mask[y0 - step:y0, x0:x1].all():
            y0 -= step; moved = True
        if y1 + step <= ly1 and mask[y1:y1 + step, x0:x1].all():
            y1 += step; moved = True
    return x0, y0, x1, y1


def wrap(text, font, width):
    """Greedy word wrap; None if a single word is wider than the width."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= width:
            cur = trial
        elif not cur:
            return None
        else:
            lines.append(cur)
            cur = word
            if font.getlength(cur) > width:
                return None
    return lines + [cur] if cur else lines


def fit(text, w, h):
    """Largest font size at which the wrapped text fits in w x h. Returns (font, lines, line_height, fits)."""
    for size in range(MAX_SIZE, MIN_SIZE - 1, -1):
        font = ImageFont.truetype(FONT, size)
        lines = wrap(text, font, w)
        if lines is not None and int(size * 1.12) * len(lines) <= h:
            return font, lines, int(size * 1.12), True
    font = ImageFont.truetype(FONT, MIN_SIZE)
    return font, wrap(text, font, 10 ** 6), int(MIN_SIZE * 1.12), False


def render_page(page, final):
    img = page.copy()
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    boxes = final["boxes"]
    items, report = [], []
    for r in final["regions"]:
        text = (r["translation"] or "").strip()
        if not text:
            report.append({"ids": [r["id"]], "status": "skipped: no translation"})
        elif not is_white_bubble(gray, boxes[r["id"]]):
            report.append({"ids": [r["id"]], "status": "skipped: not a plain white bubble"})
        else:
            items.append((r["id"], boxes[r["id"]], text))
    for _, box, _ in items:
        erase(img, gray, box)

    # Per box: find the white area around it in a small window (the bubble interior), grow the box inside it,
    # never into another box's text. A global connected-components pass merges bubbles that touch the gutters.
    erased = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    taken = np.zeros(gray.shape, bool)
    draws = []
    for i, box, text in items:
        x0, y0, x1, y1 = box
        ex, ey = max(30, x1 - x0), max(30, (y1 - y0) // 2)  # most the box may grow on each side
        limit = clamp((x0 - ex, y0 - ey, x1 + ex, y1 + ey), gray.shape)
        wx0, wy0, wx1, wy1 = clamp(limit, gray.shape, 8)
        _, lab = cv2.connectedComponents((erased[wy0:wy1, wx0:wx1] > WHITE).astype(np.uint8), connectivity=4)
        cy, cx = (y0 + y1) // 2 - wy0, (x0 + x1) // 2 - wx0
        mask = np.zeros(gray.shape, bool)
        if lab[cy, cx]:
            inner = cv2.erode((lab == lab[cy, cx]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            mask[wy0:wy1, wx0:wx1] = inner
        mask &= ~taken
        rect = grow(mask, (x0, y0, x1, y1), limit)
        note = "" if lab[cy, cx] else " (no white area around the box, used the box)"
        font, lines, lh, fits = fit(text, rect[2] - rect[0], rect[3] - rect[1])
        taken[rect[1]:rect[3], rect[0]:rect[2]] = True
        draws.append((rect, font, lines, lh))
        report.append({"ids": [i], "status": ("rendered" if fits else "overflow: text did not fit at the minimum size") + note,
                       "rect": list(rect), "font_size": font.size, "text": text})

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for (x0, y0, x1, y1), font, lines, lh in draws:
        top = (y0 + y1) / 2 - lh * len(lines) / 2
        for k, line in enumerate(lines):
            d.text(((x0 + x1) / 2, top + k * lh), line, font=font, fill=(0, 0, 0), anchor="ma")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--variant", required=True, help="staged.py variant to read, e.g. S7")
    ap.add_argument("--pages", nargs="*", help="image numbers, e.g. 20 21")
    a = ap.parse_args()

    src = ROOT / "data/intermediate/staged" / a.variant
    out = ROOT / "output/rendered" / a.variant
    out.mkdir(parents=True, exist_ok=True)
    full = {}
    for name, page in load_pages(a.folder, a.pages):
        f = src / f"{name}.json"
        if not f.exists():
            continue
        page = shrink(page)
        rendered, report = render_page(page, json.loads(f.read_text()))
        cv2.imwrite(str(out / f"{name}.png"), rendered)
        both = cv2.hconcat([page, rendered])
        s = 1100 / both.shape[0]
        cv2.imwrite(str(out / f"{name}.compare.jpg"), cv2.resize(both, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        full[name] = report
        ok = sum(1 for g in report if g["status"].startswith(("rendered",)))
        print(f"{name}: {ok}/{len(report)} groups rendered, "
              f"{sum(1 for g in report if g['status'].startswith('skipped'))} skipped, "
              f"{sum(1 for g in report if g['status'].startswith('overflow'))} overflow")
    (out / "report.json").write_text(json.dumps(full, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
