"""Phase 3 prototype: erase the original lettering in plain white bubbles and typeset the English translation.

Reads the final per-page JSON written by staged.py (boxes + translations), the matching page image and the font picks
from pick_fonts.py, and writes
  output/rendered/<variant>/<page>.png          translated page
  output/rendered/<variant>/<page>.compare.jpg  original | translated, side by side
  output/rendered/<variant>/report.json         per text block: rendered / overflow / skipped, and why

Models: comic_text_detector gives a pixel mask of the lettering, so only the glyphs are erased (inside the translated
boxes, then inpainted); bubbles_yolov8 gives each bubble's boundary, so text is placed inside the bubble that holds it, and
boxes in the same bubble are typeset as one block. Easy case only: a box counts as a plain white bubble if, ignoring the
lettering and dark pixels, its inside is mostly paper-white. Anything else is left untouched and reported.

Usage:
  .venv/bin/python scripts/render.py daiboken-volume-2 --variant S7 --pages 20 21
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from compare_detectors import BubblesYolo, ComicTextDetector
from lettering import font_of
from translate_pages import load_pages, shrink

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FONT = {"name": "Comic Sans MS Bold", "path": "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf", "index": 0}
INK = 150          # gray below this is dark (lettering, outlines, black backgrounds), not background
WHITE = 215            # gray above this is white paper (photographed pages: paper is about 225-240). White paper is the main case.
MIN_PAPER = 0.8        # share of a box's background pixels that must be white for it to count as a plain white bubble
MIN_PAPER_LEVEL = 175  # fallback for shaded paper near the spine: the background must still be at least this bright (median gray)
FLAT_GRAD = 25         # background pixels with a brightness gradient below this count as flat
MIN_FLAT = 0.7         # share of background pixels that must be flat for a plain bubble
PAPER_FLOOR = 150      # nothing darker than this is paper when growing the text area
PAPER_SLACK = 35       # paper = within this much of the brightest pixel nearby, so shaded paper still counts
SPECK_AREA = 80        # a non-paper blob smaller than this (pixels) is a leftover speck, not an outline
MIN_READABLE = 14      # stay inside the bubble unless that makes the text smaller than this (px) and the detector box would allow bigger
ERASE_PAD = 3      # lettering mask is only erased this far outside the detector box
LOUD = 0.75        # emotion intensity at or above this (or bold / jagged / outlined lettering) uses the secondary font
TEXT_PAD = 6       # margin kept between the English and the edge of the space it is fitted into
MAX_SIZE, MIN_SIZE = 40, 10


def clamp(box, shape, pad=0):
    H, W = shape[:2]
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def is_plain_bubble(gray, grad, mask, box):
    """True if, ignoring the lettering and dark pixels, the inside of the box is plain paper.

    First choice: it is mostly white (most manga pages). Fallback: it is light and flat but shaded, as bubbles next to the
    book's spine are in a scan (median 175-225). Screentone and art are not flat, so they still fail."""
    x0, y0, x1, y1 = clamp(box, gray.shape)
    g = gray[y0:y1, x0:x1]
    background = ~cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (g >= INK)
    if background.sum() < 30:
        return False
    if ((g > WHITE) & background).sum() / background.sum() >= MIN_PAPER:
        return True
    return np.median(g[background]) >= MIN_PAPER_LEVEL and (grad[y0:y1, x0:x1][background] < FLAT_GRAD).mean() >= MIN_FLAT


def erase(img, mask, boxes):
    """Inpaint the lettering mask, but only within the translated boxes (plus a few pixels): nothing else on the page moves."""
    allowed = np.zeros(mask.shape, np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = clamp(b, mask.shape, ERASE_PAD)
        allowed[y0:y1, x0:x1] = 1
    target = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) & allowed  # dilate to take in anti-aliased edges
    return cv2.inpaint(img, target * 255, 3, cv2.INPAINT_TELEA)


def grow(mask, rect, limit, step=4):
    """Grow rect side by side while the new strip stays inside mask, never past limit."""
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


def fit(text, spec, w, h):
    """Largest size at which the wrapped text fits in w x h. Returns (font, lines, line_height, fits)."""
    for size in range(MAX_SIZE, MIN_SIZE - 1, -1):
        font = font_of(spec, size)
        lines = wrap(text, font, w)
        if lines is not None and int(size * 1.12) * len(lines) <= h:
            return font, lines, int(size * 1.12), True
    font = font_of(spec, MIN_SIZE)
    return font, wrap(text, font, 10 ** 6), int(MIN_SIZE * 1.12), False


def is_loud(region):
    s = region.get("style") or {}
    return (region.get("intensity") or 0) >= LOUD or s.get("weight") == "bold" or s.get("shape") in ("jagged", "outlined")


class Renderer:
    def __init__(self, fonts):
        self.text, self.bubbles = ComicTextDetector(), BubblesYolo()
        self.primary = fonts.get("primary", DEFAULT_FONT)
        self.secondary = fonts.get("secondary", DEFAULT_FONT)

    def page(self, page, final):
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        mask = self.text.mask(page)
        bubbles = [b["bbox"] for b in self.bubbles(page)]
        smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        grad = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
        boxes = final["boxes"]
        items, report = [], []
        for r in final["regions"]:
            if not (r["translation"] or "").strip():
                report.append({"ids": [r["id"]], "status": "skipped: no translation"})
            elif not is_plain_bubble(gray, grad, mask, boxes[r["id"]]):
                report.append({"ids": [r["id"]], "status": "skipped: not a plain white bubble"})
            else:
                items.append(r)
        img = erase(page, mask, [boxes[r["id"]] for r in items])

        # group boxes by the bubble that contains their centre (smallest one wins); boxes in no bubble stand alone
        groups = {}
        for r in items:
            x0, y0, x1, y1 = boxes[r["id"]]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            inside = [k for k, b in enumerate(bubbles) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
            key = min(inside, key=lambda k: (bubbles[k][2] - bubbles[k][0]) * (bubbles[k][3] - bubbles[k][1])) if inside else f"solo-{r['id']}"
            groups.setdefault(key, []).append(r)

        # Paper = white where white paper is nearby (the normal case). Only where no white is within 15 px (shaded paper by the
        # spine) does it fall back to "close to the brightest pixel nearby", so white-paper pages behave exactly as before.
        g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        local = cv2.dilate(g2, np.ones((31, 31), np.uint8))
        paper_px = np.where(local > WHITE, g2 > WHITE, (g2 >= local.astype(np.int16) - PAPER_SLACK) & (g2 >= PAPER_FLOOR))
        # Ghost strokes and dots left after erasing are small dark specks; they would stop the text area from growing. Count
        # small non-paper blobs as paper. Bubble outlines are long, so they stay.
        n, labels, stats, _ = cv2.connectedComponentsWithStats((~paper_px).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < SPECK_AREA:
                paper_px[labels == k] = True
        paper = cv2.erode(paper_px.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        taken = np.zeros(gray.shape, bool)
        draws = []
        for key, members in groups.items():
            bs = [boxes[m["id"]] for m in members]
            union = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
            if isinstance(key, int):
                bx0, by0, bx1, by1 = bubbles[key]
                mx, my = int(0.08 * (bx1 - bx0)), int(0.08 * (by1 - by0))
                limit = (min(bx0 + mx, union[0]), min(by0 + my, union[1]), max(bx1 - mx, union[2]), max(by1 - my, union[3]))
                note = ""
            else:  # no bubble found: let the box grow a limited amount into whatever white is around it
                ex, ey = max(30, union[2] - union[0]), max(30, (union[3] - union[1]) // 2)
                limit = clamp((union[0] - ex, union[1] - ey, union[2] + ex, union[3] + ey), gray.shape)
                note = " (no bubble box found)"
            # Grow from the middle of the text, not from the detector box: that box often touches the bubble outline, and a
            # rectangle that starts on the outline keeps it. If the middle is blocked the rectangle stays small, so the detector box is the fallback.
            cx, cy = (union[0] + union[2]) // 2, (union[1] + union[3]) // 2
            rect = grow(paper & ~taken, (cx - 6, cy - 6, cx + 6, cy + 6), limit)
            spec = self.secondary if any(is_loud(m) for m in members) else self.primary
            text = " ".join(m["translation"].strip() for m in members)
            font, lines, lh, fits = fit(text, spec, rect[2] - rect[0] - 2 * TEXT_PAD, rect[3] - rect[1] - 2 * TEXT_PAD)
            # Text inside the bubble is preferred. Only if the grown area is too small to read, take the detector box, which
            # may reach the outline but gives bigger text.
            alt = fit(text, spec, union[2] - union[0] - 2 * TEXT_PAD, union[3] - union[1] - 2 * TEXT_PAD)
            if not fits or (font.size < MIN_READABLE and alt[0].size > font.size):
                rect, (font, lines, lh, fits) = union, alt
                note += " (grown area too small, used the box)"
            taken[rect[1]:rect[3], rect[0]:rect[2]] = True
            draws.append((rect, font, lines, lh))
            report.append({"ids": [m["id"] for m in members], "status": ("rendered" if fits else "overflow: text did not fit at the minimum size") + note,
                           "font": spec["name"], "font_size": font.size, "rect": list(rect), "text": text})

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
    ap.add_argument("--fonts", help="font picks JSON (default data/fonts/<folder>.json if it exists)")
    a = ap.parse_args()

    fonts_path = Path(a.fonts) if a.fonts else ROOT / "data/fonts" / f"{Path(a.folder).name}.json"
    fonts = json.loads(fonts_path.read_text()) if fonts_path.exists() else {}
    print(f"fonts: primary {fonts.get('primary', DEFAULT_FONT)['name']}, secondary {fonts.get('secondary', DEFAULT_FONT)['name']}")
    renderer = Renderer(fonts)

    src = ROOT / "data/intermediate/staged" / a.variant
    out = ROOT / "output/rendered" / a.variant
    out.mkdir(parents=True, exist_ok=True)
    full = {}
    for name, page in load_pages(a.folder, a.pages):
        f = src / f"{name}.json"
        if not f.exists():
            continue
        page = shrink(page)
        rendered, report = renderer.page(page, json.loads(f.read_text()))
        cv2.imwrite(str(out / f"{name}.png"), rendered)
        both = cv2.hconcat([page, rendered])
        s = 1100 / both.shape[0]
        cv2.imwrite(str(out / f"{name}.compare.jpg"), cv2.resize(both, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        full[name] = report
        print(f"{name}: {sum(g['status'].startswith('rendered') for g in report)} blocks rendered, "
              f"{sum(g['status'].startswith('skipped') for g in report)} skipped, "
              f"{sum(g['status'].startswith('overflow') for g in report)} overflow")
    (out / "report.json").write_text(json.dumps(full, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
