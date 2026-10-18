"""Erase the original lettering and typeset the English translation.

Models: comic_text_detector gives a pixel mask of the lettering, so only the glyphs are erased (inside the translated
boxes, then inpainted); bubbles_yolov8 gives each bubble's boundary, so text is placed inside the bubble that holds it, and
boxes in the same bubble are typeset as one block.

Each box is one of three kinds, decided from the pixels around the lettering:
  plain    a white bubble (or light, flat, shaded paper near the book's spine): English in black, inside the bubble
  inverse  white lettering on a flat dark area: the lettering is erased to black and the English is white
  overlay  lettering over screentone or art: erased with LaMa (neural inpainting, if its weights are present, else OpenCV's
           basic inpainting), English in black with a white outline so it reads on any background
Sound effects (the emotion step labels them) are left alone.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .detect import BubblesYolo, ComicTextDetector
from .fonts import DEFAULT_FONT, font_of

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
DARK_LEVEL = 70      # median gray below this (and flat) is a dark area with light lettering: the "inverse" kind
DARK_FLAT = 0.5      # share of the dark area's pixels that must be flat
OVERLAY_PAD = 8      # overlay text may use the detector box plus this many pixels around it
OUTLINE = 3          # width in pixels of the white outline around overlay text
WIDE_TARGET = 16     # tall narrow boxes (vertical Japanese) are widened until the English reaches this size, if there is room
TALL = 1.8           # a box counts as a tall narrow column when its height is more than this times its width
ART_SHARE = 0.06     # a plain box with more than this share of dark non-lettering pixels has drawn art in it: erase it with LaMa


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


def classify(gray, grad, mask, box):
    """"plain", "inverse" or "overlay" for the lettering in this box (see the module docstring)."""
    if is_plain_bubble(gray, grad, mask, box):
        return "plain"
    x0, y0, x1, y1 = clamp(box, gray.shape)
    g = gray[y0:y1, x0:x1]
    around = ~cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # everything but the lettering
    if around.sum() >= 30 and np.median(g[around]) < DARK_LEVEL and (grad[y0:y1, x0:x1][around] < FLAT_GRAD).mean() >= DARK_FLAT:
        return "inverse"
    return "overlay"


def has_art(gray, mask, box):
    """True if the box holds drawn art (hair, a face, line work) besides the lettering: its non-lettering pixels are partly dark."""
    x0, y0, x1, y1 = clamp(box, gray.shape)
    g = gray[y0:y1, x0:x1]
    around = ~cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return around.sum() >= 30 and ((g < INK) & around).sum() / around.sum() > ART_SHARE


def paper_top(rect, paper, min_share=0.7, min_height=40):
    """For text that touches art (a head of hair under a caption): the rectangle cut down to its top run of rows that are mostly
    free paper, so the English stays on the white part. Returns rect unchanged if that run would be too short to use."""
    x0, y0, x1, y1 = rect
    share = paper[y0:y1, x0:x1].mean(axis=1)
    run = 0
    while run < len(share) and share[run] >= min_share:
        run += 1
    return (x0, y0, x1, y0 + run) if run >= min_height else rect


def widen(rect, free, text, spec, bounds, gray=None, target=WIDE_TARGET, step=6):
    """Vertical Japanese is a narrow column, but English is wider than it is tall. Grow the rectangle sideways, a strip at a
    time over the full height, while the strip is `free` (and, if `gray` is given, holds no long dark line such as a panel
    border), until the English reaches `target` pixels or there is no room. Returns (rect, fit result)."""
    x0, y0, x1, y1 = rect
    best = (rect, fit(text, spec, x1 - x0 - 2 * TEXT_PAD, y1 - y0 - 2 * TEXT_PAD))
    max_width = max(110, int(2.6 * (x1 - x0)))
    left = right = True

    def ok(a, b):
        if not free[y0:y1, a:b].all():
            return False
        return gray is None or (gray[y0:y1, a:b] < 90).mean(axis=0).max() < 0.85

    while (x1 - x0) < max_width and (left or right) and best[1][0].size < target:
        if left:
            if x0 - step >= bounds[0] and ok(x0 - step, x0):
                x0 -= step
            else:
                left = False
        if right:
            if x1 + step <= bounds[2] and ok(x1, x1 + step):
                x1 += step
            else:
                right = False
        f = fit(text, spec, x1 - x0 - 2 * TEXT_PAD, y1 - y0 - 2 * TEXT_PAD)
        if f[0].size > best[1][0].size:
            best = ((x0, y0, x1, y1), f)
    return best


def is_sound_effect(region):
    """The emotion step labels sound effects as the speaker."""
    speaker = (region.get("speaker") or "").lower()
    return "sound" in speaker or "sfx" in speaker


def erase(img, mask, boxes, dark_boxes=()):
    """Inpaint the lettering mask, but only within the translated boxes (plus a few pixels): nothing else on the page moves.

    dark_boxes: boxes of light lettering on a dark area. Inpainting blends grey into the dark there, so these get a flat fill
    with the area's own colour instead (a wider mask, to take in the lettering's soft edges)."""
    allowed = np.zeros(mask.shape, np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = clamp(b, mask.shape, ERASE_PAD)
        allowed[y0:y1, x0:x1] = 1
    target = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) & allowed  # dilate to take in anti-aliased edges
    out = cv2.inpaint(img, target * 255, 3, cv2.INPAINT_TELEA)
    for b in dark_boxes:
        x0, y0, x1, y1 = clamp(b, mask.shape, ERASE_PAD)
        region = np.zeros(mask.shape, bool)
        region[y0:y1, x0:x1] = True
        wide = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & region
        rest = region & ~wide
        if rest.sum() >= 30:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dark = rest & (gray < DARK_LEVEL + 30)  # the area's own colour, not nearby light art
            if dark.sum() >= 30:
                out[wide] = np.median(img[dark], axis=0).astype(np.uint8)
    return out


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
    widest = max((font.getlength(word) for word in text.split()), default=0)
    return font, wrap(text, font, max(w, widest)) or [text], int(MIN_SIZE * 1.12), False  # too tall for the space, but not one long line


def is_loud(region):
    s = region.get("style") or {}
    return (region.get("intensity") or 0) >= LOUD or s.get("weight") == "bold" or s.get("shape") in ("jagged", "outlined")


class Renderer:
    def __init__(self, fonts=None, text_detector=None, bubble_detector=None, overlay=True, lama=None):
        """fonts: {primary, secondary} font specs (see fonts.pick_fonts). Detectors are created if not given.
        overlay: also translate text on dark areas and over screentone/art; False leaves everything but plain bubbles alone."""
        self.overlay = overlay
        self.lama = lama  # an inpaint.Lama, or None to use OpenCV's basic inpainting everywhere
        fonts = fonts or {}
        self.text = text_detector or ComicTextDetector()
        self.bubbles = bubble_detector or BubblesYolo()
        self.primary = fonts.get("primary", DEFAULT_FONT)
        self.secondary = fonts.get("secondary", DEFAULT_FONT)

    def page(self, page, final):
        """Translate one page. `final` is the page's result JSON (boxes + regions with translations).
        Returns (translated BGR image, report: one entry per text block saying rendered / overflow / skipped and why)."""
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        mask = self.text.mask(page)
        bubbles = [b["bbox"] for b in self.bubbles(page)]
        smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        grad = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
        boxes = final["boxes"]
        items, others, report = [], [], []  # items: plain bubbles; others: (region, kind) for inverse and overlay
        for r in final["regions"]:
            if not (r["translation"] or "").strip():
                report.append({"ids": [r["id"]], "status": f"skipped: {r.get('skip') or 'no translation'}"})
                continue
            kind = classify(gray, grad, mask, boxes[r["id"]])
            if kind == "plain":
                items.append(r)
            elif is_sound_effect(r):
                report.append({"ids": [r["id"]], "status": "skipped: sound effect"})
            elif not self.overlay:
                report.append({"ids": [r["id"]], "status": "skipped: not a plain white bubble"})
            else:
                others.append((r, kind))
        img = erase(page, mask, [boxes[r["id"]] for r in items] + [boxes[r["id"]] for r, _ in others],
                    dark_boxes=[boxes[r["id"]] for r, kind in others if kind == "inverse"])
        over_boxes = [boxes[r["id"]] for r, kind in others if kind == "overlay"]
        art_ids = {r["id"] for r in items if has_art(gray, mask, boxes[r["id"]])}  # plain, but the lettering touches drawn art
        over_boxes += [boxes[i] for i in art_ids]
        if self.lama is not None and over_boxes:  # screentone and line art: continue the texture instead of blurring it
            allowed = np.zeros(mask.shape, np.uint8)
            for b in over_boxes:
                x0, y0, x1, y1 = clamp(b, mask.shape, ERASE_PAD)
                allowed[y0:y1, x0:x1] = 1
            target = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) & allowed
            filled = self.lama.erase_regions(page, target, over_boxes)
            img[target > 0] = filled[target > 0]

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
        all_text = np.zeros(gray.shape, bool)  # every text box on the page, so widened English does not run over another one
        for b in boxes.values():
            x0, y0, x1, y1 = clamp(b, gray.shape, 3)
            all_text[y0:y1, x0:x1] = True
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
            top = False
            if members[0]["id"] in art_ids:
                # text with drawn art under part of it: keep the English on the white part, widened sideways
                crect = paper_top(rect, paper & ~taken)
                if crect != rect:
                    free = paper & ~taken
                    bounds = (limit[0], 0, limit[2], 0) if isinstance(key, int) else (0, 0, gray.shape[1], 0)
                    wrect, wfit = widen(crect, free, text, spec, bounds, target=26)
                    if wfit[3]:
                        rect, (font, lines, lh, fits) = wrect, wfit
                        note += " (kept off the art)"
                        top = True
            elif font.size < WIDE_TARGET and (rect[3] - rect[1]) > TALL * (rect[2] - rect[0]):  # a narrow column: use free paper beside it
                free = paper & ~taken
                wrect, wfit = widen(rect, free, text, spec, (limit[0] if isinstance(key, int) else 0, 0, limit[2] if isinstance(key, int) else gray.shape[1], 0))
                if wfit[0].size > font.size:
                    rect, (font, lines, lh, fits) = wrect, wfit
                    note += " (widened)"
                    top = len(lines) * lh < 0.6 * (rect[3] - rect[1])
            taken[rect[1]:rect[3], rect[0]:rect[2]] = True
            draws.append((rect, font, lines, lh, (0, 0, 0), None, top))
            report.append({"ids": [m["id"] for m in members], "status": ("rendered" if fits else "overflow: text did not fit at the minimum size") + note,
                           "kind": "plain", "font": spec["name"], "font_size": font.size, "rect": list(rect), "text": text})

        # Dark areas and over-art text: English white on black, or black with a white outline.
        if others:
            dark_px = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < DARK_LEVEL
            n, labels, stats, _ = cv2.connectedComponentsWithStats((~dark_px).astype(np.uint8), connectivity=8)
            for k in range(1, n):  # light specks left in the dark (ghost strokes) count as dark
                if stats[k, cv2.CC_STAT_AREA] < SPECK_AREA:
                    dark_px[labels == k] = True
            dark = cv2.erode(dark_px.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        for r, kind in others:
            x0, y0, x1, y1 = boxes[r["id"]]
            spec = self.secondary if is_loud(r) else self.primary
            text = r["translation"].strip()
            if kind == "inverse":
                ex, ey = max(30, x1 - x0), max(30, (y1 - y0) // 2)
                limit = clamp((x0 - ex, y0 - ey, x1 + ex, y1 + ey), gray.shape)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                rect = grow(dark & ~taken, (cx - 6, cy - 6, cx + 6, cy + 6), limit)
                if (rect[2] - rect[0]) * (rect[3] - rect[1]) < 0.5 * (x1 - x0) * (y1 - y0):
                    rect = clamp((x0, y0, x1, y1), gray.shape)  # the dark area is smaller than the lettering was: use its box
                fill, outline = (255, 255, 255), None
            else:
                rect = clamp((x0, y0, x1, y1), gray.shape, OVERLAY_PAD)
                fill, outline = (0, 0, 0), (255, 255, 255)
            font, lines, lh, fits = fit(text, spec, rect[2] - rect[0] - 2 * TEXT_PAD, rect[3] - rect[1] - 2 * TEXT_PAD)
            how = "on dark" if kind == "inverse" else "over art"
            top = False
            if font.size < WIDE_TARGET and (rect[3] - rect[1]) > TALL * (rect[2] - rect[0]):  # a narrow column: use the room beside it
                own = clamp((x0, y0, x1, y1), gray.shape, 3)
                others_text = all_text.copy()
                others_text[own[1]:own[3], own[0]:own[2]] = False
                free = (dark if kind == "inverse" else np.ones(gray.shape, bool)) & ~taken & ~others_text
                wrect, wfit = widen(rect, free, text, spec, (0, 0, gray.shape[1], 0), gray=gray)
                if wfit[0].size > font.size:
                    rect, (font, lines, lh, fits) = wrect, wfit
                    how += ", widened"
                    top = len(lines) * lh < 0.6 * (rect[3] - rect[1])
            taken[rect[1]:rect[3], rect[0]:rect[2]] = True
            draws.append((rect, font, lines, lh, fill, outline, top))
            report.append({"ids": [r["id"]], "status": ("rendered" if fits else "overflow: text did not fit at the minimum size") + f" ({how})",
                           "kind": kind, "font": spec["name"], "font_size": font.size, "rect": list(rect), "text": text})

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for (x0, y0, x1, y1), font, lines, lh, fill, outline, at_top in draws:
            top = (y0 + TEXT_PAD) if at_top else (y0 + y1) / 2 - lh * len(lines) / 2
            for k, line in enumerate(lines):
                kw = {"stroke_width": OUTLINE, "stroke_fill": outline} if outline else {}
                d.text(((x0 + x1) / 2, top + k * lh), line, font=font, fill=fill, anchor="ma", **kw)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), report
