"""Check a finished page: is any Japanese still on it?

Run the text detector on the translated page, ignore the English we placed ourselves, read everything else with the OCR, and
keep what is Japanese. Each leftover is labelled with a likely cause:

  skipped         the renderer left it on purpose (not a plain bubble: text over art, sound effects, dark backgrounds)
  erase_failed    it sits where we translated, so the original lettering was not fully erased
  not_detected    nothing found it before (too small, or the detectors missed it)

No language model is involved, so the answer is a list of boxes with the text read from each, ready to be sent back through
the pipeline. The detector runs on the whole page and on two overlapping zoomed halves, with a lower threshold than the first
pass, because tiny text is what gets missed.
"""
import re

import numpy as np

from .detect import ComicTextDetector
from .ocr import JAPANESE, read_crop



def is_japanese(text, min_chars=2):
    return len(JAPANESE.findall(text)) >= min_chars


def _overlap(a, b):
    """Share of box a that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def covered_by(box, others, share=0.5):
    return any(_overlap(box, o) >= share for o in others)


def detect_tiled(detector, img, overlap=0.15):
    """Boxes from the whole page plus two overlapping halves (each is shown to the model larger than in the whole page)."""
    H = img.shape[0]
    cut = int(H * (0.5 + overlap / 2))
    start = int(H * (0.5 - overlap / 2))
    boxes = [g["bbox"] for g in detector(img)]
    for y0, y1 in ((0, cut), (start, H)):
        for g in detector(img[y0:y1]):
            x0, a, x1, b = g["bbox"]
            boxes.append([x0, a + y0, x1, b + y0])
    keep = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):  # larger first; drop what a larger box already holds
        if not covered_by(b, keep, 0.8):
            keep.append(b)
    return keep


class Checker:
    def __init__(self, min_side=16, detector=None):
        self.detector = detector or ComicTextDetector(conf=0.2)  # lower threshold than the first pass
        self.min_side = min_side

    def page(self, rendered, results, report):
        """rendered: the finished BGR page. results: the page's result JSON. report: the renderer's report for the page.
        Returns a list of leftovers: {"bbox", "text", "cause"}."""
        ours = [b["rect"] for b in report if "rect" in b]  # where English was typeset
        skipped = [results["boxes"][i] for b in report if b["status"].startswith("skipped") for i in b["ids"]]
        typeset_src = [results["boxes"][i] for b in report if not b["status"].startswith("skipped") for i in b["ids"]]
        out = []
        for box in detect_tiled(self.detector, rendered):
            x0, y0, x1, y1 = box
            if x1 - x0 < self.min_side or y1 - y0 < self.min_side or covered_by(box, ours, 0.5):
                continue
            text = read_crop(rendered, box)
            if not is_japanese(text):
                continue
            cause = ("skipped" if covered_by(box, skipped, 0.3) else
                     "erase_failed" if covered_by(box, typeset_src, 0.3) else "not_detected")
            out.append({"bbox": [int(v) for v in box], "text": text, "cause": cause})
        return out
