"""Getting pages in: a PDF or a folder of images, spread splitting and shrinking.

Every page the models see (and every page written to the output) is at most MAX_SIDE pixels on its longest side.
Two-page spreads (wider than tall) are split into the right page then the left page, the manga reading order.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pdfio import pdf_page_count, pdf_pages

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIDE = 1536
SPREAD_RATIO = 1.1  # width / height above this counts as a two-page spread


@dataclass
class Page:
    name: str            # file-safe, used for the work directory, e.g. p0003_R or download_1_
    image: np.ndarray    # BGR
    out_name: str = ""   # file name (without extension) for the output image: the input file's own name, e.g. "download (1)"

    def __post_init__(self):
        self.out_name = self.out_name or self.name


def shrink(img, max_side=MAX_SIDE):
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def parse_pages(spec, total):
    """'1-3,7' -> {1, 2, 3, 7} (1-based). None or '' means every page."""
    if not spec:
        return set(range(1, total + 1))
    out = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b or a)
        out.update(range(lo, hi + 1))
    return {p for p in out if 1 <= p <= total}


def split_spread(name, img, split=True):
    """[(name, image)]: one entry for a normal page, right then left for a spread."""
    h, w = img.shape[:2]
    if split and w / h > SPREAD_RATIO:
        return [(f"{name}_R", img[:, w // 2:]), (f"{name}_L", img[:, : w // 2])]
    return [(name, img)]


def natural_key(path):
    """Sort page2 before page10: split the name into text and numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]


def resolve_source(inputs):
    """What was given on the command line -> ("pdf", Path) or ("images", [Path, ...] in page order).

    One .pdf; or one folder of images; or several image files (shell globs like pages/*.png work)."""
    paths = [Path(x).expanduser() for x in (inputs if isinstance(inputs, (list, tuple)) else [inputs])]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
    if len(paths) == 1 and paths[0].is_file() and paths[0].suffix.lower() == ".pdf":
        return "pdf", paths[0]
    if len(paths) == 1 and paths[0].is_dir():
        files = sorted((f for f in paths[0].iterdir() if f.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
        if not files:
            raise ValueError(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {paths[0]}")
        return "images", files
    bad = [p for p in paths if not (p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)]
    if bad:
        raise ValueError("give one PDF, one folder of images, or several image files; not: " + ", ".join(map(str, bad)))
    return "images", sorted(paths, key=natural_key)


def count_pages(source):
    kind, what = resolve_source(source)
    return pdf_page_count(what) if kind == "pdf" else len(what)


def load_pages(source, pages=None, split=True, max_side=MAX_SIDE):
    """Yield Page objects from a PDF, a folder of images or a list of image files. `pages` is a 1-based range such as '1-3,7'."""
    kind, what = resolve_source(source)
    if kind == "pdf":
        wanted = parse_pages(pages, pdf_page_count(what))
        for n, img in pdf_pages(what, wanted, height=max_side):
            for name, part in split_spread(f"p{n:04d}", img, split):
                yield Page(name, shrink(part, max_side))
        return
    wanted = parse_pages(pages, len(what))
    seen = {}
    for n, f in enumerate(what, 1):
        if n not in wanted:
            continue
        img = cv2.imread(str(f))
        if img is None:
            raise ValueError(f"cannot read image {f}")
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f.stem)  # file-safe: names become folder and file names in the work directory
        own = f.stem
        seen[stem] = seen.get(stem, 0) + 1
        if seen[stem] > 1:  # two files with the same name from different folders
            stem, own = f"{stem}_{seen[stem]}", f"{own}_{seen[stem]}"
        for (name, part), (suffix, _) in zip(split_spread(stem, img, split), split_spread("", img, split)):
            yield Page(name, shrink(part, max_side), own + suffix)


def contact_sheet(page, regions, row_h=300, width=1400, pad=0.08):
    """One image with a labelled close-up of every box (cut from the clean page, so no overlay on the text)."""
    H, W = page.shape[:2]
    tiles = []
    for g in regions:
        x0, y0, x1, y1 = g["bbox"]
        px, py = int((x1 - x0) * pad) + 6, int((y1 - y0) * pad) + 6
        crop = page[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)]
        s = min(row_h / crop.shape[0], (width - 8) / crop.shape[1])  # very wide boxes: shrink to fit the sheet
        crop = cv2.resize(crop, (max(60, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
        tile = cv2.copyMakeBorder(crop, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, g["label"], (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        tile = cv2.copyMakeBorder(tile, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        tiles.append(tile)
    rows, cur, w = [], [], 0
    for t in tiles:  # shelf-pack left to right
        if cur and w + t.shape[1] > width:
            rows.append(cur)
            cur, w = [], 0
        cur.append(t)
        w += t.shape[1]
    rows.append(cur)
    out = []
    for r in rows:
        row = cv2.hconcat([cv2.copyMakeBorder(t, 0, row_h + 52 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in r])
        out.append(cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    return cv2.vconcat(out)
