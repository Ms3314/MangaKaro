"""Reading the text in each box with manga-ocr, plus cleanup of the boxes it reads."""
import re

import cv2

_MOCR = None


def drop_contained(regions, thresh=0.9):
    """Drop text boxes that sit (almost) entirely inside a larger text box: the larger box already covers their text."""
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])
    texts = [g for g in regions if g["type"] == "text"]
    drop = set()
    for i, a in enumerate(texts):
        for b in texts:
            if a is b or area(a["bbox"]) >= area(b["bbox"]):
                continue
            ix = max(0, min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0]))
            iy = max(0, min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1]))
            if ix * iy / area(a["bbox"]) >= thresh:
                drop.add(i)
    dropped = {id(texts[i]) for i in drop}
    return [g for g in regions if id(g) not in dropped]


def normalize_ellipses(text):
    """manga-ocr writes ellipses as runs of full-width dots (．．．); turn each run into … (one per three dots)."""
    return re.sub(r"[．.・]{2,}", lambda m: "…" * max(1, len(m.group()) // 3), text)


def read_crop(page, bbox, pad=4):
    """OCR one region of a BGR page. Returns the cleaned-up text."""
    global _MOCR
    from manga_ocr import MangaOcr
    from PIL import Image
    _MOCR = _MOCR or MangaOcr()
    H, W = page.shape[:2]
    x0, y0, x1, y1 = bbox
    crop = page[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
    return normalize_ellipses(_MOCR(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))


def read_boxes(page, texts, pad=4):
    """OCR every box on the clean page. It can only output what is in the crop, so it cannot invent text elsewhere."""
    regions = [{"id": g["label"], "source_text": read_crop(page, g["bbox"], pad)} for g in texts]
    return {"model": "manga-ocr", "ids_ok": True, "regions": regions, "raw": None}
