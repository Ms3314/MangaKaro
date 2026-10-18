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


# hiragana, katakana, CJK ideographs, half-width katakana
JAPANESE = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")
BIG_BOX = 100_000      # px² at the 1536 px page size: bigger than any single speech bubble in the test pages
SPARSE = 8_000         # px² of box per character read: a real bubble is about 2,500 to 6,000
HUGE_BOX = 300_000     # a box this big holds several blocks of text whatever the OCR returned


def triage(texts, read):
    """Decide which boxes the models should not touch. Returns {label: reason} for the boxes to leave as they are.

    no Japanese: the OCR found no kana or kanji but letters or digits (a Latin logo, a page number, a page number), so there is nothing to translate.
    unreliable read: the box is far too big for the characters read from it (a block of prose, or big title lettering).
    manga-ocr reads one line or bubble; on a big box it returns a short, invented sentence, and erasing the box
    then replaces the real text with it."""
    by_id = {r["id"]: r["source_text"] for r in read["regions"]}
    skip = {}
    for g in texts:
        text = by_id.get(g["label"], "")
        x0, y0, x1, y1 = g["bbox"]
        area = (x1 - x0) * (y1 - y0)
        if not JAPANESE.search(text) and (not text.strip() or any(c.isalnum() for c in text)):  # punctuation-only bubbles are real
            skip[g["label"]] = "no Japanese in the box"
        elif area >= HUGE_BOX or (area >= BIG_BOX and area / max(1, len(text)) >= SPARSE):
            skip[g["label"]] = "unreliable read: box too big for what was read, needs review"
    return skip


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
