"""Numbered review sheets for eyeballing the renderer: a random sample of translated boxes, original crop next to the result.

Each card shows the original crop and the translated crop of one text block, with its number, the OCR text, the English and
the font size underneath. Skipped boxes are included as cards too (the Japanese stays on the page). Reading the sheets and
reporting the card numbers that look wrong gives a rough error rate, and a fixed seed keeps the sample the same between runs.

Usage:
  .venv/bin/python -m experiments.review_sheets daiboken-volume-2 --variant S7 --rendered 20 --skipped 4
Writes output/review/<variant>/sheet-1.png, sheet-2.png, ... and index.json (number -> page, ids, text, status).
"""
import argparse
import json
import random
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from mangakaro.pages import load_pages
from mangakaro.paths import ROOT

JP_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
CROP_H, PAD, CARD_W, PER_SHEET, COLS = 260, 45, 760, 8, 2


def crop(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[max(0, y0 - PAD):min(H, y1 + PAD), max(0, x0 - PAD):min(W, x1 + PAD)]


def fit_h(im, h):
    s = h / im.shape[0]
    return cv2.resize(im, (max(1, int(im.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def wrap(draw, text, font, width):
    """Wrap by measured width (works for Japanese too, where there are no spaces)."""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur]


def card(n, before, after, info):
    half = CARD_W // 2 - 6
    b, a = fit_h(before, CROP_H), fit_h(after, CROP_H)
    b, a = (cv2.resize(x, (half, int(x.shape[0] * half / x.shape[1]))) if x.shape[1] > half else x for x in (b, a))
    canvas = Image.new("RGB", (CARD_W, CROP_H + 150), "white")
    for k, im in enumerate((b, a)):
        canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (k * (half + 12), 30))
    d = ImageDraw.Draw(canvas)
    small, body = ImageFont.truetype(JP_FONT, 14), ImageFont.truetype(JP_FONT, 16)
    d.text((6, 6), f"#{n}   {info['page']}  {', '.join(info['ids'])}", font=ImageFont.truetype(JP_FONT, 16), fill=(180, 0, 0))
    d.text((half + 18, 6), "after" if not info["status"].startswith("skipped") else "after (skipped, unchanged)", font=small, fill=(110, 110, 110))
    y = CROP_H + 36
    for label, text in (("JP", info["source"]), ("EN", info["text"] or "(not translated)")):
        for i, line in enumerate(wrap(d, f"{label}  {text}", body, CARD_W - 12)[:2]):
            d.text((6, y), line, font=body, fill="black")
            y += 20
    d.text((6, y + 4), info["status"] + (f"   font {info['font']} {info['size']}px" if info.get("size") else ""), font=small, fill=(90, 90, 90))
    d.rectangle([0, 0, CARD_W - 1, canvas.height - 1], outline=(190, 190, 190))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--rendered", type=int, default=20)
    ap.add_argument("--skipped", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--like", help="index.json from an earlier run: show the same cards, in the same order, from the current render")
    ap.add_argument("--tag", default="", help="write to output/review/<variant>-<tag> instead of overwriting")
    a = ap.parse_args()

    src = ROOT / "data/intermediate/staged" / a.variant
    out = ROOT / "output/review" / (a.variant + (f"-{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / "output/rendered" / a.variant / "report.json").read_text())

    pool_ok, pool_skip, by_key = [], [], {}
    for page, blocks in report.items():
        final = json.loads((src / f"{page}.json").read_text())
        by_id = {r["id"]: r for r in final["regions"]}
        for b in blocks:
            src_text = " ".join(by_id[i]["source_text"] for i in b["ids"]).replace("\n", "")
            tr = " ".join((by_id[i]["translation"] or "") for i in b["ids"])
            row = {"page": page, "ids": b["ids"], "status": b["status"], "source": src_text, "text": b.get("text") or tr,
                   "font": b.get("font"), "size": b.get("font_size"), "box": final["boxes"][b["ids"][0]]}
            by_key[(page, tuple(b["ids"]))] = row
            if b["status"].startswith(("rendered", "overflow")):
                row["box"] = b["rect"]
                pool_ok.append(row)
            elif b["status"].startswith("skipped: not a plain"):
                pool_skip.append(row)
    if a.like:
        earlier = json.loads(Path(a.like).read_text())
        picks = [by_key[(e["page"], tuple(e["ids"]))] for _, e in sorted(earlier.items(), key=lambda kv: int(kv[0]))]
        for row in picks:
            if row["status"].startswith(("rendered", "overflow")):
                row["box"] = row["box"]
    else:
        rnd = random.Random(a.seed)
        picks = rnd.sample(pool_ok, min(a.rendered, len(pool_ok))) + rnd.sample(pool_skip, min(a.skipped, len(pool_skip)))
        rnd.shuffle(picks)

    pages = {}
    for pg in load_pages(a.folder):
        if pg.name in report:
            pages[pg.name] = pg.image
    cards, index = [], {}
    for n, info in enumerate(picks, 1):
        before = pages[info["page"]]
        after = cv2.imread(str(ROOT / "output/rendered" / a.variant / f"{info['page']}.png"))
        cards.append(card(n, crop(before, info["box"]), crop(after, info["box"]), info))
        index[n] = {k: info[k] for k in ("page", "ids", "status", "source", "text")}

    for s in range(0, len(cards), PER_SHEET):
        chunk = cards[s:s + PER_SHEET]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (CARD_W * COLS + 10 * (COLS + 1), (chunk[0].height + 10) * rows + 10), (235, 235, 235))
        for k, c in enumerate(chunk):
            sheet.paste(c, (10 + (k % COLS) * (CARD_W + 10), 10 + (k // COLS) * (c.height + 10)))
        sheet.save(out / f"sheet-{s // PER_SHEET + 1}.png")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"{len(cards)} cards -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
