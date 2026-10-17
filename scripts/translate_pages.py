"""Phase 2 harness: page -> detector boxes -> vision LLM (Ollama) -> per-ID JSON (text, emotion, translation, style).

For each input image (two-page spreads are split into right page, left page):
  1. run the panels_text detector, keep text boxes, give them reading-order IDs (2.1, 2.2, ...)
  2. draw the boxes with big ID tags and send that image + the ID/box list to the model
  3. save the model's JSON to data/intermediate/translate/<model>/<page>.json (+ annotated image)

Usage:
  .venv/bin/python scripts/translate_pages.py daiboken-volume-2 --pages 10 11 --model qwen3-vl:8b
  .venv/bin/python scripts/translate_pages.py daiboken-volume-2 --pages 10 --dry-run   # detection only, no model call
"""
import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from compare_detectors import PanelsText, assign_ids, draw

ROOT = Path(__file__).resolve().parent.parent
OLLAMA = "http://localhost:11434/api/chat"
MAX_SIDE = 1536

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted",
            "embarrassed", "excited", "sarcastic", "determined", "pained", "other"]

SCHEMA = {
    "type": "object",
    "properties": {"regions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "source_text": {"type": "string"},
            "emotion": {"type": "string", "enum": EMOTIONS},
            "intensity": {"type": "number"},
            "translation": {"type": "string"},
            "style": {"type": "object", "properties": {
                "weight": {"type": "string", "enum": ["light", "normal", "bold"]},
                "slant": {"type": "string", "enum": ["upright", "italic"]},
                "size": {"type": "string", "enum": ["small", "medium", "large"]},
                "shape": {"type": "string", "enum": ["normal", "jagged", "wavy", "outlined"]},
            }, "required": ["weight", "slant", "size", "shape"]},
            "flag": {"type": ["string", "null"]},
        },
        "required": ["id", "source_text", "emotion", "intensity", "translation", "style", "flag"],
    }}},
    "required": ["regions"],
}

PROMPT = """You are translating a manga page into English.

The image is one manga page. Red boxes mark text regions. Each box has a numbered tag (like 2.1) attached to its top-left corner, \
just outside the box: the tag is a label, not part of the text. The regions are listed below in reading order with their pixel boxes [x1,y1,x2,y2] \
on this image:

{regions}

For EVERY listed id, read the text inside its red box and return:
- source_text: the text exactly as written. Japanese is usually vertical: read each column top to bottom, columns right to left. \
Ignore furigana (the tiny reading hints beside kanji). If the box has no readable text, use "".
- emotion: how the speaker/narration feels, one of {emotions}. Use the character's face, pose and the scene, not only the words.
- intensity: 0.0 (calm) to 1.0 (extreme).
- translation: natural English that fits a speech bubble, matching the tone. Keep it short. "" if there is no text.
- style: how the ORIGINAL lettering looks, so the English can be typeset to match: weight (light/normal/bold), slant (upright/italic), \
size (small/medium/large relative to other text on the page), shape (normal/jagged for shouting/wavy for trembling or sarcasm/outlined for big sound effects).
- flag: null, or a short reason if the text is obscene or problematic.

Return one entry per id, in the same order, and nothing else."""


CROPS_NOTE = """

You get two images. Image 1 is the full page with the red boxes. Image 2 is a contact sheet: a close-up of the inside of each box, \
labelled with its id in red. Use image 2 to read the exact characters (do not guess characters you cannot see) and image 1 for the scene, \
the speaker's face and the emotion."""


def load_pages(folder, wanted):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if wanted:
        files = [f for f in files if any(re.search(rf"_0*{n}\.", f.name) for n in wanted)]
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        if w / h > 1.1:  # two-page spread: right page first (manga reading order)
            yield f"{f.stem}_R", img[:, w // 2:]
            yield f"{f.stem}_L", img[:, : w // 2]
        else:
            yield f.stem, img


def shrink(img):
    s = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


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


def b64_jpg(im):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()


def ask(model, images, regions, think):
    listing = "\n".join(f"- {g['label']}: {g['bbox']}" for g in regions)
    b64 = [b64_jpg(im) for im in images]
    body = {
        "model": model, "stream": False, "format": SCHEMA, "think": think,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "user", "content": PROMPT.format(regions=listing, emotions=", ".join(EMOTIONS)) + (CROPS_NOTE if len(images) > 1 else ""),
                      "images": b64}],
    }
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--pages", nargs="*", help="image numbers, e.g. 10 11 (matches the _00NN suffix)")
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--think", action="store_true", help="let the model think first (slower)")
    ap.add_argument("--no-crops", action="store_true", help="send only the annotated page (baseline)")
    ap.add_argument("--dry-run", action="store_true", help="detect + annotate only, no model call")
    a = ap.parse_args()

    det = PanelsText()
    tag = a.model.replace(":", "_").replace("/", "_") + ("" if a.no_crops else "_crops")
    out = ROOT / "data/intermediate/translate" / ("dry_run" if a.dry_run else tag)
    out.mkdir(parents=True, exist_ok=True)

    for name, page in load_pages(a.folder, a.pages):
        page = shrink(page)
        regions = assign_ids(det(page))
        texts = [g for g in regions if g["type"] == "text"]
        annotated = draw(page, texts)
        sheet = None if a.no_crops or not texts else contact_sheet(page, texts)
        if sheet is not None:
            cv2.imwrite(str(out / f"{name}.sheet.jpg"), sheet)
        cv2.imwrite(str(out / f"{name}.annotated.jpg"), annotated)
        if a.dry_run or not texts:
            print(f"{name}: {len(texts)} text boxes (no model call)")
            continue
        t = time.perf_counter()
        try:
            resp = ask(a.model, [annotated] + ([sheet] if sheet is not None else []), texts, a.think)
        except urllib.error.HTTPError as e:
            print(f"{name}: Ollama error {e.code}: {e.read().decode()[:300]}")
            return
        except urllib.error.URLError as e:
            print(f"{name}: cannot reach Ollama ({e.reason}). Is `ollama serve` running?")
            return
        dt = time.perf_counter() - t
        raw = resp["message"]["content"]
        try:
            parsed = json.loads(raw)["regions"]
            valid = {g["label"] for g in texts} == {r.get("id") for r in parsed}
        except (json.JSONDecodeError, KeyError):
            parsed, valid = None, False
        (out / f"{name}.json").write_text(json.dumps(
            {"page": name, "model": a.model, "seconds": round(dt, 1), "json_ok": parsed is not None,
             "ids_match_detector": valid, "boxes": {g["label"]: g["bbox"] for g in texts},
             "regions": parsed, "raw": None if parsed is not None else raw}, ensure_ascii=False, indent=1))
        print(f"{name}: {len(texts)} boxes, {dt:.0f}s, json_ok={parsed is not None}, ids_match={valid}")


if __name__ == "__main__":
    main()
