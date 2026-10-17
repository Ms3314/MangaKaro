"""One-shot read + emotion + translate in a single vision-model call (experiment S0, kept for comparison).

For each page: run the panels_text detector, draw the boxes with ID tags, send that image (and optionally a contact sheet of
close-ups) to an Ollama vision model, and save its per-ID JSON (text, emotion, translation, style) to
data/intermediate/translate/<model>/<page>.json. The staged pipeline in the mangakaro package replaced this approach.

Usage:
  .venv/bin/python -m experiments.translate_pages daiboken-volume-2 --pages 10-11 --model qwen3-vl:8b
  .venv/bin/python -m experiments.translate_pages daiboken-volume-2 --pages 10 --dry-run   # detection only, no model call
(--pages counts the files in the folder from 1; it no longer matches the number in the file name.)
"""
import argparse
import json
import time
import urllib.error
import urllib.request

import cv2

from mangakaro import llm
from mangakaro.detect import PanelsText, assign_ids, draw
from mangakaro.pages import contact_sheet, load_pages
from mangakaro.paths import ROOT
from mangakaro.prompts import EMOTIONS

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


def ask(model, images, regions, think):
    listing = "\n".join(f"- {g['label']}: {g['bbox']}" for g in regions)
    body = {
        "model": model, "stream": False, "format": SCHEMA, "think": think,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "user", "content": PROMPT.format(regions=listing, emotions=", ".join(EMOTIONS)) + (CROPS_NOTE if len(images) > 1 else ""),
                      "images": [llm.b64_jpg(im) for im in images]}],
    }
    req = urllib.request.Request(f"{llm.DEFAULT_URL}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--pages", help="e.g. 10-11 (1-based position in the folder)")
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--think", action="store_true", help="let the model think first (slower)")
    ap.add_argument("--no-crops", action="store_true", help="send only the annotated page (baseline)")
    ap.add_argument("--dry-run", action="store_true", help="detect + annotate only, no model call")
    a = ap.parse_args()

    det = PanelsText()
    tag = a.model.replace(":", "_").replace("/", "_") + ("" if a.no_crops else "_crops")
    out = ROOT / "data/intermediate/translate" / ("dry_run" if a.dry_run else tag)
    out.mkdir(parents=True, exist_ok=True)

    for pg in load_pages(a.folder, a.pages):
        name, page = pg.name, pg.image
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
