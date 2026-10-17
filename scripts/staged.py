"""Staged pipeline: read -> emotion -> translate, one focused model call per stage.

Stages (each cached as its own JSON, so a later stage can be re-run without redoing earlier ones):
  1. read       page + crop sheet  -> source_text per id           (vision model)
  2. emotion    page + stage-1 text -> emotion, intensity, style, scene note  (vision model)
  3. translate  text only, whole page at once -> English per id    (text model, no image)

Output: data/intermediate/staged/<variant>/<page>/{boxes,read,emotion,translate}.json and <page>.json
(final file uses the same fields as translate_pages.py so it can be compared with the one-shot S0 results).

Usage:
  .venv/bin/python scripts/staged.py daiboken-volume-2 --pages 10 --variant S1 --read-model qwen3.5:4b --emotion-model qwen3.5:4b --translate-model qwen3.5:4b
  .venv/bin/python scripts/staged.py daiboken-volume-2 --pages 10 --variant S2 --translate-model qwen3:8b
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from compare_detectors import PanelsText, assign_ids, draw
from translate_pages import EMOTIONS, OLLAMA, b64_jpg, contact_sheet, load_pages, shrink

ROOT = Path(__file__).resolve().parent.parent

READ_SCHEMA = {"type": "object", "properties": {"regions": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "source_text": {"type": "string"}},
    "required": ["id", "source_text"]}}}, "required": ["regions"]}

EMOTION_SCHEMA = {"type": "object", "properties": {
    "scene": {"type": "string"},
    "regions": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "speaker": {"type": "string"},
        "emotion": {"type": "string", "enum": EMOTIONS},
        "intensity": {"type": "number"},
        "style": {"type": "object", "properties": {
            "weight": {"type": "string", "enum": ["light", "normal", "bold"]},
            "slant": {"type": "string", "enum": ["upright", "italic"]},
            "size": {"type": "string", "enum": ["small", "medium", "large"]},
            "shape": {"type": "string", "enum": ["normal", "jagged", "wavy", "outlined"]}},
            "required": ["weight", "slant", "size", "shape"]},
    }, "required": ["id", "speaker", "emotion", "intensity", "style"]}}}, "required": ["scene", "regions"]}

TRANSLATE_SCHEMA = {"type": "object", "properties": {"regions": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "translation": {"type": "string"},
                                     "flag": {"type": ["string", "null"]}},
    "required": ["id", "translation", "flag"]}}}, "required": ["regions"]}

READ_PROMPT = """This is one manga page. Red boxes mark text regions, each with a numbered tag (like 2.1) at its top-left corner, \
outside the box: the tag is a label, not part of the text. Regions and pixel boxes [x1,y1,x2,y2]:

{regions}

Image 2 is a contact sheet with a close-up of the inside of each box, labelled with its id. Use it to read the exact characters.

Transcribe the text of EVERY listed id exactly as written, and nothing else. Japanese is usually vertical: read each column top to \
bottom, columns right to left. Ignore furigana (tiny reading hints beside kanji). Do not correct, translate or guess characters you \
cannot see. If a box has no readable text, use "". Return one entry per id, in the same order."""

EMOTION_PROMPT = """This is one manga page. Red boxes mark text regions with numbered tags. The text in each region has already been \
transcribed:

{regions}

Do NOT translate. For every id return:
- speaker: who is speaking, in a few English words (e.g. "boy with goggles", "narration", "old man"), or "sound effect"
- emotion: one of {emotions}, judged from the character's face, pose and the scene as well as the words
- intensity: 0.0 (calm) to 1.0 (extreme)
- style: how the ORIGINAL lettering looks: weight (light/normal/bold), slant (upright/italic), size (small/medium/large relative to \
the rest of the page), shape (normal/jagged for shouting/wavy for trembling or sarcasm/outlined for big sound effects)
Also give "scene": one short English sentence on what is happening on this page. Return one entry per id, in the same order."""

TRANSLATE_PROMPT = """You are translating Japanese manga dialogue into natural English for speech bubbles.

Scene: {scene}

Lines in reading order (id | speaker | emotion, intensity | Japanese):
{lines}

Translate every line. Read the page as one conversation: keep each speaker's voice and pronouns consistent, and let later lines \
clarify earlier ones. Match the emotion. Keep it short enough for a speech bubble. The Japanese was transcribed from an image and may \
contain small character errors: if a line looks garbled, translate the most plausible meaning from context. Use "" for an empty line. \
flag: null, or a short reason if the text is obscene or problematic. Return one entry per id, in the same order."""


def chat(model, prompt, schema, images=(), think=False):
    body = {"model": model, "stream": False, "format": schema, "think": think,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "user", "content": prompt, **({"images": [b64_jpg(i) for i in images]} if images else {})}]}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"]


def run_stage(path, fn):
    """Run fn() once and cache the result; later runs reuse the file."""
    if path.exists():
        return json.loads(path.read_text())
    t = time.perf_counter()
    result = fn()
    result["seconds"] = round(time.perf_counter() - t, 1)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def parse(raw, ids, key="regions"):
    """Parse model JSON and report whether it covers exactly the detector ids."""
    try:
        regions = json.loads(raw)[key] if key else json.loads(raw)
    except (json.JSONDecodeError, KeyError):
        return None, False
    return regions, {r.get("id") for r in regions} == set(ids)


def stage_read(model, annotated, sheet, texts):
    listing = "\n".join(f"- {g['label']}: {g['bbox']}" for g in texts)
    raw = chat(model, READ_PROMPT.format(regions=listing), READ_SCHEMA, [annotated, sheet])
    regions, ok = parse(raw, [g["label"] for g in texts])
    return {"model": model, "ids_ok": ok, "regions": regions, "raw": None if regions else raw}


def stage_emotion(model, annotated, texts, read):
    by_id = {r["id"]: r["source_text"] for r in read["regions"]}
    listing = "\n".join(f"- {g['label']}: {by_id.get(g['label'], '')!r}" for g in texts)
    raw = chat(model, EMOTION_PROMPT.format(regions=listing, emotions=", ".join(EMOTIONS)), EMOTION_SCHEMA, [annotated])
    try:
        d = json.loads(raw)
        ok = {r.get("id") for r in d["regions"]} == {g["label"] for g in texts}
        return {"model": model, "ids_ok": ok, "scene": d["scene"], "regions": d["regions"], "raw": None}
    except (json.JSONDecodeError, KeyError):
        return {"model": model, "ids_ok": False, "scene": "", "regions": None, "raw": raw}


def stage_translate(model, texts, read, emo):
    text = {r["id"]: r["source_text"] for r in read["regions"]}
    meta = {r["id"]: r for r in (emo["regions"] or [])}
    lines = "\n".join(
        f"{g['label']} | {meta.get(g['label'], {}).get('speaker', '?')} | "
        f"{meta.get(g['label'], {}).get('emotion', '?')}, {meta.get(g['label'], {}).get('intensity', '?')} | "
        f"{text.get(g['label'], '')}" for g in texts)
    raw = chat(model, TRANSLATE_PROMPT.format(scene=emo["scene"], lines=lines), TRANSLATE_SCHEMA)
    regions, ok = parse(raw, [g["label"] for g in texts])
    return {"model": model, "ids_ok": ok, "regions": regions, "raw": None if regions else raw}


def merge(name, texts, read, emo, tr):
    """Final per-page JSON, same fields as the one-shot translate_pages.py output."""
    r = {x["id"]: x for x in read["regions"] or []}
    e = {x["id"]: x for x in emo["regions"] or []}
    t = {x["id"]: x for x in tr["regions"] or []}
    regions = [{"id": g["label"], "source_text": r.get(g["label"], {}).get("source_text", ""),
                "emotion": e.get(g["label"], {}).get("emotion"), "intensity": e.get(g["label"], {}).get("intensity"),
                "translation": t.get(g["label"], {}).get("translation", ""),
                "style": e.get(g["label"], {}).get("style"), "flag": t.get(g["label"], {}).get("flag")} for g in texts]
    return {"page": name, "models": {"read": read["model"], "emotion": emo["model"], "translate": tr["model"]},
            "seconds": {"read": read["seconds"], "emotion": emo["seconds"], "translate": tr["seconds"]},
            "ids_ok": read["ids_ok"] and emo["ids_ok"] and tr["ids_ok"], "scene": emo["scene"],
            "boxes": {g["label"]: g["bbox"] for g in texts}, "regions": regions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--pages", nargs="*", help="image numbers, e.g. 10 (matches the _00NN suffix)")
    ap.add_argument("--variant", required=True, help="label for the output folder, e.g. S1")
    ap.add_argument("--read-model", default="qwen3.5:4b")
    ap.add_argument("--emotion-model", default="qwen3.5:4b")
    ap.add_argument("--translate-model", default="qwen3.5:4b")
    ap.add_argument("--redo", nargs="*", default=[], choices=["read", "emotion", "translate"],
                    help="ignore the cache for these stages and every stage after them")
    a = ap.parse_args()

    out = ROOT / "data/intermediate/staged" / a.variant
    det = PanelsText()
    order = ["read", "emotion", "translate"]
    redo = set(order[min(order.index(s) for s in a.redo):]) if a.redo else set()

    for name, page in load_pages(a.folder, a.pages):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for s in redo:
            (d / f"{s}.json").unlink(missing_ok=True)
        page = shrink(page)
        regions = assign_ids(det(page))
        texts = [g for g in regions if g["type"] == "text"]
        if not texts:
            print(f"{name}: no text boxes")
            continue
        annotated, sheet = draw(page, texts), contact_sheet(page, texts)
        cv2.imwrite(str(d / "annotated.jpg"), annotated)
        cv2.imwrite(str(d / "sheet.jpg"), sheet)
        try:
            read = run_stage(d / "read.json", lambda: stage_read(a.read_model, annotated, sheet, texts))
            emo = run_stage(d / "emotion.json", lambda: stage_emotion(a.emotion_model, annotated, texts, read))
            tr = run_stage(d / "translate.json", lambda: stage_translate(a.translate_model, texts, read, emo))
        except urllib.error.URLError as e:
            print(f"{name}: Ollama error ({e}). Is `ollama serve` running and the model pulled?")
            return
        final = merge(name, texts, read, emo, tr)
        (out / f"{name}.json").write_text(json.dumps(final, ensure_ascii=False, indent=1))
        s = final["seconds"]
        print(f"{name}: {len(texts)} boxes, read {s['read']}s / emotion {s['emotion']}s / translate {s['translate']}s, ids_ok={final['ids_ok']}")


if __name__ == "__main__":
    main()
