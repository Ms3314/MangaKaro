"""The model stages, each one focused call, each cached as its own JSON so a later stage can be redone on its own.

  read       OCR (manga-ocr), or optionally the vision model reading a contact sheet     -> source text per box
  emotion    vision model: page image + the text -> speaker, emotion, intensity, lettering style, scene
  translate  text-only model: whole page at once -> English per box

Each stage reports `ids_ok` (did the model return exactly the boxes it was given) and retries skipped boxes.
"""
import json
import time

from .llm import OllamaTimeout, chat
from .prompts import (EMOTION_PROMPT, EMOTION_SCHEMA, EMOTIONS, READ_PROMPT, READ_SCHEMA, TRANSLATE_PROMPT, TRANSLATE_SCHEMA)

MAX_RETRIES = 2


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
    prompt = EMOTION_PROMPT.format(regions=listing, emotions=", ".join(EMOTIONS))
    ids = [g["label"] for g in texts]
    got, scene, raw, retries = {}, "", None, 0
    while True:
        extra = "" if not got else (f"\n\nYour previous answer skipped some lines. Return entries ONLY for these ids: "
                                    f"{', '.join(i for i in ids if i not in got)}. Use the exact ids as listed.")
        try:
            raw = chat(model, prompt + extra, EMOTION_SCHEMA, [annotated])
        except OllamaTimeout:  # a stalled call: count it as a try and go round again
            raw = None
        try:
            d = json.loads(raw)
            scene = scene or d["scene"]
            got.update({r["id"]: r for r in d["regions"] if r.get("id") in ids and r["id"] not in got})  # mangled ids are ignored
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        if len(got) == len(ids) or retries >= MAX_RETRIES:
            break
        retries += 1
    return {"model": model, "ids_ok": len(got) == len(ids), "retries": retries, "scene": scene,
            "regions": [got[i] for i in ids if i in got] or None, "raw": None if got else raw}


CHUNK = 4  # lines per request when a whole-page translation stalls


def _translate_ids(model, prompt, want, all_ids):
    """Ask for the lines `want` (a subset of all_ids). Returns {id: region}; a timed-out or unparseable call returns {}."""
    extra = "" if len(want) == len(all_ids) else (
        f"\n\nTranslate ONLY these ids: {', '.join(want)}. The other lines are context. Return exactly one entry per listed id.")
    try:
        raw = chat(model, prompt + extra, TRANSLATE_SCHEMA)
    except OllamaTimeout:
        return {}
    regions, _ = parse(raw, want)
    return {r["id"]: r for r in regions or [] if r.get("id") in want}


def stage_translate(model, texts, read, emo):
    text = {r["id"]: r["source_text"] for r in read["regions"]}
    meta = {r["id"]: r for r in (emo["regions"] or [])}
    lines = "\n".join(
        f"{g['label']} | {meta.get(g['label'], {}).get('speaker', '?')} | "
        f"{meta.get(g['label'], {}).get('emotion', '?')}, {meta.get(g['label'], {}).get('intensity', '?')} | "
        f"{text.get(g['label'], '')}" for g in texts)
    prompt = TRANSLATE_PROMPT.format(scene=emo["scene"], lines=lines)
    ids = [g["label"] for g in texts]
    got = _translate_ids(model, prompt, ids, ids)
    retries = 0
    if not got and len(ids) > CHUNK:  # the whole page produced nothing (stalled or looped): try a few lines at a time
        for i in range(0, len(ids), CHUNK):
            got.update(_translate_ids(model, prompt, ids[i:i + CHUNK], ids))
    while len(got) < len(ids) and retries < MAX_RETRIES:  # model skipped some ids: ask again for just those
        retries += 1
        got.update(_translate_ids(model, prompt, [i for i in ids if i not in got], ids))
    for i in [i for i in ids if i not in got]:  # last resort: one line per request, so one bad line cannot take the others down
        got.update(_translate_ids(model, prompt, [i], ids))
    return {"model": model, "ids_ok": len(got) == len(ids), "retries": retries,
            "regions": [got[i] for i in ids if i in got] or None, "raw": None}


def merge(name, texts, read, emo, tr, skip=None):
    skip = skip or {}
    """Final per-page JSON, same fields as the one-shot translate_pages.py output."""
    r = {x["id"]: x for x in read["regions"] or []}
    e = {x["id"]: x for x in emo["regions"] or []}
    t = {x["id"]: x for x in tr["regions"] or []}
    regions = [{"id": g["label"], "source_text": r.get(g["label"], {}).get("source_text", ""),
                "speaker": e.get(g["label"], {}).get("speaker"), "emotion": e.get(g["label"], {}).get("emotion"), "intensity": e.get(g["label"], {}).get("intensity"),
                "translation": "" if g["label"] in skip else t.get(g["label"], {}).get("translation", ""), "skip": skip.get(g["label"]),
                "style": e.get(g["label"], {}).get("style"), "flag": t.get(g["label"], {}).get("flag")} for g in texts]
    return {"page": name, "models": {"read": read["model"], "emotion": emo["model"], "translate": tr["model"]},
            "seconds": {"read": read["seconds"], "emotion": emo["seconds"], "translate": tr["seconds"]},
            "ids_ok": read["ids_ok"] and emo["ids_ok"] and tr["ids_ok"], "scene": emo["scene"],
            "boxes": {g["label"]: g["bbox"] for g in texts}, "regions": regions}
