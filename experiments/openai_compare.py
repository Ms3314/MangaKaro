"""Side-by-side comparison of ways to translate the same manga pages, with the cost and the end-to-end time of each.

  A        one image model: an OpenAI image-edit model gets the page and is asked to return it translated (no pipeline)
  B-<m>    our boxes + manga-ocr (local), OpenAI model <m> for emotion and translation, our renderer
  C-<m>    like B, but OpenAI model <m> also reads the Japanese (instead of manga-ocr)
  D-local  everything local: the same detectors and OCR, Gemma through Ollama for emotion and translation, our renderer

Every pipeline arm (B, C, D) is run from scratch in one go, with no cached detection, OCR or model results, so each arm's time covers
the whole path: load, detect, read, emotion, translate, erase, typeset. The time of each page is the gap between page finishes
(the first page also pays for loading the models, as a real run does). The check for left-over Japanese is off in every arm.
All arms use the same fonts (from the earlier local run). The Ollama model is warmed up once before D is timed.

Run from the project root, with the key in your environment (it is never written to disk), one command per line:
  export OPENAI_API_KEY="..."
  .venv/bin/python -m experiments.openai_compare --dry-run
  .venv/bin/python -m experiments.openai_compare --max-usd 1.5

Defaults: 4 pages of test-123 (the pages must exist in manga/output/test-123/work). A hard spending cap stops every API arm. The image arm
runs one page first and stops if the projected cost of the rest would pass the cap. Prices are in mangakaro/openai_backend.py.
Close other heavy programs while it runs, or the local timings are not comparable. The judging sheets hide which arm is which
(random letters, 4 panels per row); the key is saved beside them in key.json. Results go to manga/output/compare-openai-v2/."""
import argparse
import base64
import json
import random
import shutil
import sys
import time

import cv2
import numpy as np

from mangakaro import llm
from mangakaro import openai_backend as ob
from mangakaro.paths import MANGA
from mangakaro.pipeline import Settings, run

SRC = MANGA / "output" / "test-123" / "work"
OUT = MANGA / "output" / "compare-openai-v2"
PAGES = ["006_L", "005_L", "009_L", "002_R"]  # clean, over art, mixed, hard
LOCAL = Settings()  # the default local models
PROMPT_A = ("This is a page of a Japanese manga. Translate all the Japanese text into natural English and return the same page. "
            "Keep the drawing, panels, speech bubbles and layout exactly as they are. Replace the Japanese lettering in place with "
            "English lettering that fits inside each bubble. Do not add anything else.")


def check_models(names, image_model):
    have = {m.id for m in ob.client().models.list()}
    missing = [m for m in list(names) + ([image_model] if image_model else []) if m not in have]
    if missing:
        sys.exit(f"these model ids are not available to your key: {', '.join(missing)}\n"
                 f"models that look relevant: {', '.join(sorted(m for m in have if m.startswith(('gpt-5', 'gpt-6', 'gpt-image', 'gpt-4.1')))[:30])}")


def run_pipeline_arm(label, settings, pages, fonts):
    """One pipeline arm from scratch. Returns the total wall time and the time each page took."""
    work, out = OUT / label / "work", OUT / label / "pages"
    shutil.rmtree(work, ignore_errors=True)  # nothing cached: every stage runs
    s = settings
    s.split_spreads, s.check, s.fonts_file = False, False, fonts
    stamps = []

    def log(m):
        if "->" in m:  # one line per finished page
            stamps.append(time.perf_counter())
        print(f"  [{label}] {m}", flush=True)

    t0 = time.perf_counter()
    summary = run([OUT / "input" / f"{p}.png" for p in pages], out, work, s, log=log)
    total = time.perf_counter() - t0
    per = [round(b - a, 1) for a, b in zip([t0] + stamps, stamps)]
    return {"total_seconds": round(total, 1), "seconds_per_page": dict(zip(pages, per)) if len(per) == len(pages) else per,
            "first_page_includes_loading_models": True, "pages_failed": summary["pages_failed"]}


def run_arm_a(model, pages):
    """Arm A: the page goes to the image-edit model and a picture comes back."""
    label = f"A-{model}"
    ob.LEDGER.arm = label
    prices = ob.IMAGE_PRICES[model]
    out = OUT / label / "pages"
    out.mkdir(parents=True, exist_ok=True)
    per, spent_first, t_all = {}, 0.0, time.perf_counter()
    for i, p in enumerate(pages):
        ob.LEDGER.check()
        if i == 1 and spent_first * (len(pages) - 1) + ob.LEDGER.spent > ob.LEDGER.max_usd:
            print(f"  [{label}] stopping: the remaining pages would pass the cap at ${spent_first:.3f} per page")
            break
        t = time.perf_counter()
        with open(OUT / "input" / f"{p}.png", "rb") as f:
            r = ob.client().images.edit(model=model, image=f, prompt=PROMPT_A, size="1024x1536", quality="medium")
        (out / f"{p}.png").write_bytes(base64.b64decode(r.data[0].b64_json))
        u = getattr(r, "usage", None)
        if u is None:
            print("  [A] warning: no usage in the response, cost not counted for this call")
            usd, tin, tout = 0.0, 0, 0
        else:
            d = getattr(u, "input_tokens_details", None)
            txt, img = (getattr(d, "text_tokens", 0), getattr(d, "image_tokens", 0)) if d else (u.input_tokens, 0)
            tin, tout = u.input_tokens, u.output_tokens
            usd = (txt * prices[0] + img * prices[1] + tout * prices[2]) / 1e6
        per[p] = round(time.perf_counter() - t, 1)
        ob.LEDGER.add(model, tin, tout, usd, per[p])
        print(f"  [{label}] {p}: ${usd:.3f}, {per[p]:.0f}s", flush=True)
        if i == 0:
            spent_first = usd
    return {"total_seconds": round(time.perf_counter() - t_all, 1), "seconds_per_page": per}


def warm_up_ollama():
    """Load the local model once so that the timed local run does not pay for it (the models in memory stay for 30 minutes)."""
    for m in {LOCAL.emotion_model, LOCAL.translate_model}:
        try:
            llm.chat(m, "Reply with {}.", {"type": "object"}, timeout=300, max_tokens=8)
        except llm.OllamaError as e:
            print(f"  warm-up of {m} failed ({e}); the first local page will include loading it")


def import_earlier(pages):
    """Add the pages of the earlier run (first comparison, and the local run) to this run's folders so the sheets show everything.
    Returns {label: seconds} for what was imported. Nothing is run."""
    old = MANGA / "output" / "compare-openai"
    got = {}
    for src, label in [("A", "A-gpt-image-2"), ("B-gpt-6.1-sol", "B-gpt-6.1-sol"), ("C-gpt-6.1-sol", "C-gpt-6.1-sol")]:
        if all((old / src / "pages" / f"{p}.png").exists() for p in pages):
            (OUT / label / "pages").mkdir(parents=True, exist_ok=True)
            for p in pages:
                shutil.copy(old / src / "pages" / f"{p}.png", OUT / label / "pages" / f"{p}.png")
            got[label] = "earlier run"
    if all((SRC / "rendered" / f"{p}.png").exists() for p in pages):
        (OUT / "D-local" / "pages").mkdir(parents=True, exist_ok=True)
        for p in pages:
            shutil.copy(SRC / "rendered" / f"{p}.png", OUT / "D-local" / "pages" / f"{p}.png")
        got["D-local"] = "earlier local run"
    return got


def judging_sheets(pages, arms):
    """Original plus every arm, in random order with letters hiding which is which, 4 panels per row."""
    (OUT / "sheets").mkdir(exist_ok=True)
    rng = random.Random(11)
    key = {}
    for p in pages:
        orig = cv2.imread(str(OUT / "input" / f"{p}.png"))
        h, w = orig.shape[:2]
        have = [a for a in arms if (OUT / a / "pages" / f"{p}.png").exists()]
        rng.shuffle(have)
        key[p] = {chr(65 + i): a for i, a in enumerate(have)}
        panels = [orig]
        for i, a in enumerate(have):
            im = cv2.resize(cv2.imread(str(OUT / a / "pages" / f"{p}.png")), (w, h))
            cv2.putText(im, chr(65 + i), (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            panels.append(im)
        while len(panels) % 4:
            panels.append(np.full_like(orig, 255))
        rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
        cv2.imwrite(str(OUT / "sheets" / f"{p}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT / "key.json").write_text(json.dumps(key, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", nargs="+", default=PAGES)
    ap.add_argument("--text-models", default="gpt-5.4-mini", help="comma list; each runs as arm B and arm C")
    ap.add_argument("--image-model", default="gpt-image-1-mini")
    ap.add_argument("--max-usd", type=float, default=0.5, help="hard spending cap over all API arms")
    ap.add_argument("--arms", default="A,B,C", help="which arms to run (D = rerun the local pipeline, off by default)")
    ap.add_argument("--out", default="compare-openai-v2", help="folder name under manga/output/ for this run's results")
    ap.add_argument("--no-warmup", action="store_true", help="do not load the Ollama model before timing arm D (its first page then pays for loading it)")
    ap.add_argument("--no-import", action="store_true", help="do not add the earlier run's pages to the judging sheets")
    ap.add_argument("--dry-run", action="store_true", help="check the key, models and files; make no paid call and run nothing")
    a = ap.parse_args()
    global OUT
    OUT = MANGA / "output" / a.out
    models = [m for m in a.text_models.split(",") if m]
    arms = a.arms.split(",")
    ob.LEDGER.max_usd = a.max_usd
    for p in a.pages:
        if not (SRC / "pages" / f"{p}.png").exists():
            sys.exit(f"missing {SRC / 'pages' / (p + '.png')}: run the local pipeline on test-123 first")
    if not (SRC / "fonts.json").exists():
        sys.exit(f"missing {SRC / 'fonts.json'}")
    unknown = [m for m in models if m not in ob.PRICES] + ([a.image_model] if a.image_model not in ob.IMAGE_PRICES and "A" in arms else [])
    if unknown:
        sys.exit(f"no price known for: {', '.join(unknown)}. Add them to mangakaro/openai_backend.py first.")
    if set(arms) & {"A", "B", "C"}:
        check_models(models if set(arms) & {"B", "C"} else [], a.image_model if "A" in arms else None)
    print(f"ready. pages {a.pages}, text models {models}, image model {a.image_model}, cap ${a.max_usd:.2f}, arms {arms}")
    if a.dry_run:
        return
    (OUT / "input").mkdir(parents=True, exist_ok=True)
    for p in a.pages:
        shutil.copy(SRC / "pages" / f"{p}.png", OUT / "input" / f"{p}.png")
    fonts = SRC / "fonts.json"
    started = time.perf_counter()
    meta = {"pages": a.pages, "cap_usd": a.max_usd, "arms": {}}
    arm_dirs = []

    def pipeline(label, **kw):
        ob.LEDGER.arm = label
        meta["arms"][label] = run_pipeline_arm(label, Settings(**kw), a.pages, fonts)
        arm_dirs.append(label)

    try:
        if "D" in arms:
            if not a.no_warmup:
                warm_up_ollama()
            pipeline("D-local")
        for m in models:
            if "B" in arms:
                pipeline(f"B-{m}", emotion_model=f"openai:{m}", translate_model=f"openai:{m}", reader="ocr")
            if "C" in arms:
                pipeline(f"C-{m}", emotion_model=f"openai:{m}", translate_model=f"openai:{m}", reader="vlm", read_model=f"openai:{m}")
        if "A" in arms:
            meta["arms"][f"A-{a.image_model}"] = run_arm_a(a.image_model, a.pages)
            arm_dirs.append(f"A-{a.image_model}")
    except ob.BudgetExceeded as e:
        print(f"STOPPED: {e}")
        meta["stopped"] = str(e)
    if not a.no_import:
        meta["imported_from_earlier_runs"] = import_earlier(a.pages)
        arm_dirs += [x for x in meta["imported_from_earlier_runs"] if x not in arm_dirs]
    meta["cost_by_arm"] = ob.LEDGER.by_arm()
    meta["total_usd"] = round(ob.LEDGER.spent, 4)
    meta["wall_seconds_everything"] = round(time.perf_counter() - started)
    (OUT / "cost.json").write_text(json.dumps(meta, indent=1))
    judging_sheets(a.pages, arm_dirs)
    print(f"\ntotal API cost ${ob.LEDGER.spent:.3f}")
    for label, v in meta["arms"].items():
        c = meta["cost_by_arm"].get(label, {"usd": 0.0})
        print(f"  {label:28s} {v['total_seconds']:7.1f} s for {len(a.pages)} pages   ${c['usd']:.3f}")
    print(f"judging sheets: {OUT / 'sheets'} (key in {OUT / 'key.json'}, costs and times in {OUT / 'cost.json'})")


if __name__ == "__main__":
    main()
