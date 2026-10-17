"""The whole job: a PDF (or a folder of images) in, a translated PDF out.

Phases, each resumable because every result is cached in the work directory:
  A understand  per page: find text boxes -> read (OCR) -> emotion/scene -> translate        (cache/<page>/*.json, results/<page>.json)
  B fonts       once per manga: measure its lettering weight and pick a primary and a secondary English font (fonts.json)
  C render      per page: erase the original lettering and typeset the English                (rendered/<page>.png, report.json)
  D pdf         assemble all pages, translated or not, into the output PDF

Work directory layout:
  pages/<page>.png      the page images exactly as the models saw them
  cache/<page>/         boxes.json, read.json, emotion.json, translate.json (delete one to redo it and what follows)
  results/<page>.json   final per-page result: boxes + source text + emotion + translation
  fonts.json, fonts.png font picks and the specimen sheet
  rendered/<page>.png   translated pages        report.json  what happened to every text block
  summary.json          timings, counts, failures
"""
import importlib.util
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import llm
from .detect import BubblesYolo, ComicTextDetector, PanelsText, assign_ids, draw
from .fonts import DEFAULT_FONT, candidate_fonts, measure_manga, pick_fonts, write_specimen
from .ocr import drop_contained, read_boxes
from .pages import MAX_SIDE, contact_sheet, load_pages
from .paths import WEIGHT_FILES, WEIGHTS
from .pdfio import images_to_pdf
from .render import Renderer
from .stages import merge, run_stage, stage_emotion, stage_read, stage_translate

STAGES = ["read", "emotion", "translate"]


class SetupError(RuntimeError):
    pass


@dataclass
class Settings:
    emotion_model: str = "gemma4:e4b-mlx"
    translate_model: str = "gemma4:e4b-mlx"
    reader: str = "ocr"                 # "ocr" (manga-ocr) or "vlm" (the vision model reads a contact sheet)
    read_model: str = "qwen3.5:4b"      # only used when reader == "vlm"
    max_side: int = MAX_SIDE
    split_spreads: bool = True
    pages: str | None = None            # e.g. "1-3,7"
    fonts_file: Path | None = None
    redo: list = field(default_factory=list)   # stages whose cache to ignore, and every stage after them
    ollama_url: str = llm.DEFAULT_URL

    def models(self):
        return [self.emotion_model, self.translate_model] + ([self.read_model] if self.reader == "vlm" else [])


def check_setup(s):
    """Everything that must be in place before a run. Returns a list of problems; empty means ready."""
    problems = []
    for name, source in WEIGHT_FILES.items():
        if not (WEIGHTS / name).exists():
            problems.append(f"missing weights file {WEIGHTS / name}\n  get it from {source}")
    for module, package in [("manga_ocr", "manga-ocr"), ("pypdfium2", "pypdfium2"), ("ultralytics", "ultralytics"),
                            ("onnxruntime", "onnxruntime"), ("cv2", "opencv-python"), ("PIL", "pillow")]:
        if importlib.util.find_spec(module) is None:
            problems.append(f"python package not installed: {package}  (uv pip install {package})")
    llm.configure(s.ollama_url)
    try:
        llm.require_models(s.models())
    except llm.OllamaError as e:
        problems.append(str(e))
    if not Path(DEFAULT_FONT["path"]).exists() and not candidate_fonts():
        problems.append("no usable English font found: put a .ttf or .otf file in the project's fonts/ folder")
    return problems


class Models:
    """Loads each detector the first time it is used, so a run that only needs some of them stays light."""
    def __init__(self):
        self._m = {}

    def get(self, cls):
        if cls not in self._m:
            self._m[cls] = cls()
        return self._m[cls]

    panels = property(lambda self: self.get(PanelsText))
    text = property(lambda self: self.get(ComicTextDetector))
    bubbles = property(lambda self: self.get(BubblesYolo))


def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def understand_page(name, page, s, models, work):
    """Phase A for one page. Returns the page's result dict, or None if it has no text."""
    d = work / "cache" / name
    d.mkdir(parents=True, exist_ok=True)
    redo = set(STAGES[min(STAGES.index(x) for x in s.redo):]) if s.redo else set()
    for stage in redo:
        (d / f"{stage}.json").unlink(missing_ok=True)
    boxes_file = d / "boxes.json"
    if boxes_file.exists():
        texts = json.loads(boxes_file.read_text())
    else:
        regions = assign_ids(drop_contained(models.panels(page)))
        texts = [g for g in regions if g["type"] == "text"]
        _write(boxes_file, texts)
    if not texts:
        return None
    annotated = draw(page, texts)
    if s.reader == "vlm":
        sheet = contact_sheet(page, texts)
        read = run_stage(d / "read.json", lambda: stage_read(s.read_model, annotated, sheet, texts))
    else:
        read = run_stage(d / "read.json", lambda: read_boxes(page, texts))
    emo = run_stage(d / "emotion.json", lambda: stage_emotion(s.emotion_model, annotated, texts, read))
    tr = run_stage(d / "translate.json", lambda: stage_translate(s.translate_model, texts, read, emo))
    final = merge(name, texts, read, emo, tr)
    _write(work / "results" / f"{name}.json", final)
    return final


def choose_fonts(s, work, models, results, log):
    """Phase B. Returns {primary, secondary} font specs."""
    if s.fonts_file:
        log(f"fonts: using {s.fonts_file}")
        return json.loads(Path(s.fonts_file).read_text())
    cached = work / "fonts.json"
    if cached.exists():
        picks = json.loads(cached.read_text())
        log(f"fonts: reusing {cached.name}: primary {picks['primary']['name']}, secondary {picks['secondary']['name']}")
        return picks
    items = [(cv2.imread(str(work / "pages" / f"{name}.png")), list(r["boxes"].values())) for name, r in results.items()]
    picks = pick_fonts(measure_manga(items, models.text))
    if picks is None:
        log(f"fonts: too little lettering to measure, using {DEFAULT_FONT['name']} for everything")
        return {"primary": DEFAULT_FONT, "secondary": DEFAULT_FONT}
    write_specimen(picks, work / "fonts.png")
    saved = {k: v for k, v in picks.items() if k != "_cands"}
    _write(cached, saved)
    log(f"fonts: primary {picks['primary']['name']}, secondary {picks['secondary']['name']} "
        f"(lettering weight median {picks['measured']['median']:.3f}, {picks['measured']['boxes']} boxes; see fonts.png)")
    return saved


def run(source, output, work, s, log=print):
    """Translate `source` (PDF or folder of images) into the PDF `output`. Returns a summary dict (also saved as summary.json)."""
    problems = check_setup(s)
    if problems:
        raise SetupError("\n\n".join(problems))
    started = time.perf_counter()
    for sub in ("pages", "cache", "results", "rendered"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    models = Models()

    # Phase A: understand every page
    names, results, failures = [], {}, {}
    for page in load_pages(source, s.pages, s.split_spreads, s.max_side):
        names.append(page.name)
        cv2.imwrite(str(work / "pages" / f"{page.name}.png"), page.image)
        t = time.perf_counter()
        try:
            final = understand_page(page.name, page.image, s, models, work)
        except Exception as e:  # one bad page must not stop a long run: record it, keep the page unchanged, go on
            failures[page.name] = f"{type(e).__name__}: {e}"
            with open(work / "errors.log", "a") as f:
                f.write(f"--- {page.name}\n{traceback.format_exc()}\n")
            log(f"[{len(names)}] {page.name}: FAILED ({type(e).__name__}: {e}); page left unchanged, details in errors.log")
            continue
        if final is None:
            log(f"[{len(names)}] {page.name}: no text found")
            continue
        results[page.name] = final
        sec = final["seconds"]
        log(f"[{len(names)}] {page.name}: {len(final['regions'])} text boxes "
            f"(read {sec['read']}s, emotion {sec['emotion']}s, translate {sec['translate']}s)"
            + ("" if final["ids_ok"] else "  WARNING: some boxes were not returned by the model") + f"  [{time.perf_counter() - t:.0f}s]")
    if not names:
        raise SetupError(f"no pages found in {source}")

    # Phase B: fonts
    fonts = choose_fonts(s, work, models, results, log) if results else {}

    # Phase C + D: render, then assemble
    renderer = Renderer(fonts, models.text, models.bubbles)
    report, out_images = {}, []
    for i, name in enumerate(names, 1):
        image = cv2.imread(str(work / "pages" / f"{name}.png"))
        if name in results:
            image, rep = renderer.page(image, results[name])
            report[name] = rep
            cv2.imwrite(str(work / "rendered" / f"{name}.png"), image)
            log(f"render {i}/{len(names)} {name}: {sum(b['status'].startswith('rendered') for b in rep)} blocks typeset, "
                f"{sum(b['status'].startswith('skipped') for b in rep)} skipped, {sum(b['status'].startswith('overflow') for b in rep)} overflow")
        out_images.append(image)
    _write(work / "report.json", report)
    images_to_pdf(out_images, output)

    blocks = [b for rep in report.values() for b in rep]
    summary = {"input": str(source), "output": str(output), "pages": len(names), "pages_with_text": len(results),
               "pages_failed": failures, "text_blocks": len(blocks),
               "typeset": sum(b["status"].startswith("rendered") for b in blocks),
               "overflow": sum(b["status"].startswith("overflow") for b in blocks),
               "skipped": sum(b["status"].startswith("skipped") for b in blocks),
               "models": {"emotion": s.emotion_model, "translate": s.translate_model,
                          "read": "manga-ocr" if s.reader == "ocr" else s.read_model},
               "fonts": {k: v["name"] for k, v in fonts.items() if k in ("primary", "secondary")},
               "seconds": round(time.perf_counter() - started)}
    _write(work / "summary.json", summary)
    return summary
