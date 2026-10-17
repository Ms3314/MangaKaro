# Architecture

How the pipeline works, what each module does, and why it is built this way.

## The pipeline

```
PDF, folder of images, or a list of image files
        │
        ▼
  pages.py / pdfio.py     render PDF pages, split two-page spreads (right page first), shrink to max 1536 px
        │
        ▼   per page ───────────────────────────────────────────────────────────────────────────────┐
  detect.py (panels_text)   find text boxes and panels; give boxes reading-order IDs (1.1, 1.2, 2.1 …)  │  Phase A
        │                   drop boxes that sit inside a larger one (ocr.drop_contained)                │  understand
        ▼                                                                                               │
  ocr.py (manga-ocr)        read the Japanese in each box crop                                          │
        ▼                                                                                               │
  stages.py: emotion        vision model sees the page + the text → speaker, emotion, intensity,        │
        │                   lettering style, one-line scene description                                 │
        ▼                                                                                               │
  stages.py: translate      text-only model translates the whole page at once, with the emotion tags    │
        ▼                                                                                               │
  results/<page>.json  ◄────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  fonts.py                  once per manga: measure its lettering weight, pick primary + secondary font   Phase B
        │
        ▼   per page
  render.py                 erase the original lettering, typeset the English inside the bubble          Phase C
        │
        ▼
  pdfio.py                  all pages (translated or untouched) into the output PDF                      Phase D
```

Every model call and every page result is cached in the work directory, so an interrupted run resumes where it stopped and
a single stage can be redone (`--redo translate`).

## Modules (`mangakaro/`)

| Module | Job |
| --- | --- |
| `cli.py`, `__main__.py` | `mangakaro translate` and `mangakaro check` |
| `pipeline.py` | Runs phases A to D, handles caching, per-page failures, setup checks, the summary |
| `pages.py` | Works out what the input is (PDF, folder, or list of image files), sorts images naturally, loads pages, splits spreads, shrinks, builds the contact sheet used by the optional vision reader |
| `pdfio.py` | PDF reading (pypdfium2) and writing (Pillow) |
| `detect.py` | The three detectors, reading order and ID assignment, the overlay drawing |
| `ocr.py` | manga-ocr reading, ellipsis cleanup, duplicate-box removal |
| `llm.py` | Ollama client, setup check for installed models |
| `prompts.py` | Every prompt and JSON schema sent to the models. Edit here to change what is asked |
| `stages.py` | The read, emotion and translate stages, their retry logic, result merging, caching |
| `fonts.py` | Candidate fonts, lettering-weight measurement, choosing the primary and secondary font |
| `render.py` | Erasing, placing and fitting text |
| `paths.py` | Where weights and extra fonts live (`MANGAKARO_WEIGHTS` overrides the weights folder) |

`experiments/` holds the scripts used to decide the design (detector comparison, the one-shot S0 approach, review sheets).
They are not part of the installed package.

## Models

| Job | Model | Runs on | License (from each model's page; check before redistributing) |
| --- | --- | --- | --- |
| Text boxes and panels | `comic-panels-text-detect`, YOLO26n-seg | CPU | Apache-2.0 |
| Speech bubbles | `manga_bubbles_detect`, YOLOv8 | CPU | MIT |
| Lettering pixel mask | `comic-text-detector` (ONNX, from manga-image-translator) | CPU | GPL-3.0 |
| Reading Japanese | `kha-white/manga-ocr-base` | Apple GPU (MPS) or CPU | Apache-2.0 |
| Emotion and scene | `gemma4:e4b-mlx` through Ollama (default) | local | see Ollama |
| Translation | `gemma4:e4b-mlx` through Ollama (default) | local | see Ollama |

The YOLO weights are loaded with Ultralytics, which is AGPL-3.0. The PDF reader, pypdfium2, is permissively licensed.
System fonts (Comic Sans MS and others) are used from the machine and are not redistributed.

## Result formats

`cache/<page>/boxes.json`: the text boxes of the page.

```json
[{"type": "text", "bbox": [x0, y0, x1, y1], "conf": 0.91, "label": "2.1", "panel": 2}]
```

`cache/<page>/read.json`, `emotion.json`, `translate.json`: one per stage, each with `model`, `seconds`, `ids_ok` (did the
model return exactly the boxes it was given), `retries`, and `regions`. `emotion.json` also has `scene`.

`results/<page>.json`: the merged page result, which is what the renderer reads.

```json
{"page": "p0002", "models": {"read": "manga-ocr", "emotion": "...", "translate": "..."},
 "seconds": {"read": 6.0, "emotion": 15.4, "translate": 4.4}, "ids_ok": true, "scene": "…",
 "boxes": {"1.1": [x0, y0, x1, y1]},
 "regions": [{"id": "1.1", "source_text": "なっ…！", "emotion": "surprised", "intensity": 0.8,
              "translation": "Wha—?!", "style": {"weight": "normal", "slant": "upright", "size": "medium", "shape": "normal"},
              "flag": null}]}
```

`report.json`: per page, one entry per text block with `status` (`rendered`, `overflow: …`, `skipped: …`), the font and size
used, the rectangle the text was fitted into, and the text.

## Design decisions and the evidence behind them

**OCR reads the text, the vision model does not.** A 4B vision model reading Japanese off a page produced invented text and
romaji in the middle of Japanese (experiment S1). manga-ocr can only output what is in the crop, and read the same pages
mostly correctly in about 1.5 s per page. It still misreads some short vertical strings (see Known limitations).

**Stages, not one call.** Reading, emotion and translation are separate calls, each with a small output schema. Staging
alone did not help the 4B model (S1 was slower than the one-shot S0 and no better at reading), but it let each stage use a
different model and be redone on its own. Translation is text-only, so it is fast (3 to 11 s per page).

**Translate the whole page at once.** The translator sees every line of the page with speaker and emotion, so voices and
pronouns stay consistent. `gemma4:e4b-mlx` gave the most natural English of the three translators compared (4B Qwen,
`qwen3:8b`, Gemma).

**Retry skipped boxes.** Models sometimes return valid JSON that silently omits boxes (Gemma returned 1 of 11 on one page,
and dropped boxes in the emotion stage on 3 of 18 pages). Both stages now ask again for just the missing boxes, up to twice.

**Erase from a pixel mask, not by darkness.** Erasing every dark pixel in the detector box also removed bubble outlines and
black backgrounds. `comic_text_detector`'s segmentation mask follows the lettering stroke by stroke, so only glyphs are
removed (inside the translated boxes, plus 3 px), then inpainted.

**Grow the text area from the middle of the bubble.** The detector's box often already touches the bubble outline, so fitting
text to it spills over. The text area starts as a small square at the centre of the text and grows while the new strip is all
paper, so it can only cover paper. If that area is too small to read (under 14 px text) the detector box is used instead,
which can reach the outline.

**White paper first.** A box is a plain bubble if at least 80% of its non-lettering, non-dark pixels are white (gray above
215). Only if that fails does a fallback accept light, flat, shaded paper (median 175 or brighter, 70% flat), which covers
bubbles next to the book's spine in photographed scans. Screentone and art are not flat and are left untouched.

**Fonts from the manga's own lettering.** The vision model's style tags were not usable (144 of 146 boxes "normal").
Instead, stroke thickness divided by glyph size is measured on the manga's lettering and on each candidate font, and the
closest weights are chosen. Only weight is matched, not letter shape.

## Thresholds worth knowing (`render.py`)

| Name | Value | Meaning |
| --- | --- | --- |
| `WHITE` | 215 | gray above this is white paper |
| `MIN_PAPER` | 0.8 | share of background pixels that must be white |
| `MIN_PAPER_LEVEL`, `MIN_FLAT` | 175, 0.7 | shaded-paper fallback |
| `ERASE_PAD` | 3 px | how far outside a detector box the lettering mask may be erased |
| `TEXT_PAD` | 6 px | margin between the English and the edge of its area |
| `MIN_READABLE` | 14 px | smaller than this inside the bubble and the detector box is tried instead |
| `MAX_SIZE`, `MIN_SIZE` | 40, 10 px | font size range |
| `LOUD` | 0.75 | emotion intensity at or above this uses the secondary font |

## Extending it

- **Different models:** `--emotion-model` and `--translate-model` take any Ollama model name. Vision is needed for emotion.
- **Different prompts:** edit `prompts.py`. Delete `cache/<page>/emotion.json` or use `--redo emotion` to apply them.
- **More fonts:** put `.ttf` or `.otf` files in `fonts/` (check each font's license). They are added to the candidates.
- **A fixed font pair:** pass `--fonts picks.json` with `{"primary": {...}, "secondary": {...}}` (see `fonts.json` in a work directory).
