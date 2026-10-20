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
| `llm.py` | Ollama client, setup check for installed models. A model named `openai:<id>` is routed to `openai_backend.py` instead |
| `openai_backend.py` | Optional, for the cloud comparison only: the OpenAI call with the same contract as the Ollama one, a price table, and a ledger of tokens, seconds and dollars with a hard spending cap. Needs the `compare` extra and `OPENAI_API_KEY` |
| `prompts.py` | Every prompt and JSON schema sent to the models. Edit here to change what is asked |
| `stages.py` | The read, emotion and translate stages, their retry logic, result merging, caching |
| `fonts.py` | Candidate fonts, lettering-weight measurement, choosing the primary and secondary font |
| `render.py` | Classifies each box (plain, inverse, overlay), erases, places and fits text |
| `inpaint.py` | LaMa neural inpainting on small crops, used for text over screentone and art |
| `verify.py` | After typesetting: finds Japanese still on the page and labels the cause |
| `paths.py` | Where weights and extra fonts live (`MANGAKARO_WEIGHTS` overrides the weights folder) |

`experiments/` holds the scripts used to decide the design: detector comparison, the one-shot S0 approach, review sheets, and the cloud comparison
(`openai_compare.py` runs the arms, `judge_pack.py` and `judge_pack2.py` build blind sheets for judging them). They are not part of the installed package.

## Models

| Job | Model | Runs on | License (from each model's page; check before redistributing) |
| --- | --- | --- | --- |
| Text boxes and panels | `comic-panels-text-detect`, YOLO26n-seg | CPU | Apache-2.0 |
| Speech bubbles | `manga_bubbles_detect`, YOLOv8 | CPU | MIT |
| Lettering pixel mask | `comic-text-detector` (ONNX, from manga-image-translator) | CPU | GPL-3.0 |
| Reading Japanese | `kha-white/manga-ocr-base` | Apple GPU (MPS) or CPU | Apache-2.0 |
| Erasing over art | `big-lama` (LaMa, TorchScript) | CPU | Apache-2.0 per the LaMa repository (check before redistributing) |
| Emotion and scene | `gemma4:e4b-mlx` through Ollama (default) | local | see Ollama |
| Translation | `gemma4:e4b-mlx` through Ollama (default) | local | see Ollama |

The YOLO weights are loaded with Ultralytics, which is AGPL-3.0. The PDF reader, pypdfium2, is permissively licensed.
System fonts (Comic Sans MS and others) are used from the machine and are not redistributed.

## The cloud comparison harness

`experiments/openai_compare.py` runs the same pages through several *arms* and records cost and time. The pipeline arms are this same program with the model behind some steps swapped; the image arm skips the pipeline.

| Arm | Read | Emotion and translation | Erase and typeset |
| --- | --- | --- | --- |
| D, local | manga-ocr | Gemma through Ollama | ours |
| B-`<m>` | manga-ocr | OpenAI model `<m>` | ours |
| C-`<m>` | OpenAI model `<m>` | OpenAI model `<m>` | ours |
| A | none | none | one OpenAI image-edit call returns the page |

- **From scratch.** Each pipeline arm deletes its work directory first, so detection, reading, the models and the typesetting all run and are timed. The first page of an arm pays for loading the models; the per-page time is the gap
  between page finishes, which puts nearly all the time on the first page (all model calls for every page run before the rendering), so use the arm total. The left-over-Japanese check is off in every arm.
- **Cost.** `openai_backend.py` counts tokens from the API's own usage numbers and multiplies by the prices in `PRICES` and `IMAGE_PRICES` (copied from OpenAI's page; check them). A call is refused once the cap is reached, and the image arm
  stops if the projected cost of the remaining pages would pass it.
- **Judging.** `judge_pack.py` writes, per page and per cloud arm, a sheet with the original and two candidates in random order (X and Y) and a key file, plus a table of the Japanese read and English written per box.
  `judge_pack2.py` writes half-page sheets at 1.6x for checking small lettering. The key is never shown to the judge. Results of one judging round are in [JUDGE_REVIEW.md](JUDGE_REVIEW.md) and [judge_verdicts.json](judge_verdicts.json).
- **Not blind in practice.** A judge that has read the per-line tables, or that judged an earlier pass, knows which side is local. A real second judgment needs a separate call with no such memory.

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

**Three kinds of text box.** Each box is classified from the pixels around its lettering (not counting the lettering itself):
*plain* (mostly white paper, or light and flat shaded paper near the spine): black English inside the bubble; *inverse*
(flat area darker than gray 70): the lettering is erased with a flat fill of the area's own colour and the English is white;
*overlay* (anything else, such as screentone or art): erased with LaMa, English in black with a 3 px white outline so it reads
on any background. Sound effects, as labelled by the emotion step, are left alone.

**LaMa for overlay text only.** OpenCV's inpainting blurs a region, which leaves cloudy patches over screentone and hatching.
LaMa continues the texture. It was tested against the OpenCV result on page 14 and 19 of the 20-page chapter: clearly better
over screentone and hatching, better but not perfect on hair and face (a dark smudge can remain), and worse than a flat fill on
solid black, so inverse boxes keep the flat fill. It runs on 48 px of context around each box, on the CPU (its FFT layers are not
reliable on Apple's GPU backend), at about half a second per box.

**Tall narrow text.** Vertical Japanese is a narrow column; English is not. For a tall narrow box whose English would be under 16 px,
`widen()` grows the text rectangle sideways over free space (not other text, not a panel border) until it reaches 16 px or 2.6 times the
original width. A plain box whose lettering touches drawn art (`has_art`: more than 6% of the non-lettering pixels are dark) is erased
with LaMa, and its English is kept on the white part above the art (`paper_top`) and aligned to the top.

**Triage before the models.** `ocr.triage` leaves a box untouched (and reports it as `skipped: ...`) when the OCR found letters but
no Japanese, or when the box is too big for the characters read from it (a block of prose or big title lettering, where manga-ocr
returns a short invented sentence that would otherwise replace the real text). Such boxes skip the emotion and translation calls.

**Check after typesetting.** `verify.py` runs the lettering detector at a lower threshold on the finished page (whole page plus
two overlapping zoomed halves), drops anything inside the English we placed, reads what is left with manga-ocr, and keeps what
contains Japanese. Cause: `skipped` (we left it on purpose), `erase_failed` (inside a box we translated), `not_detected`.
It reports; it does not fix. It cannot see sound effects, because both detectors find only dialogue.

**Two detectors for boxes.** `panels_text` gives the boxes and the panels. `comic_text_detector` often finds text it misses (a
thought bubble, margin text). `detect.merge_text_boxes` adds its boxes where less than 40% is already covered. The box cache
carries a version number, so cached pages are redone when the way boxes are found changes.

**A pipeline around the model, not a model that does it all.** On four test pages a one-shot image model (`gpt-image-2`) invented lines on all four and redrew the art, while the same pipeline with local Gemma kept the page faithful and landed close to the pipeline with `gpt-5.4-mini`.
The faithfulness comes from the structure: boxes found by detectors, text read by an OCR that can only output what is in the crop, translation constrained to those boxes, and erasing and typesetting done by code that never touches the art outside them.
The OCR is also the ceiling: its misreads reach every arm that uses it. See [JUDGE_REVIEW.md](JUDGE_REVIEW.md).

**Stalled model calls.** One model call can loop on garbled OCR text. Calls have a 180 s limit and a length cap; a page whose
translation fails is retried in chunks of 4 lines, then in rounds for missing lines, then one line at a time, so one bad line
cannot take a page down. A page that still fails is left unchanged and listed at the end.

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
| `DARK_LEVEL`, `DARK_FLAT` | 70, 0.5 | median gray below this and flat = the "inverse" kind |
| `OVERLAY_PAD`, `OUTLINE` | 8 px, 3 px | extra room and white outline width for text over art |
| `WIDE_TARGET`, `TALL` | 16 px, 1.8 | widen a column whose English is under this size when height is more than this times width |
| `ART_SHARE` | 0.06 | dark non-lettering share above which a plain box counts as touching art |

## Extending it

- **Different models:** `--emotion-model` and `--translate-model` take any Ollama model name. Vision is needed for emotion.
- **Different prompts:** edit `prompts.py`. Delete `cache/<page>/emotion.json` or use `--redo emotion` to apply them.
- **More fonts:** put `.ttf` or `.otf` files in `fonts/` (check each font's license). They are added to the candidates.
- **A cloud model for comparison only:** `--emotion-model openai:gpt-5.4-mini` and `--translate-model openai:gpt-5.4-mini` work because `llm.chat` routes the prefix (needs `OPENAI_API_KEY`, the `compare` extra, and a price in `openai_backend.PRICES`).
  Cost and the spending cap are only reported by `experiments/openai_compare.py`; the CLI does not print them.
- **A fixed font pair:** pass `--fonts picks.json` with `{"primary": {...}, "secondary": {...}}` (see `fonts.json` in a work directory).
