# Progress log

What was built, in what order, what each experiment showed, and what went wrong on the way. Last updated 2026-10-17.

None of the quality results below are scored against a reference translation. Where this log says "better", it means a
person (or I) looked at the output. The evaluation phase of the plan exists to replace that.

## Status against the plan

The original plan is in `Manga Translation Pipeline — Phase-wise MVP Plan.md`.

| Phase | Status |
| --- | --- |
| 0 Setup and test data | Mostly done. Test data: 18 Jujutsu Kaisen pages, 99 Daiboken vol. 2 scans, 5 digital pages (`manga-demo`). Still missing: a manhwa page, a sound-effect-heavy page, official English versions for scoring |
| 1 Detection | Done. Three detectors compared, `panels_text` chosen for text boxes and panels |
| 2 Read, emotion, translate | Working. OCR reads, a vision model tags emotion, a text model translates. Not scored |
| 3 Erase and render | Working for plain white bubbles. Sound effects and text over art are left untouched |
| 4 Automation | Done in basic form: `mangakaro translate book.pdf`, resumable, per-page logging and caching |
| 5 Evaluation | Not started. Needs reference translations |
| 6 Improvements | Not started |

## Phase 1: detector comparison

Three pretrained detectors on the 18 Jujutsu Kaisen pages, on CPU.

| Detector | Found | Seconds per page | Peak memory (cumulative) |
| --- | --- | --- | --- |
| `panels_text` | 131 text, 55 panels | 0.11 | 1,006 MB |
| `bubbles_yolov8` | 121 bubbles | 0.15 | 1,006 MB |
| `comic_text_detector` | 143 Japanese text, 4 English | 0.31 | 1,457 MB |

`panels_text` is the primary detector (fastest, gives text and panels). Later all three were used for different jobs:
`panels_text` for boxes and order, `comic_text_detector` for the lettering pixel mask, `bubbles_yolov8` for bubble boundaries.
Misses and false boxes per detector were not recorded. The choice rests on speed, coverage and looking at overlays.

## Phase 2: experiments S0 to S7

First four Daiboken half-pages (33 boxes), then 18 new half-pages for S7. Models run locally through Ollama.

| Variant | Setup | Result |
| --- | --- | --- |
| S0 | One call: `qwen3.5:4b` reads, tags and translates | 30 to 73 s per page. Garbled characters and mistranslations even where the reading was right |
| S1 | Three stages, `qwen3.5:4b` in all | 57 to 92 s per page, slower than S0. Reading did not improve. New errors: romaji inside Japanese, an invented word, a scene note naming the wrong series |
| S4 | manga-ocr reads, `qwen3.5:4b` tags emotion and translates | Reading takes about 1.5 s per page and is mostly correct. Translation errors remain |
| S5 | S4 with `qwen3:8b` translating | 10 to 19 s per page. Left シロート untranslated, returned an empty line for a "..." box |
| S6 | S4 with `gemma4:e4b-mlx` translating | 3 to 11 s per page. Most natural English. Returned 1 of 11 boxes on one page until a retry was added |
| S7 | Whole pipeline on Gemma plus OCR, 18 unseen half-pages | 146 boxes, 867 s of model time (median per page: read 1.5 s, emotion 31 s, translate 9.5 s). Emotion stage dropped boxes on 3 of 18 pages until a retry was added there too |

The same lines through each variant:

| Japanese | S0 | S1 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- |
| ママ、うんま〜 | Mom, I'm good. | Mom, I'm hungry~. | Mom, I'm doing great...! | Mom, this is delicious... | Mama, yummm... |
| シロート？！ | Rube?!? | Amateur?! | A fool?! | Shirouto!? | A rookie!? |
| 人があいさつしてんのに〜 | People are greeting each other. | They're even greeting us~. | Someone's greeting me...!!! | You're greeting someone? ~~~~~ | Someone's greeting us, but... |
| つかれまちた | I got caught. | We're exhausted. | I'm getting tired. | I'm exhausted. | I'm tired. |

What this showed: staging with the same small model did not help; the reading was the bottleneck, and OCR fixed most of it.
Gemma was the best of the three translators tried. A bigger Gemma (12B, 26B) and Llama models were not tried.
Planned but not run: S2 (stronger translator on S1's text) and S3 (a bigger vision model for reading).

## Phase 3: rendering

Iterations, each driven by looking at rendered pages:

1. Erase every dark pixel in the box and fit text to the box. Failed: erased bubble outlines and black backgrounds, text spilled out.
2. Erase from `comic_text_detector`'s pixel mask only, inside the translated boxes, then inpaint. Outlines survive.
3. Place text with `bubbles_yolov8`'s bubble boxes and grow the text area from the middle of the text.
4. Pick fonts per manga from measured lettering weight (140 Daiboken boxes: median 0.144, 90th percentile 0.227 →
   Comic Sans MS Regular and Bold; the 17 demo boxes → Comic Sans MS Regular and Avenir Next Condensed Demi Bold).
5. Review round on 24 numbered cards (20 random rendered, 4 random skipped). Feedback: four translated bubbles had been
   skipped, four had English running out of the bubble. After the fixes all eight are translated and inside their bubbles.

Result on the 18 S7 half-pages (146 boxes):

| | First version | After fixes |
| --- | --- | --- |
| Typeset | 121 | 143 |
| Overflowing | 2 | 2 |
| Skipped as not plain bubble | 23 | 1 |
| Blocks at 12 px or smaller | 7 of 123 | 10 of 145 |

The 23 skipped boxes were bubbles near the book's spine, where the scan is shaded. White paper stays the first test; a
light-and-flat fallback accepts shaded paper.

## Demo pages (digital scans, not photographs)

Five pages, 19 text boxes, run end to end. All 19 boxes were typeset. Three pages came out clean. Two boxes were misread by
the OCR: a bubble that says 約束する ("I promise") was read with extra text and became "And I promise that,", and a bubble that
says ——よし was read as しかし、 and became "But,", with the leftover dash still visible in the bubble.

## Mistakes along the way

| What went wrong | What it taught |
| --- | --- |
| Called two boxes "overlapping duplicates" without measuring. They were adjacent columns of one bubble (overlap 0.09); the only real duplicate sat inside another box | Measure before describing |
| Expected staging the same 4B model to help; S1 was slower and no better | The bottleneck was reading, not the number of jobs per call |
| Let the vision model read Japanese: invented text, romaji, a wrong series name | OCR cannot output text that is not in the crop |
| `.venv/bin/pip` does not exist in a uv environment | Use `uv pip` |
| A script imported a helper that did not exist | Search before the first run |
| Gemma returned valid JSON with 10 of 11 boxes missing, no error | Check the returned IDs against the requested ones and retry |
| The same failure existed in the emotion stage and was found only on S7 | Apply a fix to every stage with the same risk |
| Judged white bubbles by a ring around the box: tight boxes sit next to the outline, so every box failed | Test the inside of the box |
| Filled erased text with pure white; scanned paper is about 232, so patches showed | Use the local paper colour, then inpainting |
| Grouped boxes by connected white areas; panel gutters joined unrelated bubbles into one block | Use the bubble detector's boxes |
| Erased by darkness; deleted outlines and black backgrounds inside boxes | Erase only the detector's lettering mask |
| Tested "is it white?" with a fixed brightness; bubbles by the spine are shaded and were all skipped | Keep white as the first test and add a light-and-flat fallback |
| Started the text area from the detector box, which already touches the outline | Grow from the middle; the box is only a fallback |
| First attempt to relax the fallback rule fell back to the old box 114 times out of 145 | Check how often a fallback fires |
| The vision model's lettering-style tags (144 of 146 "normal") were useless for choosing fonts | Measure from pixels |
| Tried to send images to the user as files; the session could not deliver them | Publish them where the user can open them |
| One edit to the shared review document moved a comment onto different text | Re-read what a comment is attached to before rewriting it |

## Known limitations

- Japanese to English only. The OCR is Japanese-specific.
- Only plain bubbles are typeset. Sound effects, text over art and captions without a clean background stay in Japanese.
- OCR can misread or add text on very short vertical strings (two boxes in 19 on the demo pages).
- Very narrow bubbles get small English text (10 of 145 blocks at 12 px or smaller on the S7 pages).
- The English can still touch an outline when the bubble is too small for readable text and the detector box is used.
- Output pages are raster images, at most 1,536 px on the longest side, and two-page spreads become two pages.
- Translation quality has not been scored. There are no reference translations yet.

## Next

1. Get official English pages for some of the test pages and score the variants instead of judging by eye.
2. Fix short-vertical-text misreads (a vision-model check on OCR output, or a second OCR pass).
3. Handle narrow bubbles: fit text to the bubble's real shape, not a rectangle.
4. Extend rendering to sound effects and text over art.
5. Try a bigger Gemma and Llama models as translators.
