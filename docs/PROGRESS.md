# Progress log

What was built, in what order, what each experiment showed, and what went wrong on the way. Last updated 2026-10-18.

None of the quality results below are scored against a reference translation. Where this log says "better", it means a
person (or I) looked at the output. The evaluation phase of the plan exists to replace that.

## Summary for a technical write-up

This section collects what a paper would need in one place. The rest of the log is the evidence behind it.

**Problem.** Translate Japanese manga pages (images or PDF) into English with no cloud services and no manual steps: find the text,
read it, translate it with context, remove the Japanese from the page and typeset the English so the page still looks like a
manga page. All models run locally.

**System.** A staged pipeline, each stage cached so runs resume and single stages can be redone:

| Stage | Method | Model |
| --- | --- | --- |
| Text boxes and panels | detector, merged with a second detector's boxes that the first missed | `comic-panels-text-detect` (YOLO26n-seg); `comic-text-detector` |
| Reading | OCR on each box crop, ellipsis clean-up | `manga-ocr` |
| Emotion, speaker, scene | vision-language model sees the page and the OCR text | `gemma4:e4b-mlx` via Ollama |
| Translation | text-only model translates the whole page in one request, using speaker and emotion tags; retries for skipped lines | `gemma4:e4b-mlx` via Ollama |
| Font choice | stroke-weight measure of the manga's lettering matched to candidate fonts | none (image statistics) |
| Erasing | pixel mask of the lettering; flat fill on black, LaMa where the lettering touches screentone or art, OpenCV inpainting elsewhere | `comic-text-detector` mask; `big-lama` |
| Placement | bubble boxes plus region growing from the text centre; column widening for vertical text; outline text over art | `manga_bubbles_detect` (YOLOv8) |
| Check | detector at a lower threshold on the finished page, OCR on what is left, Japanese kept as a leftover with a cause | same detector and OCR |

**Findings so far, each backed by an experiment in this log.**

1. Reading is the bottleneck, not translation: a 4B vision model reading Japanese off the page invented text and mixed in romaji;
   a dedicated manga OCR read the same pages mostly correctly in about 1.5 s per page (S1 vs S4).
2. Splitting one overloaded call into stages did not help a small model by itself (S1 was slower than S0), but it made each stage
   replaceable and cacheable.
3. Of three translators compared by reading the output, `gemma4:e4b` gave the most natural English (S4 to S6). Not scored.
4. Models skip items silently: valid JSON with 10 of 11 boxes missing, or mangled IDs. Checking returned IDs against requested
   IDs and re-asking for the missing ones fixed it in both stages that needed it.
5. A single stalled or looping generation can cost a page: one garbled OCR line made a request run for the full time limit on two
   of 20 pages. Time limits, length caps and a chunked fallback made the 20-page run finish without failures.
6. Erasing is where quality is lost: erasing by darkness removes outlines; erasing from a lettering mask keeps them; basic inpainting
   smears over screentone and hatching; LaMa continues the texture but is worse than a flat fill on solid black.
7. Where the English goes matters as much as erasing: vertical Japanese is a narrow column and English is not, so fitting English into
   the column's rectangle gives tiny text (10 px). Letting it widen sideways into free space gave 17 px on the page where it was tried.

**Main numbers.**

| Measurement | Value | Source |
| --- | --- | --- |
| Detector speed on CPU (text boxes / bubbles / text mask) | 0.11 / 0.15 / 0.31 s per page | Phase 1 |
| OCR time | about 1.5 s per page | S4 |
| Median model time per page, emotion / translate | 31 s / 9.5 s | S7 |
| 20-page chapter, first full run | 280 s, no page failed | mangahaha run |
| Text blocks typeset, first version / with over-art handling | 152 of 187 / 176 of 187 | mangahaha run |
| Pages with Japanese left after the check | 15 then 8 of 20 | check results |

**Test data.** 18 Jujutsu Kaisen pages (English edition, so used only for detector tests), 99 Daiboken vol. 2 scans (photographed,
spreads), 5 digital pages (`manga-demo`), and a 20-page digital chapter (`mangahaha`) with screentone. All are copyrighted and none
is in the repository.

**Environment.** Apple-silicon Mac (the original plan names an M4 Air). Python 3.13.5, torch 2.14.1, ultralytics 8.4.177,
manga-ocr 0.1.16, onnxruntime 1.31.0, opencv 5.0.0.93, Pillow 12.3.0, numpy 2.5.4, pypdfium2 5.14.0, transformers 5.20.1, Ollama
with `gemma4:e4b-mlx`, `qwen3.5:4b` and `qwen3:8b`. Temperature 0 for all model calls. Commands: see the README. Weights and
their sources: README.

**What is not measured yet, and a paper would need.**

| Missing | Why it matters | What it takes |
| --- | --- | --- |
| Reference translations and a translation score | Every "better" in this log is a person looking at output | Official English pages for some of the test pages, a human or judge-model rating, reported per variant |
| OCR character error rate | The reading errors flow into the translation | Hand-typed transcriptions of a sample of boxes |
| Detector precision and recall per detector | Only counts were recorded in Phase 1 | Hand-labelled boxes on a sample of pages |
| Human rating of rendering quality | The 24-card review was one person, not a rating scale | A fixed rubric and several raters on a random sample of blocks |
| Ablations on the rendering steps | The claims about LaMa, widening and the two-detector merge are from a few pages | Run each change on and off over the whole chapter and report the counts |
| Confidence intervals and repeats | All runs are single, at temperature 0 | Several chapters, report spread across them |
| Sound effects | Not translated and not detected | A detector or classifier for them, and a decision on what to output |

**Threats to validity.** One reader judged all output. Test pages come from a handful of series and mostly one visual style
(digital, black-and-white, screentone). Thresholds (paper level 215, flat gradient 25, dark level 70, art share 6%) were tuned on
the same pages they are reported on. The model that translates also writes the emotion tags it then uses. Page images are
re-encoded when they pass through a PDF, which changes some OCR readings.

## Status against the plan

The original plan is in `Manga Translation Pipeline — Phase-wise MVP Plan.md`.

| Phase | Status |
| --- | --- |
| 0 Setup and test data | Mostly done. Test data: 18 Jujutsu Kaisen pages, 99 Daiboken vol. 2 scans, 5 digital pages (`manga-demo`). Still missing: a manhwa page, a sound-effect-heavy page, official English versions for scoring |
| 1 Detection | Done. Three detectors compared, `panels_text` chosen for text boxes and panels |
| 2 Read, emotion, translate | Working. OCR reads, a vision model tags emotion, a text model translates. Not scored |
| 3 Erase and render | Working for plain bubbles, white-on-black panels and text over art (LaMa). Sound effects and background signs are left alone |
| 4 Automation | Done in basic form: `mangakaro translate` takes a PDF, a folder of images or a list of image files, and writes a PDF or a folder of images; resumable, per-page logging and caching |
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

## Part 1 to 3a: finding more text, checking pages, text over art (2026-10-18)

Run on a 20-page chapter of images (`mangahaha`), all Japanese, digital scans with screentone.

- **Missed text (Part 1).** The main detector missed a thought bubble and margin text on one page that the second detector had found.
  Boxes from `comic_text_detector` that the first detector missed are now merged in.
- **Stalls.** Two pages stalled for the full 15-minute wait. The cause was OCR garbage in stylised titles that probably made the
  model loop. Model calls now have a 3-minute limit and a cap on length, and a failed page translation is retried in chunks and
  then one line at a time.
- **Check after typesetting (Part 2a).** The detector runs on the finished page, the English we placed is ignored, and the rest
  is read with OCR; Japanese is kept as a leftover with a cause (skipped, erase_failed, not_detected). It found real leftovers
  (a tiny note, a bubble over screentone, a background sign, a sound effect) but it cannot see sound effects, because both
  detectors only find dialogue. Both detectors miss them on page 6 (チュン, ガラ, ぐっ).
- **Text over art and on dark panels (Part 3a).** 26 of 187 text blocks had been left in Japanese. Now each box is plain,
  inverse (white lettering on flat black: erased with a flat fill, English in white) or overlay (lettering over screentone or art:
  English in black with a white outline). Sound effects, as labelled by the emotion step, are left alone.

| | Before 3a | After 3a |
| --- | --- | --- |
| Typeset | 152 | 175 |
| Left in Japanese on purpose | 26 | 1 |
| English too long for its space | 9 | 11 |
| Pages with Japanese still on them | 15 | 8 |

What 3a showed: the black panels (page 15) work well. Lettering over screentone and drawn hair is the weak case: basic inpainting
smears it into cloudy streaks (page 14, left middle panel), which needs a neural inpainting model. One English line (page 19)
is a wrong translation of a misread line, which a check on translation quality would catch.

## Part 3b: LaMa for erasing over art (2026-10-18)

Question: does a neural inpainting model fix the cloudy smears OpenCV leaves where lettering sat on screentone and drawn hair?
Test: three hard crops from the 20-page chapter, original / OpenCV / LaMa side by side, with the same mask.

| Crop | OpenCV (before) | LaMa |
| --- | --- | --- |
| Hatching, page 14 top left | grey diamond-shaped blotches | texture continues, only a faint patch |
| Screentone, page 19 | blurred grey patches | dot pattern continues, a faint patch |
| Hair and face on dark, page 14 | white and grey cloud | grey cloud gone; a dark smudge and a few marks remain |
| Flat black, page 19 | clean (flat fill) | faint marks left, so not used here |

Result: used for overlay boxes only; inverse boxes keep the flat fill. About 0.5 s per box on the CPU. On the full chapter nothing
else changed (still 175 typeset, 1 skipped) and page 14's left panel lost its cloudy white streaks.

Mistakes and findings from this step:

| What | Lesson |
| --- | --- |
| `pip install simple-lama-inpainting` would have downgraded numpy (2.5 to 1.26), opencv (5 to 4.11) and pillow (12 to 9.5) | Dry-run an install first. The package only wraps a TorchScript file, so the weights are loaded with torch directly |
| LaMa on the Apple GPU backend was not used | Its FFT layers are unreliable there; the CPU is fast enough |
| A first crop for the page 19 test showed the wrong region (no over-art text in it) | Take the box list from the render report instead of guessing coordinates |
| One test crop command failed silently because of an output filter | Check that the file was written before reading it |

## Part 3c: uneven text sizes and smudged hair on one page (2026-10-18)

A review of page 14 of the 20-page chapter found two problems, both visible only side by side with the original.

1. **Tiny English beside the original's large lettering.** The two blocks in the middle-left panel are vertical Japanese columns 33 and
   69 px wide and 260 px tall. English fitted into the same rectangle needed 10 px text.
2. **Smudged hair.** A line of free-standing vertical text sat partly on a character's hair. The box was classified as a plain
   bubble (mostly white around the lettering), so it was erased with OpenCV inpainting, which blurred the hair. The English was set
   in the same narrow column and ran down over the hair.

Changes (`render.py`):

- `widen()`: for a tall narrow box (height more than 1.8 times its width) whose English would be under 16 px, the rectangle grows
  sideways, 6 px at a time over its full height, while the strip is free: not another text box, not already used, and (over art) with
  no long dark line such as a panel border. It stops at 16 px or at 2.6 times the original width.
- `has_art()`: a plain box whose non-lettering pixels are more than 6% dark has drawn art in it, so it is erased with LaMa.
- `paper_top()`: for such a box the rectangle is cut to its top run of rows that are mostly free paper, widened sideways, and the
  English is aligned to the top, so it stays on the white part.
- A first version applied the art rule only to free-standing boxes. This box had been assigned to a detected bubble, so the rule never ran. It
  now applies to both.

Result on page 14:

| Block | Before | After |
| --- | --- | --- |
| Middle left, two-line comment | 10 px in a 21 px wide column | 17 px, three lines in free space beside the column |
| Bottom left, free-standing caption | 24 px, over the hair, hair smudged | 16 px, on the white above the hair, hair rebuilt by LaMa |
| Middle panel, second column | 10 px | 10 px (no free room found; cause not investigated) |

Remaining on this page: one small light mark on the hair and a pale patch near the ear, a thin panel line through a few letters, and
the third row above. On the full chapter 176 of 187 blocks are typeset, 1 skipped and 10 overflow. These were judged by eye on page 14
only; the other pages were not re-inspected, and no before/after count over the whole chapter was taken.

| What went wrong | Lesson |
| --- | --- |
| The first crop for the hair showed the wrong part of the page | Take coordinates from the recorded boxes |
| The "widen" step only ran when the text was small, so 24 px text over the hair was untouched | The trigger must match the problem: here it was position, not size |
| The new "keep off the art" rule did not run because of an earlier condition on free-standing text | Print which branch ran before concluding the logic is wrong |

## Known limitations

- Japanese to English only. The OCR is Japanese-specific.
- Text over art is a first version: LaMa leaves small marks on hair and faces; narrow vertical columns are widened only where free space exists (one 10 px column remains on page 14). Sound effects and background signs are not translated.
- OCR can misread or add text on very short vertical strings (two boxes in 19 on the demo pages).
- Very narrow bubbles get small English text (10 of 145 blocks at 12 px or smaller on the S7 pages).
- The English can still touch an outline when the bubble is too small for readable text and the detector box is used.
- Output pages are raster images, at most 1,536 px on the longest side, and two-page spreads become two pages.
- Translation quality has not been scored. There are no reference translations yet.

## Next

1. Get official English pages for some of the test pages and score the variants instead of judging by eye (the biggest gap for a paper).
2. Hand-label a sample for OCR error rate and detector precision and recall.
3. Fix short-vertical-text misreads (a vision-model check on OCR output, or a second OCR pass) and wrong translations of misread lines.
4. Handle narrow bubbles: fit text to the bubble's real shape, not a rectangle.
5. Sound effects: detect them first, then decide whether to translate, leave or label them.
6. Try a bigger Gemma and Llama models as translators.
7. Rendering ablations over the whole chapter (LaMa, widening, two-detector merge on and off).
