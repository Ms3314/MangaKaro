# Manga Translation Pipeline: Phase-wise MVP Plan

Goal: pass in a manga/manhwa chapter and get back translated pages automatically, using only open-source models, with no manual steps. MVP = multimodal-only reading (no OCR). OCR comes back later as a judge-triggered fallback.

Purpose: a learning and portfolio project. The distinctive part is reading manga with a local vision-language model and measuring whether emotion context improves the translation (Phase 5), so scope stays at the MVP and the OCR fallback and fine-tuning are future work.

## Phase 0: Setup and Test Data

- Collect a small test set: 10 to 20 pages across a few styles (manga with bubbles, manhwa with free-floating text, one page with heavy sound effects); use Manga109 pages for the manga side, since the dataset is free for academic use
- Where possible, pair each page with an existing official English version for evaluation later
- Dev environment on the Mac M4 Air; heavier model runs on the friend's 8GB GPU machine
- Decide the folder structure: input chapter, per-page intermediate JSON, output pages

**Done when:** test pages are in a folder and the project runs a hello-world on both machines.

## Phase 1: Detection Layer (Pretrained Detectors + OpenCV)

- Detect panels, speech bubbles, and free-floating text regions on each page
- Draw boxes around each region and assign every region an ID
- Save per-page output as JSON: region ID, type (bubble / free text / SFX), bounding box
- For vertical-scroll manhwa, slice the long strip into page-sized chunks first
- If plain OpenCV misses free-floating text, train or fine-tune a small detector for it

**Model shortlist.** Run these side by side on the Phase 0 test pages before training anything:

- [comic-panels-text-detect](https://huggingface.co/Oliverdsfdsf/comic-panels-text-detect): YOLO26n-seg with Panel and Text classes, built for comics, manga and manhwa, 1280px input; 12.4 MB, Apache-2.0
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector): text boxes, text lines and segmentation masks, so its masks can also feed the erase step in Phase 3; weights come from a manga-image-translator release (size not checked), GPL-3.0
- [manga\_bubbles\_detect](https://huggingface.co/PSImera/manga_bubbles_detect): YOLOv8 bubble-only detector, self-reported validation mAP50 of about 0.977; 22.6 MB, MIT
- Panels only, if needed: [inkwell-panel-models](https://huggingface.co/cedarrapidsboy/inkwell-panel-models) (12.3 MB ONNX, AGPL-3.0 per its model card) and [DeepPanel](https://huggingface.co/codeShare/comic-panel-extract) (MIT, size not checked)
- Dropped: Manga-Bubble-YOLO, because its Hugging Face page could not be opened

Fine-tune on [Manga109](http://www.manga109.org/en/) plus a few hand-labeled manhwa pages only if free-floating text is being missed. Check each model's license before reuse: one public bubble detector is academic-use only, and Ultralytics-based weights may carry AGPL.

**Detector comparison (first hands-on step).** Run the three shortlisted detectors on the same 5 to 10 test pages and pick the primary one from the results.

1. Download the weights: the safetensors file from comic-panels-text-detect, `bubbles_detect.pt` from manga\_bubbles\_detect, and the comic-text-detector weights from the manga-image-translator release it links to
2. Write one script that runs all three on every page and saves overlay images with the detected boxes
3. Record per model: panels found, bubbles found, free-floating text found, misses and false boxes, seconds per page, peak memory
4. Pick the primary detector; fine-tune only if free-floating text is still missed on manhwa pages

All three are small, so they should run on CPU or on the M4 Air. The vision-language model, not the detectors, will be the main VRAM cost, so run the comparison first and free the detectors before loading it. Memory use is not yet measured. The `.pt` file is a pickle, so load it only from the repo named above.

**Training your own (stretch goal).** Fine-tune a pretrained YOLO nano model instead of training from scratch, since it needs far less data and time. Data: Manga109 frame and text annotations (free for academic use) plus a few hand-labeled manhwa pages. Hardware: a free Colab or Kaggle T4 GPU should be enough, and another manga bubble detector reports training on a T4 at 1280px. Ultralytics weights are AGPL-3.0, so state the license in the README.

**Done when:** on the test set, most text regions are boxed with sensible IDs, and you can eyeball the overlay images.

## Phase 2: Multimodal Read, Emotion, Translate

- Pick 2 to 3 open vision-language models that fit in 8GB VRAM (quantized, 7B-class) and compare how well each reads Japanese/Korean/Chinese text directly off the image
- Input per panel: the full panel image plus the list of region IDs and boxes
- Output per ID, as structured JSON: source text, detected emotion/tone, English translation
- Same call also handles guardrails: flag obscene or problematic content per ID
- Iterate on the prompt until the JSON output is reliably parseable

Also set up the Phase 5 baselines (text-only translation and a BLIP-style captioner) behind the same input and output format, so every variant reads the same detector output.

**Done when:** the chosen model reads the test pages accurately enough that OCR feels unnecessary, or you've measured exactly where it fails.

## Phase 3: Erase and Render

- Build the text mask for each region (original text pixels only, not the whole bubble)
- Inpaint: simple fill for flat white bubbles, OpenCV inpainting for text over art
- Render the English text into the bubble: line wrapping, auto font sizing to fit, centered placement
- Pick one or two comic-style fonts, and vary style by emotion tag later (e.g. bold for shouting)
- Free-floating text and sound effects get handled last, since they are the hardest

**Done when:** translated pages look clean and readable on the test set without manual touch-ups for bubble text.

## Phase 4: Automation

- One command: point it at a chapter folder or CBZ, get a translated folder back
- Loop over pages sequentially, with per-page logging
- Resume support so a crash on page 30 doesn't redo pages 1 to 29
- Cache model outputs per page so re-rendering doesn't re-run the model

**Done when:** a full chapter runs start to finish unattended.

## Phase 5: Evaluation

- LLM-as-judge compares the pipeline's English against the official reference translation
- Score on accuracy, tone match, and readability; also log obvious failures (missed boxes, overflowing text)
- Keep a small results table per model and per prompt version so changes are comparable

**Comparison study.** Run four variants on the same Phase 0 pages, with the same detector output and the same judge, to test whether image and emotion context actually improves the translation.

| Variant | Reads text with | Context source | Translates with |
| --- | --- | --- | --- |
| A: Text-only | OCR or VLM transcription | None | Text LLM |
| B: Captioner | OCR or VLM transcription | BLIP-style scene caption | Text LLM |
| C: VLM one-pass (MVP) | VLM | VLM sees the full panel | Same VLM |
| D: VLM + OCR fallback (v2) | VLM, OCR when the judge flags it | VLM sees the full panel | Same VLM |

Record per variant: judge score against the reference translation, tone match, text-reading errors, seconds per page and peak VRAM. If C does not beat A, report that: it is a finding, not a failure.

**Done when:** you have scores for variants A to D and a list of the top failure types.

## Phase 6 (v2): Improvements After the MVP

- OCR as a conditional fallback: triggered by the judge when it sees low confidence or text misalignment
- Better detector trained on manga-specific data
- Better inpainting for text over complex art
- Emotion-aware font and lettering styles
- Optional: speed-ups, batching, a simple UI

## Risks to Watch

- Multimodal models may misread stylized fonts or sound effects; Phase 2 testing will show how bad
- Plain OpenCV may struggle with irregular panels and free-floating text
- 8GB VRAM limits model size, so response quality and speed tradeoffs are likely
- Text fitting: English is often longer than the original and may not fit small bubbles

Also: public model pages can disappear (the Manga-Bubble-YOLO page could not be opened), so keep local copies of the weights you use and check each license before reuse.

## Open Questions

- Which open multimodal model to prototype with first
- Whether SFX should be translated, left as-is, or annotated
- What chapter source to use for test data and references
