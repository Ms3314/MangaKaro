# mangaKaro

Translate manga from Japanese to English with local models. Give it a PDF or a set of page images, and get the translated
pages back as a PDF or as images.

```
mangakaro translate chapter.pdf          →  chapter.translated.pdf
mangakaro translate pages/               →  pages_translated/   (one translated image per page)
mangakaro translate pages/*.png -o out/  →  out/               (the same, from a list of files)
```

Everything runs on your own machine: no cloud services, no API keys. It finds the text on each page, reads it with an OCR
model built for manga, asks a vision model to judge the emotion of each line, translates the whole page with a text model,
then erases the Japanese and typesets the English inside the speech bubbles.

**Status: a working prototype.** It handles plain speech bubbles well. Sound effects and text over artwork are left in
Japanese. Translation quality has not been scored against a reference. See [Known limitations](#known-limitations) and
[docs/PROGRESS.md](docs/PROGRESS.md).

## What you need

- A Mac with Apple silicon was used for development. Other machines should work but are untested.
- Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/) (or pip).
- [Ollama](https://ollama.com) running locally, with a vision-capable model. The default is `gemma4:e4b-mlx` (an Apple
  silicon build). On other machines pick another vision model from Ollama's library and pass it with `--emotion-model` and
  `--translate-model`.
- Three model weight files in `weights/` (see below). They are not included in the repository.
- Disk and memory for the models: the default Ollama model is about 9.5 GB.

## Install

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
ollama pull gemma4:e4b-mlx
```

The first run also downloads the OCR model (`kha-white/manga-ocr-base`) from Hugging Face.

### Weights

Put these files in `weights/` (or set `MANGAKARO_WEIGHTS` to another folder). Check each model's license before reuse.

| File | Source |
| --- | --- |
| `comic-panels-and-text-detect.safetensors` | https://huggingface.co/Oliverdsfdsf/comic-panels-text-detect |
| `bubbles_detect.pt` | https://huggingface.co/PSImera/manga_bubbles_detect |
| `comictextdetector.pt.onnx` | the `comic-text-detector` release linked from https://github.com/zyddnys/manga-image-translator/releases |

`bubbles_detect.pt` is a pickle file: only load it from the source above.

**Optional, recommended:** `weights/big-lama.pt` (about 196 MB, TorchScript) from
https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt. If it is present, text over screentone
and line art is erased with LaMa; if not, OpenCV's basic inpainting is used. It is loaded with `torch` directly. Do not
`pip install simple-lama-inpainting`: it pins old numpy, opencv and pillow and would downgrade them.

### Check the setup

```bash
.venv/bin/mangakaro check
```

This checks the weights, the Python packages, a usable English font, that Ollama is running, and that the models you asked
for are installed. It prints exactly what is missing and how to fix it.

## Use it

```bash
# a PDF in, a PDF out (written next to the input as book.translated.pdf)
.venv/bin/mangakaro translate book.pdf

# a folder of page images in, a folder of translated images out (pages_translated/)
.venv/bin/mangakaro translate pages/

# several image files in (the shell expands the *), images out into a folder of your choice
.venv/bin/mangakaro translate pages/*.png -o translated/

# any input can produce a PDF instead: end the output name in .pdf
.venv/bin/mangakaro translate pages/ -o book-en.pdf

# try the first three pages first
.venv/bin/mangakaro translate book.pdf --pages 1-3
```

Image files are put in natural order (`page2` before `page10`). Supported image types: `.png`, `.jpg`, `.jpeg`, `.webp`.
The page order is the order of the files, so name them so that they sort in reading order.

Pages are processed one at a time with a progress line each. If a run is interrupted, run the same command again: finished
pages and finished model calls are reused from the work directory.

| Option | What it does |
| --- | --- |
| `-o, --output` | a path ending in `.pdf` writes one PDF; any other path is a folder that receives the translated page images. Default: `<input>.translated.pdf` for a PDF, `<folder>_translated/` for a folder, `./translated/` for a list of files |
| `--work-dir` | where page images and cached results go (default `<output>.work`) |
| `--pages 1-5,8` | only these pages (1-based position in the PDF, or in the sorted list of images) |
| `--emotion-model`, `--translate-model` | Ollama models for the emotion and translation steps |
| `--reader ocr\|vlm` | read text with manga-ocr (default) or with a vision model; `--read-model` sets which |
| `--max-side 1536` | longest side in pixels of the pages the models see and the output has |
| `--plain-only` | only translate text in plain white bubbles; leave text on dark areas and over art alone |
| `--no-lama` | erase text over art with OpenCV's basic inpainting instead of LaMa (a little faster, smears more) |
| `--no-check` | skip the check for Japanese left on each finished page |
| `--no-split-spreads` | keep two-page spreads as one page |
| `--fonts picks.json` | use these fonts instead of choosing them automatically |
| `--redo translate` | ignore cached results from this step onward (`read`, `emotion`, `translate`) |
| `--ollama-url` | Ollama server address (default `http://localhost:11434`) |

### What you get

- One output page per input page, translated or not, as one PDF or as one PNG per page (named after the input file, with `_R`
  and `_L` added for a split two-page spread). Spreads become two pages, right page first.
- The work directory keeps everything: `pages/` (what the models saw), `rendered/` (translated pages as images),
  `results/` (the Japanese text, emotion and English for every box), `fonts.json` and `fonts.png` (the fonts chosen and a
  sample sheet), `report.json` (what happened to every text block) and `summary.json`.
- Output pages are raster images, so the text in an output PDF is not selectable.

### How long it takes

On the development Mac with the default models, about 20 to 60 seconds per page (a 5-page test took 176 seconds in total).
The vision step is the slowest part. A 20-page chapter takes roughly 10 to 20 minutes.

## How it works

1. Pages are loaded (rendered from the PDF, or read from the image files), two-page spreads are split, and each page is shrunk to at most 1,536 px.
2. A detector finds text boxes and panels, and numbers the boxes in reading order (right to left, top to bottom).
3. manga-ocr reads each box.
4. A vision model looks at the page and the text and tags each line with speaker, emotion and intensity.
5. A text model translates the whole page at once, using the emotion tags, so voices stay consistent.
6. Once per manga, the weight of its lettering is measured and a primary and a secondary English font are chosen.
7. The Japanese letters are erased by their pixel mask. In plain bubbles the English is fitted inside the bubble; on flat black it is
   drawn in white; over screentone or art the area is inpainted with LaMa and the English is drawn black with a white outline.
8. A check looks for Japanese still on the finished page (`checks/<page>.json`).

The full design, the data formats and the reasons behind each choice are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Known limitations

- **Japanese to English only.**
- **Text over art is a first version.** Plain bubbles are the reliable case. White lettering on flat black panels works well. Lettering over screentone or drawn hair is erased with LaMa (neural inpainting), which continues the texture but can leave small marks, and its English is drawn black with a white outline. `--plain-only` turns this off and leaves such text in Japanese. Sound effects and background signs are not translated.
- **OCR mistakes reach the translation.** Short vertical strings can be misread or given extra characters, and the translator
  tends to turn garbled text into a confident, wrong sentence. The same page can read slightly differently after a PDF
  round trip, because the page images are re-encoded.
- **Narrow bubbles get small text,** and in a few bubbles the English still touches the outline.
- **Not scored.** The quality notes in the docs come from looking at the output. There are no reference translations yet.
- **Fonts:** only installed fonts are used (Comic Sans MS and a few others on a Mac) and they are matched on weight, not shape.
  Add `.ttf` or `.otf` files to `fonts/` to widen the choice, after checking each font's license.

## Project layout

```
mangakaro/        the package: cli, pipeline, detect, ocr, stages, prompts, fonts, render, pages, pdfio, llm
experiments/      scripts used to decide the design (detector comparison, one-shot translation, review sheets)
tests/            unit tests for everything that needs no models
docs/             ARCHITECTURE.md, PROGRESS.md
data/             cached results from the experiments (JSON only)
weights/          model weights (not in the repository)
```

`Manga Translation Pipeline — Phase-wise MVP Plan.md` is the original plan, with the current status of each phase added at the top.

## Tests

```bash
.venv/bin/python -m unittest discover tests -v
```

The tests cover page handling, the PDF round trip, OCR cleanup, reading order, text fitting and the font weight measure. They
need no models. The full pipeline was checked by running the command on a 5-page PDF.

## Licenses

This project's own code has no license file yet. It depends on models and libraries with different licenses: the
text-box detector weights are Apache-2.0, the bubble detector is MIT, the lettering-mask detector is GPL-3.0, manga-ocr is
Apache-2.0, and Ultralytics (used to run the YOLO models) is AGPL-3.0. Check these before distributing anything built on
them. The manga pages used for testing are copyrighted and are not in the repository.
