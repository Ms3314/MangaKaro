"""Where things live. Weights default to <project>/weights and can be moved with MANGAKARO_WEIGHTS."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.environ.get("MANGAKARO_WEIGHTS", ROOT / "weights"))
EXTRA_FONTS = ROOT / "fonts"  # drop .ttf/.otf files here to add fonts (check each font's license first)

# Weight files the pipeline needs, and where they come from.
WEIGHT_FILES = {
    "comic-panels-and-text-detect.safetensors": "https://huggingface.co/Oliverdsfdsf/comic-panels-text-detect (Apache-2.0)",
    "bubbles_detect.pt": "https://huggingface.co/PSImera/manga_bubbles_detect (MIT)",
    "comictextdetector.pt.onnx": "https://github.com/zyddnys/manga-image-translator/releases (comic-text-detector, GPL-3.0)",
}
