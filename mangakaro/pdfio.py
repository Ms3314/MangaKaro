"""PDF in, PDF out.

Reading uses pypdfium2 (permissive license). Writing uses Pillow, which stores each page as a JPEG image inside the PDF,
so output pages are raster images of the translated pages, not selectable text.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def pdf_page_count(path):
    import pypdfium2 as pdfium
    return len(pdfium.PdfDocument(str(path)))


def pdf_pages(path, wanted, height=1536):
    """Yield (page_number, BGR image) for the wanted 1-based page numbers, rendered so the page is `height` pixels tall."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    for n in sorted(wanted):
        page = pdf[n - 1]
        _, h = page.get_size()
        pil = page.render(scale=height / h).to_pil().convert("RGB")
        yield n, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def images_to_pdf(images, out_path, quality=92, resolution=150.0):
    """Write BGR images (numpy arrays) as the pages of one PDF."""
    pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images]
    if not pil:
        raise ValueError("no pages to write")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pil[0].save(str(out_path), "PDF", save_all=True, append_images=pil[1:], quality=quality, resolution=resolution)
