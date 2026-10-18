"""Neural inpainting with LaMa (big-lama, a TorchScript file), for erasing lettering over screentone and line art.

The basic inpainting in OpenCV smears a region into a blur. That is fine on white paper and flat black, but over screentone,
hair strokes and faces it leaves cloudy streaks. LaMa continues textures and lines instead. It runs on small crops around
each box (the model is slow on whole pages) and only the masked pixels are pasted back, so nothing else on the page changes.

Weights: weights/big-lama.pt (about 196 MB), from the LaMa project (advimman/lama, Apache-2.0 per its repository; check before
redistributing), as released in https://github.com/enesmsahin/simple-lama-inpainting. It is a TorchScript archive, so load it
only from a source you trust. It runs on the CPU: the model's FFT layers are not reliable on Apple's GPU backend.
"""
import time

import cv2
import numpy as np
import torch

from .paths import WEIGHTS

MODULO = 8  # the network needs sizes divisible by this


class Lama:
    def __init__(self, path=None, device="cpu"):
        self.device = device
        self.model = torch.jit.load(str(path or WEIGHTS / "big-lama.pt"), map_location=device).eval()

    def __call__(self, img, mask):
        """img: BGR uint8. mask: bool/uint8 array, nonzero = erase. Returns BGR uint8 with the masked area filled in."""
        h, w = img.shape[:2]
        ph, pw = (-h) % MODULO, (-w) % MODULO
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
            m = np.pad(m, ((0, ph), (0, pw)), mode="symmetric")
        x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None].to(self.device)
        mk = torch.from_numpy(m.astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(x, mk)[0].permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def erase_regions(self, img, mask, boxes, context=48):
        """Erase `mask` inside each box, looking at `context` pixels around it. Only masked pixels change. Returns the new image."""
        out = img.copy()
        H, W = img.shape[:2]
        for x0, y0, x1, y1 in boxes:
            X0, Y0, X1, Y1 = max(0, x0 - context), max(0, y0 - context), min(W, x1 + context), min(H, y1 + context)
            local = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            inner = mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
            local[max(0, y0) - Y0:max(0, y0) - Y0 + inner.shape[0], max(0, x0) - X0:max(0, x0) - X0 + inner.shape[1]] = inner
            if not local.any():
                continue
            filled = self(out[Y0:Y1, X0:X1], local)
            region = out[Y0:Y1, X0:X1]
            region[local > 0] = filled[local > 0]
        return out
