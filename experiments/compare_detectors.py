"""Phase 1 detector comparison.

Runs three pretrained detectors on every page in a folder and writes, per detector:
  output/detector_compare/<detector>/<page>.png   overlay with boxes
  data/intermediate/<detector>/<page>.json        region id, type, bbox [x1,y1,x2,y2], conf
and one summary table (output/detector_compare/summary.md) with counts, sec/page, peak RSS.

Usage: .venv/bin/python -m experiments.compare_detectors [input_dir] [--limit N]
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import cv2

from mangakaro.detect import BubblesYolo, ComicTextDetector, PanelsText, assign_ids, draw
from mangakaro.paths import ROOT


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # macOS bytes, linux KB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", nargs="?", default=str(ROOT / "manga/input/jjk_demo"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    pages = sorted(Path(a.input_dir).glob("*.png"))[: a.limit]
    summary = []
    for cls in (PanelsText, BubblesYolo, ComicTextDetector):
        det = cls()
        (ROOT / "output/detector_compare" / det.name).mkdir(parents=True, exist_ok=True)
        (ROOT / "data/intermediate" / det.name).mkdir(parents=True, exist_ok=True)
        det(cv2.imread(str(pages[0])))  # warm-up, excluded from timing
        times, counts = [], {}
        for p in pages:
            img = cv2.imread(str(p))
            t = time.perf_counter()
            regs = det(img)
            times.append(time.perf_counter() - t)
            regs = assign_ids(regs)  # reading-order IDs; page flow is right-to-left like the VIZ manga edition
            for g in regs:
                g["id"] = f"{p.stem}_{g['label']}"
                counts[g["type"]] = counts.get(g["type"], 0) + 1
            cv2.imwrite(str(ROOT / "output/detector_compare" / det.name / p.name), draw(img, regs))
            (ROOT / "data/intermediate" / det.name / f"{p.stem}.json").write_text(
                json.dumps({"page": p.name, "size": img.shape[:2][::-1], "regions": regs}, indent=1))
        summary.append((det.name, counts, sum(times) / len(times), peak_rss_mb()))
        print(det.name, counts, f"{summary[-1][2]:.2f}s/page", f"{summary[-1][3]:.0f}MB peak RSS (cumulative)")
        del det
    lines = ["| detector | regions by type | sec/page | peak RSS MB (cumulative) |", "|---|---|---|---|"]
    lines += [f"| {n} | {c} | {t:.2f} | {m:.0f} |" for n, c, t, m in summary]
    (ROOT / "output/detector_compare/summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
