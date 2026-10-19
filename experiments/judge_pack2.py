"""Second blind pass for judging: pipeline arms vs local, each page split into a left and a right half and shown at 1.6x so small lettering can be read.

Each sheet is the original half, then candidates X and Y in random order. A new seed puts local on the other side from `judge_pack.py` in most
pairs. No per-line tables are written. The key goes to manga/output/judge2/key.json and should not be opened until the verdicts are written.

  .venv/bin/python -m experiments.judge_pack2"""
import json
import random

import cv2
import numpy as np

from experiments.judge_pack import ARMS, LOCAL, LOC, PAGES, label, load
from mangakaro.paths import MANGA

OUT = MANGA / "output" / "judge2"
PIPELINE = [a for a in ARMS if not a.startswith("image")]
SCALE = 1.6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng, key = random.Random(909), {}
    for page in PAGES:
        orig = cv2.imread(str(LOC / "input" / f"{page}.png"))
        size = (orig.shape[1], orig.shape[0])
        loc = load(LOCAL, page, size)
        mid = size[0] // 2
        for name in PIPELINE:
            arm = load(ARMS[name], page, size)
            first_local = rng.random() < 0.5
            x, y = (loc, arm) if first_local else (arm, loc)
            safe = name.split(" ")[0]
            key[f"{safe}__{page}"] = {"X": "local" if first_local else name, "Y": name if first_local else "local"}
            for side, (a, b) in {"left": (0, mid), "right": (mid, size[0])}.items():
                panels = [orig[:, a:b], label(x[:, a:b], "X"), label(y[:, a:b], "Y")]
                sheet = cv2.resize(np.hstack(panels), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(OUT / f"{safe}__{page}__{side}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (OUT / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"wrote {len(key) * 2} sheets for {len(key)} pairs to {OUT}")


if __name__ == "__main__":
    main()
