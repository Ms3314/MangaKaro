"""Builds the material for judging every cloud arm against the local run, page by page.

For each pair (one cloud arm vs local) and each page it writes a blind sheet: original on the left, then the two candidates in
random order labelled X and Y (which one is local is only in key.json). For each group (all image models, all gpt-6.1-sol arms,
all gpt-5.4-mini arms) it writes one sheet with the original, local and the group's arms in random order, labelled by letter.
For the pipeline arms it also writes a per-line table of the Japanese the arm read and the English it produced, next to local's.

Run from the project root, no API calls are made:
  .venv/bin/python -m experiments.judge_pack
Output: manga/output/judge/{pairs,groups,lines}/ and manga/output/judge/key.json"""
import json
import random

import cv2
import numpy as np

from mangakaro.paths import MANGA

OUT = MANGA / "output" / "judge"
V2, V1, LOC = MANGA / "output" / "compare-openai-v2", MANGA / "output" / "compare-openai", MANGA / "output" / "compare-local-timing"
PAGES = ["006_L", "005_L", "009_L", "002_R"]
LOCAL = LOC / "D-local"
# name -> folder holding pages/<page>.png (and work/results/<page>.json for pipeline arms)
ARMS = {
    "sol-B (gpt-6.1-sol + our OCR)": V1 / "B-gpt-6.1-sol",
    "sol-C (gpt-6.1-sol also reads)": V1 / "C-gpt-6.1-sol",
    "image-2 (gpt-image-2)": V2 / "A-gpt-image-2",
    "image-mini (gpt-image-1-mini)": V2 / "A-gpt-image-1-mini",
    "mini-B (gpt-5.4-mini + our OCR)": V2 / "B-gpt-5.4-mini",
    "mini-C (gpt-5.4-mini also reads)": V2 / "C-gpt-5.4-mini",
}
PAIRS = list(ARMS)
GROUPS = {
    "all-images": ["image-2 (gpt-image-2)", "image-mini (gpt-image-1-mini)"],
    "all-sol": ["sol-B (gpt-6.1-sol + our OCR)", "sol-C (gpt-6.1-sol also reads)"],
    "all-mini": ["mini-B (gpt-5.4-mini + our OCR)", "mini-C (gpt-5.4-mini also reads)"],
}


def load(folder, page, size=None):
    im = cv2.imread(str(folder / "pages" / f"{page}.png"))
    return cv2.resize(im, size) if size else im


def label(im, text):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (70, 62), (255, 255, 255), -1)
    cv2.putText(im, text, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 4)
    return im


def lines(folder, page):
    f = folder / "work" / "results" / f"{page}.json"
    if not f.exists():
        return None
    return {r["id"]: (r.get("source_text", ""), r.get("translation") or "") for r in json.loads(f.read_text())["regions"]}


def main():
    for d in ("pairs", "groups", "lines"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rng, key = random.Random(55), {"pairs": {}, "groups": {}}
    for page in PAGES:
        orig = cv2.imread(str(LOC / "input" / f"{page}.png"))
        size = (orig.shape[1], orig.shape[0])
        loc = load(LOCAL, page, size)
        for name in PAIRS:
            arm = load(ARMS[name], page, size)
            first_local = rng.random() < 0.5
            x, y = (loc, arm) if first_local else (arm, loc)
            safe = name.split(" ")[0]
            cv2.imwrite(str(OUT / "pairs" / f"{safe}__{page}.jpg"), np.hstack([orig, label(x, "X"), label(y, "Y")]), [cv2.IMWRITE_JPEG_QUALITY, 88])
            key["pairs"][f"{safe}__{page}"] = {"X": "local" if first_local else name, "Y": name if first_local else "local"}
        for g, members in GROUPS.items():
            panels = [("local", loc)] + [(m, load(ARMS[m], page, size)) for m in members]
            rng.shuffle(panels)
            key["groups"][f"{g}__{page}"] = {chr(65 + i): n for i, (n, _) in enumerate(panels)}
            row = [orig] + [label(im, chr(65 + i)) for i, (_, im) in enumerate(panels)]
            cv2.imwrite(str(OUT / "groups" / f"{g}__{page}.jpg"), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 88])
        # per-line text: local vs every pipeline arm
        loc_lines, md = lines(LOCAL, page), [f"# {page}: Japanese read and English written, per box\n"]
        header = ["id", "local read", "local English"]
        arm_lines = {n: lines(ARMS[n], page) for n in PAIRS if lines(ARMS[n], page)}
        md.append("| " + " | ".join(header + [f"{n.split(' ')[0]} read | {n.split(' ')[0]} English" for n in arm_lines]) + " |")
        md.append("|" + "---|" * (len(header) + 2 * len(arm_lines)))
        ids = sorted(set(loc_lines) | {i for a in arm_lines.values() for i in a}, key=lambda s: [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in s.split(".")])
        for i in ids:
            row = [i, *loc_lines.get(i, ("", ""))]
            for a in arm_lines.values():
                row += list(a.get(i, ("", "(missing)")))
            md.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
        (OUT / "lines" / f"{page}.md").write_text("\n".join(md) + "\n")
    (OUT / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"wrote {len(key['pairs'])} pair sheets, {len(key['groups'])} group sheets, {len(PAGES)} line tables to {OUT}")


if __name__ == "__main__":
    main()
