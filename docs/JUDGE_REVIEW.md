# Cloud arms vs local: reviewed by Claude Sonnet 5.5

<sub>[← README: Cloud vs local](../README.md#cloud-vs-local) · [Docs index](README.md)</sub>

Reviewed 2026-10-20 by Claude Sonnet 5.5 acting as the judge. The four pages are Manga109 `006_L`, `005_L`, `009_L` and `002_R`.
Each cloud arm was compared with the local run (Gemma 4 through Ollama) page by page, then in three groups.
The structured version of every verdict is in `docs/judge_verdicts.json`. The sheets and per-line tables were built by
`experiments/judge_pack.py` and are in `manga/output/judge/` (git-ignored, because the pages are copyrighted).

**Read the caveats at the bottom before quoting any of this.** One judge, four pages, no reference translation.

## Setup

| | |
| --- | --- |
| Local | `manga/output/compare-local-timing/D-local/pages/`, run from scratch today: 165 s for four pages (about 41 s a page), $0 |
| sol-B / sol-C | `gpt-6.1-sol` with our OCR / `gpt-6.1-sol` also reading (earlier run) |
| image-2 / image-mini | `gpt-image-2` / `gpt-image-1-mini`, page in, page out |
| mini-B / mini-C | `gpt-5.4-mini` with our OCR / `gpt-5.4-mini` also reading |

Rubric (1 to 5 each, scores in `verdicts.json`): **fidelity** to the Japanese, **no invention**, **naturalness** of the English,
**art preserved**, **lettering fit**. Per page a pair gets a winner: `cloud`, `local` or `tie`.

## Scores by arm (mean over the four pages, my judgment)

| Arm | Fidelity | No invention | Natural | Art kept | Fit | $ / page | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local (Gemma 4) | 3.0 | 5.0 | 3.5 | 5.0 | 3.5 | 0 | 41 s (165 s for 4) |
| sol-B | 4.0 | 5.0 | 4.5 | 5.0 | 3.5 | 0.021 | about 44 s* |
| sol-C | 4.5 | 5.0 | 4.5 | 5.0 | 3.5 | 0.034 | about 53 s* |
| mini-B | 3.5 | 5.0 | 4.0 | 5.0 | 3.5 | 0.006 | 61 s for 4 (47 s was model loading) |
| mini-C | 2.5 | 4.5 | 4.0 | 5.0 | 3.0 | 0.009 | 57 s for 4 |
| image-2 | 1.5 | 1.0 | 3.0 | 3.0 | 4.0 | 0.049 | 34 s |
| image-mini | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | 0.015 | 18 s |

\*Estimate: the measured 40 and 52 s left out box detection (and OCR for sol-B), which were reused. About 5 s of detection and 13.6 s of OCR for the 4 pages were added from the local run.

## Each cloud arm against local

### 1. `gpt-6.1-sol` (best cloud pipeline) vs local: cloud wins on translation, a draw on rendering

| Page | sol-B | sol-C | Evidence |
| --- | --- | --- | --- |
| 006_L | tie | tie | sol reads ムシのいい as "cushy"; local says "crappy". sol's bottom-row bubbles cross the outline; local's fit. |
| 005_L | cloud | cloud | Local: へんぺい足 (flat feet) became "a lot of standing around"; ノーパン became "no-bra". sol has both right. |
| 009_L | cloud | cloud | Local reverses the sukiyaki line and writes "Neshima-san". sol has "Narushima" and the right meaning. |
| 002_R | tie | tie | Both accurate; both leave かんげき!! and しーん in Japanese. |

Local has six clear errors across the four pages: "crappy", "no-bra", "a lot of standing around", the reversed sukiyaki line, "Neshima-san" and "Yaga" for 平井. sol-B has one ("Yaga", inherited from the OCR). sol-C has none that I can see.
sol's English is longer, so the bottom rows of 005_L and 006_L are slightly more crowded.

### 2. More on sol: reading with the cloud model (sol-C) vs our OCR (sol-B)

sol-C corrected three things manga-ocr got wrong: 荷が重そう ("a lot of responsibility", the OCR had 何が重そう), のんでるよ (the OCR had のんでうるよ) and the speaker name in a small
009_L bubble: the bubble reads 平井くん with the furigana ひらい above it (I checked the page at 4x), so "Hirai" is right and "Yaga" is wrong. Everything else is the same. That is a small gain for a 60% higher price, so **sol-B is the better value, sol-C the more accurate**.

### 3. `gpt-image-2` vs local: local wins on all four pages

It looks like a manga page but it is a different page. It invented the content of nearly every large bubble on 006_L ("messing around with Bami", "Kimi is the one who will protect her"),
a calligraphy teacher, tamagoyaki and a weightlifting gym on 005_L, "Asuka" and a sound effect on 009_L. On 002_R it translated the display lettering but wrongly ("I'll do my best!!" for かんげき) and wrote "Cooked White Rice" on a box that has no such text.
Bubbles and panels are redrawn and the page is shifted. Its only edge is that it tries the display lettering.

### 4. `gpt-image-1-mini` (cheaper image model) vs local: local wins on all four pages

Garbled words ("ridisdloce", "autfoszer"), panel layouts changed, a different drawing of each page. Unusable.

### 5. `gpt-5.4-mini` with our OCR (mini-B) vs local: slight cloud edge

| Page | Winner | Evidence |
| --- | --- | --- |
| 006_L | tie | "insanely easy" is better than "crappy". Fit is the same. |
| 005_L | tie | Fixes the no-pants cafe, but "N-no, nothing like that!" is wrong for コマゴマしたのダメ and "ugly swollen legs" misses flat feet. |
| 009_L | cloud | Sukiyaki line and "Narushima" right. |
| 002_R | tie | Both accurate. |

### 6. Mini with the model reading too (mini-C) vs local: local edges it

| Page | Winner | Evidence |
| --- | --- | --- |
| 006_L | local | The two top bubbles stay in Japanese: the reader returned nothing for them. |
| 005_L | cloud | "no-panty café", "pays next to nothing" are better than local's. |
| 009_L | local | Misreads: "Hirai, that place is really great" (right name, wrong sentence) and "the likes of Suki" for スキヤキ. |
| 002_R | tie | Both fine. |

The failure is silent: nothing in the output says two bubbles were skipped except the log line.

## Groups

| Group | Ranking, best to worst | Why |
| --- | --- | --- |
| All image models vs local | local, image-2, image-mini | Neither image model keeps the text faithful. image-2 is readable but invents; image-mini is garbled. |
| All `gpt-6.1-sol` arms vs local | sol-C, sol-B, local | The two sol arms are close. Both beat local on translation. Rendering is the same. |
| All `gpt-5.4-mini` arms vs local | mini-B, local, mini-C | mini-B is a little better than local. mini-C is the worst pipeline arm, but close to local: it beat local on 005_L and lost on 006_L and 009_L. |

**Best value:** mini-B, at $0.006 a page and about 5 s a page once loaded. **Best accuracy:** sol-B or sol-C. **Best at $0:** local, which is
about as good as mini-B on layout but has more translation errors. **Not worth using:** both image models and mini-C.

## Second pass (what was checked, and what could not be)

A second blind pass was set up (`experiments/judge_pack2.py`: each page split in halves at 1.6x, new random sides, no line tables). **It was not blind.**
The same session judged both passes, so I recognised the sides from the text on the first sheets I opened (006_L, sol-B and sol-C) and stopped there.
A judge that remembers the first pass cannot give an independent second one. What I did instead was check the claims the verdicts rest on against the original page at 3 to 4x:

| Claim | Original page | Result |
| --- | --- | --- |
| 009_L speaker name | 平井くん with ひらい above it | Hirai is right. local, sol-B and mini-B wrote "Yaga". |
| 009_L 成島 | furigana なるしま | "Narushima" is right, local's "Neshima-san" is wrong. |
| 005_L 荷が重そう | the page has 荷 | manga-ocr read 何, so local, sol-B and mini-B inherit the misread. Their English ("feels heavy") is still close. sol-C and mini-C read it right. |
| 009_L のんでるよ | page says のんでるよ | manga-ocr read のんでうるよ. sol-C read it right. The English barely changed. |
| 006_L fit, higher resolution | | Same picture as before: the cloud arms' bottom-row bubbles are tighter and "It's just... you know?" touches the outline, local's fit. |

No winner changed. Two things did: the misreads in the OCR are now counted as shared errors of every arm that uses it, and sol-C's reading advantage is confirmed rather than suspected.
An independent second judgment needs a separate judge call with no memory of this one, which is what the automated judge below is for.

## Weaknesses shared by every pipeline arm (so not a model difference)

- Display lettering (かんげき!!, しーん) and sound effects stay in Japanese.
- 今のいい方 on 009_L is left in Japanese in every arm.
- "Huh? Part-time work?" and "Tutoring means educating kids..." on 005_L cross their bubble outline in every arm.
- Narration text on 009_L is about 8 px.

## Caveats

- One judge (me), four pages, no reference translation. My reading of the Japanese is not a native reader's. The 平井 reading on 009_L was checked against the page; the other readings were not checked beyond my own.
- **Not blind, and a second blind pass was not possible in this session** (see the section above). I read the per-line tables, which show local's English, before I read the sheets. The image arms are obvious on sight. The pipeline arms share a renderer, so I told them apart by their text.
- Fit scores are nearly equal because the renderer is shared; they say little about the models.
- The time column mixes measurements: local is the full path run today, the sol times leave out detection and OCR, the mini times include loading the models on the first page.
- Prices were copied from OpenAI's pricing page on 2026-10-19 by a tool: check them.

## Using this as a reference for an automated judge

`verdicts.json` can be the gold set. A judge prompt that matches it needs: the original page, two candidates in random order labelled X and Y, the per-line table when both are pipeline arms,
the five rubric dimensions with their 1 to 5 anchors above, and a JSON reply `{scores_x, scores_y, winner, why}`. Compare its winners with the table above: 24 pairs, of which 8 are ties.
Randomise the order, run each pair twice with the order swapped, and count a disagreement between the two runs as a tie.
