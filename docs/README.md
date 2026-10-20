# Docs

| File | What it is | Read it when |
| --- | --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The pipeline, every module, the models, result formats, the design decisions with their evidence, thresholds, the cloud-comparison harness | you want to change the code or understand why it is built this way |
| [PROGRESS.md](PROGRESS.md) | The full record: what was built in what order, each experiment (S0 to S7, LaMa, Manga109, cloud models), the mistakes, what is not measured | you want the evidence or a write-up |
| [JUDGE_REVIEW.md](JUDGE_REVIEW.md) | The cloud arms against local, page by page, judged by Claude Sonnet 5.5 with a rubric, the second check, the caveats | you want the detail behind the README's "Cloud vs local" |
| [judge_verdicts.json](judge_verdicts.json) | The same verdicts and rubric scores as data, with the corrections made after checking the page | you are building an automated judge and need a gold set to compare against |
| `images/` | The before/after strips, the comparison strips and the header used by the README | |

Back to the [README](../README.md): [Cloud vs local](../README.md#cloud-vs-local) · [How it works](../README.md#how-it-works) · [Use it](../README.md#use-it).

The sheets the judge looked at (original plus two candidates) are not here: they contain the Manga109 pages and are written to `manga/output/judge/` by
`experiments/judge_pack.py`, which is git-ignored.
