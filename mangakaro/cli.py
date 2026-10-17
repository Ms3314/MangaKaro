"""Command line: `mangakaro translate book.pdf` and `mangakaro check`."""
import argparse
import sys
from pathlib import Path

from . import __version__, llm
from .pipeline import STAGES, SetupError, Settings, check_setup, run


def add_model_options(p):
    d = Settings()
    p.add_argument("--emotion-model", default=d.emotion_model, help="Ollama vision model that tags emotion and scene (default: %(default)s)")
    p.add_argument("--translate-model", default=d.translate_model, help="Ollama text model that translates (default: %(default)s)")
    p.add_argument("--reader", choices=["ocr", "vlm"], default=d.reader, help="read text with manga-ocr, or with a vision model (default: %(default)s)")
    p.add_argument("--read-model", default=d.read_model, help="vision model used when --reader vlm (default: %(default)s)")
    p.add_argument("--ollama-url", default=d.ollama_url, help="Ollama server (default: %(default)s)")


def settings_from(a):
    return Settings(emotion_model=a.emotion_model, translate_model=a.translate_model, reader=a.reader, read_model=a.read_model,
                    ollama_url=a.ollama_url, pages=getattr(a, "pages", None), max_side=getattr(a, "max_side", Settings.max_side),
                    split_spreads=not getattr(a, "no_split_spreads", False), fonts_file=getattr(a, "fonts", None),
                    redo=getattr(a, "redo", None) or [])


def cmd_translate(a):
    source = Path(a.input).expanduser()
    if not source.exists():
        print(f"error: {source} does not exist", file=sys.stderr)
        return 1
    if source.is_file() and source.suffix.lower() != ".pdf":
        print("error: the input must be a .pdf file or a folder of images", file=sys.stderr)
        return 1
    output = Path(a.output).expanduser() if a.output else source.with_name(f"{source.stem}.translated.pdf")
    work = Path(a.work_dir).expanduser() if a.work_dir else output.with_name(f"{output.stem}.work")
    s = settings_from(a)
    print(f"input   {source}\noutput  {output}\nwork    {work}\nmodels  emotion {s.emotion_model}, translate {s.translate_model}, "
          f"reader {'manga-ocr' if s.reader == 'ocr' else s.read_model}\n")
    try:
        summary = run(source, output, work, s, log=lambda m: print(m, flush=True))
    except SetupError as e:
        print(f"\nsetup problem:\n{e}\n\nRun `mangakaro check` to re-test after fixing.", file=sys.stderr)
        return 1
    print(f"\nDone in {summary['seconds']}s: {summary['pages']} pages ({summary['pages_with_text']} with text), "
          f"{summary['typeset']} text blocks typeset, {summary['skipped']} skipped, {summary['overflow']} overflowing.")
    if summary["pages_failed"]:
        print(f"WARNING: {len(summary['pages_failed'])} page(s) failed and were left unchanged: {', '.join(summary['pages_failed'])}")
    print(f"Wrote {output}")
    return 0


def cmd_check(a):
    problems = check_setup(settings_from(a))
    if not problems:
        print("ready: weights, python packages, fonts and Ollama models are all in place.")
        return 0
    print("not ready:\n\n" + "\n\n".join(problems), file=sys.stderr)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mangakaro", description="Translate manga (Japanese to English) with local models. PDF in, PDF out.")
    ap.add_argument("--version", action="version", version=f"mangakaro {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("translate", help="translate a PDF (or a folder of page images) into a new PDF")
    t.add_argument("input", help="a .pdf file, or a folder of page images")
    t.add_argument("-o", "--output", help="output PDF (default: <input>.translated.pdf next to the input)")
    t.add_argument("--work-dir", help="where page images and cached model results go (default: <output>.work). Re-running resumes from it")
    t.add_argument("--pages", help="only these pages, 1-based, e.g. 1-5,8")
    t.add_argument("--max-side", type=int, default=Settings.max_side, help="longest side in pixels of the pages the models see and the output has (default: %(default)s)")
    t.add_argument("--no-split-spreads", action="store_true", help="do not split two-page spreads into right and left pages")
    t.add_argument("--fonts", type=Path, help="font picks JSON to use instead of choosing fonts automatically")
    t.add_argument("--redo", nargs="+", choices=STAGES, metavar="STAGE", help=f"ignore the cache for these stages and every later one: {', '.join(STAGES)}")
    add_model_options(t)
    t.set_defaults(func=cmd_translate)

    c = sub.add_parser("check", help="check that weights, packages, fonts and Ollama models are ready")
    add_model_options(c)
    c.set_defaults(func=cmd_check)

    a = ap.parse_args(argv)
    llm.configure(a.ollama_url)
    return a.func(a)
