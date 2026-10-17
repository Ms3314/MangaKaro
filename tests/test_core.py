"""Tests for the parts that need no models: page handling, PDF round trip, OCR cleanup, text fitting, reading order.

Run with:  .venv/bin/python -m unittest discover tests -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mangakaro.detect import assign_ids, merge_text_boxes, reading_order
from mangakaro.fonts import DEFAULT_FONT, font_of, weight_ratio
from mangakaro.ocr import drop_contained, normalize_ellipses
from mangakaro.pages import load_pages, natural_key, parse_pages, resolve_source, shrink, split_spread
from mangakaro.pdfio import images_to_pdf, pdf_page_count
from mangakaro.pipeline import write_output
from mangakaro.render import fit, wrap
from mangakaro import stages
from mangakaro.llm import OllamaTimeout


def box(x0, y0, x1, y1, kind="text"):
    return {"type": kind, "bbox": [x0, y0, x1, y1], "conf": 0.9}


class PageHandling(unittest.TestCase):
    def test_parse_pages(self):
        self.assertEqual(parse_pages("1-3,7", 10), {1, 2, 3, 7})
        self.assertEqual(parse_pages(None, 3), {1, 2, 3})
        self.assertEqual(parse_pages("2-99", 4), {2, 3, 4})  # out-of-range pages are dropped

    def test_spread_is_split_right_page_first(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[:, 100:] = 255  # right half white
        parts = split_spread("p1", img)
        self.assertEqual([n for n, _ in parts], ["p1_R", "p1_L"])
        self.assertEqual(parts[0][1].mean(), 255)
        self.assertEqual(parts[1][1].mean(), 0)

    def test_single_page_and_no_split(self):
        self.assertEqual(len(split_spread("p", np.zeros((200, 100, 3), np.uint8))), 1)
        self.assertEqual(len(split_spread("p", np.zeros((100, 200, 3), np.uint8), split=False)), 1)

    def test_shrink_only_shrinks(self):
        self.assertEqual(max(shrink(np.zeros((3000, 2000, 3), np.uint8)).shape[:2]), 1536)
        self.assertEqual(shrink(np.zeros((300, 200, 3), np.uint8)).shape[:2], (300, 200))


class PdfRoundTrip(unittest.TestCase):
    def test_images_to_pdf_and_back(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "t.pdf"
            images_to_pdf([np.full((300, 200, 3), v, np.uint8) for v in (30, 120, 220)], pdf)
            self.assertEqual(pdf_page_count(pdf), 3)
            pages = list(load_pages(pdf, "2-3"))
            self.assertEqual([p.name for p in pages], ["p0002", "p0003"])
            self.assertEqual(pages[0].image.shape[0], 1536)  # PDF pages are rendered at the working height, whatever their own size

    def test_folder_of_images(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            for i in (1, 2):
                cv2.imwrite(str(Path(d) / f"page {i}.png"), np.zeros((120, 80, 3), np.uint8))
            self.assertEqual([p.name for p in load_pages(d)], ["page_1", "page_2"])  # file-safe names, folder order


class ImageInput(unittest.TestCase):
    def make(self, d, names):
        import cv2
        for n in names:
            if n.endswith(".txt"):
                (Path(d) / n).write_text("not an image")
            else:
                cv2.imwrite(str(Path(d) / n), np.zeros((120, 80, 3), np.uint8))

    def test_natural_order_not_alphabetical(self):
        names = ["page10.png", "page2.png", "page1.png"]
        self.assertEqual(sorted(names, key=natural_key), ["page1.png", "page2.png", "page10.png"])

    def test_folder_is_sorted_naturally(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, ["p10.png", "p2.png", "p1.png", "notes.txt"])
            kind, files = resolve_source(d)
            self.assertEqual((kind, [f.name for f in files]), ("images", ["p1.png", "p2.png", "p10.png"]))

    def test_several_files_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            self.make(d1, ["a.png"])
            self.make(d2, ["a.png", "b.jpg"])
            files = [Path(d2) / "b.jpg", Path(d1) / "a.png", Path(d2) / "a.png"]
            kind, got = resolve_source([str(f) for f in files])
            self.assertEqual(kind, "images")
            pages = list(load_pages([str(f) for f in files]))
            self.assertEqual([p.name for p in pages], ["a", "a_2", "b"])  # same file name from two folders gets a suffix
            self.assertEqual([p.out_name for p in pages], ["a", "a_2", "b"])

    def test_output_keeps_the_input_file_name(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, ["my page (3).png"])
            (page,) = load_pages(d)
            self.assertEqual((page.name, page.out_name), ("my_page_3_", "my page (3)"))  # safe name inside, real name out

    def test_bad_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "notes.txt").write_text("x")
            with self.assertRaises(ValueError):
                resolve_source(str(Path(d) / "notes.txt"))
            with self.assertRaises(ValueError):
                resolve_source(d)  # a folder with no images
            with self.assertRaises(FileNotFoundError):
                resolve_source(str(Path(d) / "missing.png"))

    def test_output_as_folder_or_pdf(self):
        pages = [("p1", np.full((50, 40, 3), 10, np.uint8)), ("p2", np.full((50, 40, 3), 200, np.uint8))]
        with tempfile.TemporaryDirectory() as d:
            write_output(pages, Path(d) / "out")
            self.assertEqual(sorted(p.name for p in (Path(d) / "out").iterdir()), ["p1.png", "p2.png"])
            write_output(pages, Path(d) / "out.pdf")
            self.assertEqual(pdf_page_count(Path(d) / "out.pdf"), 2)


class OcrCleanup(unittest.TestCase):
    def test_ellipsis(self):
        self.assertEqual(normalize_ellipses("あれ．．．"), "あれ…")
        self.assertEqual(normalize_ellipses("．．．．．．うっ"), "……うっ")
        self.assertEqual(normalize_ellipses("なんだ．"), "なんだ．")  # a single dot is left alone

    def test_contained_box_dropped(self):
        big, small, other = box(100, 100, 300, 300), box(150, 150, 200, 200), box(400, 100, 500, 200)
        kept = drop_contained([big, small, other])
        self.assertEqual(kept, [big, other])


class MergeDetectors(unittest.TestCase):
    def test_missed_text_is_added_and_covered_text_is_not(self):
        found = [box(100, 100, 200, 300), box(0, 0, 500, 500, "panel")]
        extra = [box(110, 120, 190, 280),      # inside a box we already have: not new
                 box(300, 400, 360, 600),      # missed text: new
                 box(305, 405, 355, 590),      # overlaps the new one: not new again
                 box(400, 100, 408, 110)]      # tiny: noise
        merged = merge_text_boxes(found, extra)
        added = [g for g in merged if g.get("source") == "comic_text_detector"]
        self.assertEqual([g["bbox"] for g in added], [[300, 400, 360, 600]])
        self.assertEqual(added[0]["type"], "text")
        self.assertEqual(merged[:2], found)  # the first detector's boxes are kept as they were

    def test_nothing_found_by_either(self):
        self.assertEqual(merge_text_boxes([], []), [])


class ReadingOrder(unittest.TestCase):
    def test_right_to_left_then_top_to_bottom(self):
        a, b, c = box(300, 0, 380, 80), box(100, 0, 180, 80), box(300, 200, 380, 280)
        order = reading_order([b, c, a])
        self.assertEqual(order, [a, b, c])

    def test_ids_by_panel(self):
        panel = box(0, 0, 500, 500, "panel")
        t1, t2 = box(300, 50, 380, 130), box(50, 50, 130, 130)
        out = assign_ids([t1, t2, panel])
        labels = {id(g): g["label"] for g in out}
        self.assertEqual((labels[id(panel)], labels[id(t1)], labels[id(t2)]), ("P1", "1.1", "1.2"))
        outside = assign_ids([box(600, 600, 650, 650)])[0]
        self.assertEqual(outside["label"], "T.1")


class TextFitting(unittest.TestCase):
    def setUp(self):
        if not Path(DEFAULT_FONT["path"]).exists():
            self.skipTest("default font not installed")
        self.font = font_of(DEFAULT_FONT, 20)

    def test_wrap_respects_width(self):
        lines = wrap("one two three four five six seven", self.font, 120)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(self.font.getlength(l) <= 120 for l in lines))
        self.assertIsNone(wrap("Supercalifragilisticexpialidocious", self.font, 40))  # one word wider than the space

    def test_fit_shrinks_to_the_space(self):
        small = fit("What is this supposed to mean?", DEFAULT_FONT, 90, 70)
        large = fit("What is this supposed to mean?", DEFAULT_FONT, 400, 300)
        self.assertTrue(small[3] and large[3])
        self.assertLess(small[0].size, large[0].size)
        impossible = fit("A long line that cannot possibly fit", DEFAULT_FONT, 20, 12)
        self.assertFalse(impossible[3])  # reports that it did not fit instead of pretending


class StalledModel(unittest.TestCase):
    """A whole-page translation that stalls must be retried in small pieces, not lose the page."""

    def run_stage(self, fake_chat):
        import json
        texts = [{"label": f"1.{i}"} for i in range(1, 10)]
        read = {"regions": [{"id": t["label"], "source_text": "あ"} for t in texts]}
        emo = {"scene": "x", "regions": []}
        original, stages.chat = stages.chat, fake_chat
        try:
            return stages.stage_translate("m", texts, read, emo), json
        finally:
            stages.chat = original

    @staticmethod
    def reply(prompt):
        import json, re
        m = re.search(r"ONLY these ids: (.+?)\. The other lines", prompt)
        ids = m.group(1).split(", ") if m else [f"1.{i}" for i in range(1, 10)]
        return json.dumps({"regions": [{"id": i, "translation": f"t{i}", "flag": None} for i in ids]})

    def test_full_page_timeout_falls_back_to_chunks(self):
        calls = []
        def fake(model, prompt, schema, **kw):
            calls.append(prompt)
            if len(calls) == 1:
                raise OllamaTimeout("stalled")
            return self.reply(prompt)
        out, _ = self.run_stage(fake)
        self.assertTrue(out["ids_ok"])
        self.assertEqual(len(out["regions"]), 9)
        self.assertGreater(len(calls), 2)  # one stalled call, then several small ones

    def test_one_bad_line_does_not_take_the_others_down(self):
        """The model loops on one particular line: every request containing it fails, the rest still get translated."""
        def fake(model, prompt, schema, **kw):
            import re
            m = re.search(r"ONLY these ids: (.+?)\. The other lines", prompt)
            want = m.group(1).split(", ") if m else [f"1.{i}" for i in range(1, 10)]
            if "1.5" in want and len(want) > 1:
                raise OllamaTimeout("looping")
            if want == ["1.5"]:
                raise OllamaTimeout("looping")
            return self.reply(prompt)
        out, _ = self.run_stage(fake)
        self.assertFalse(out["ids_ok"])
        self.assertEqual(sorted(r["id"] for r in out["regions"]), [f"1.{i}" for i in range(1, 10) if i != 5])

    def test_everything_stalls_reports_failure_without_crashing(self):
        def fake(model, prompt, schema, **kw):
            raise OllamaTimeout("stalled")
        out, _ = self.run_stage(fake)
        self.assertFalse(out["ids_ok"])
        self.assertIsNone(out["regions"])


class WeightMeasure(unittest.TestCase):
    def test_thicker_strokes_measure_heavier(self):
        def bars(thickness):
            ink = np.zeros((120, 400), bool)
            for k in range(6):
                x = 20 + k * 60
                ink[20:100, x:x + thickness] = True
            return ink
        self.assertGreater(weight_ratio(bars(12)), weight_ratio(bars(4)))
        self.assertIsNone(weight_ratio(np.zeros((50, 50), bool)))  # no ink, no measurement


if __name__ == "__main__":
    unittest.main()
