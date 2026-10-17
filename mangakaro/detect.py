"""Detectors and reading order.

panels_text          panels and text regions (YOLO26n-seg): finds the text boxes and the panels used for reading order
bubbles_yolov8       speech bubbles: where each bubble is, so English can be placed inside it
comic_text_detector  text boxes plus a pixel mask of the lettering itself, so only the glyphs are erased
"""
import cv2
import numpy as np

from .paths import WEIGHTS as W

COLORS = {"panel": (255, 120, 0), "text": (0, 0, 255), "bubble": (0, 160, 0), "text_jp": (0, 0, 255), "text_en": (200, 0, 200)}


# ---------------------------------------------------------------- detectors
class PanelsText:
    """Oliverdsfdsf/comic-panels-text-detect: YOLO26n-seg, classes Panel/Text."""
    name = "panels_text"

    def __init__(self):
        from safetensors.torch import load_file
        from ultralytics import YOLO
        from ultralytics.nn.tasks import SegmentationModel

        m = YOLO("yolo26n-seg.yaml")
        net = SegmentationModel("yolo26n-seg.yaml", nc=2, verbose=False)
        net.load_state_dict(load_file(str(W / "comic-panels-and-text-detect.safetensors")))
        net.names = {0: "Text", 1: "Panel"}  # model card lists Panel first, but verified on page 13: class 0 = bubbles/text, 1 = panels
        m.model = net
        self.m = m

    def __call__(self, img):
        r = self.m.predict(img, imgsz=1280, conf=0.25, device="cpu", verbose=False)[0]
        out = []
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            out.append({"type": "text" if int(c) == 0 else "panel", "bbox": [round(v) for v in b], "conf": round(s, 3)})
        return out


class BubblesYolo:
    """PSImera/manga_bubbles_detect: YOLOv8 bubble-only (.pt is a pickle; trusted repo only)."""
    name = "bubbles_yolov8"

    def __init__(self):
        from ultralytics import YOLO
        self.m = YOLO(str(W / "bubbles_detect.pt"))

    def __call__(self, img):
        r = self.m.predict(img, imgsz=1280, conf=0.25, device="cpu", verbose=False)[0]
        return [{"type": "bubble", "bbox": [round(v) for v in b], "conf": round(s, 3)}
                for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


class ComicTextDetector:
    """dmMaze/comic-text-detector via ONNX (weights from manga-image-translator beta-0.3)."""
    name = "comic_text_detector"
    SIZE = 1024

    def __init__(self, conf=0.4, iou=0.35):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(W / "comictextdetector.pt.onnx"), providers=["CPUExecutionProvider"])
        self.conf, self.iou = conf, iou

    def _run(self, img):
        """Letterbox the page to SIZE x SIZE and run the model. Returns (outputs, scale, resized height, resized width)."""
        h, w = img.shape[:2]
        r = self.SIZE / max(h, w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        return self.s.run(None, {self.s.get_inputs()[0].name: x}), r, nh, nw

    def mask(self, img, thresh=0.3):
        """Pixel mask (bool, page size) of the lettering itself: the model's `seg` output, cut back out of the letterbox."""
        h, w = img.shape[:2]
        outs, _, nh, nw = self._run(img)
        seg = next(o for o in outs if o.ndim == 4 and o.shape[1] == 1)[0, 0]
        return cv2.resize(seg[:nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR) > thresh

    def __call__(self, img):
        h, w = img.shape[:2]
        outs, r, nh, nw = self._run(img)
        blk = next(o for o in outs if o.ndim == 3)[0]  # [N, 5+nc]: cx,cy,w,h,obj,cls...
        cls_scores = blk[:, 5:]
        score = blk[:, 4] * cls_scores.max(1)
        keep = score > self.conf
        blk, score = blk[keep], score[keep]
        cls = cls_scores[keep].argmax(1)
        boxes = np.stack([blk[:, 0] - blk[:, 2] / 2, blk[:, 1] - blk[:, 3] / 2, blk[:, 2], blk[:, 3]], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), self.conf, self.iou)
        out = []
        for i in np.array(idx).flatten():
            x0, y0, bw, bh = boxes[i] / r
            out.append({"type": "text_jp" if cls[i] == 0 else "text_en",
                        "bbox": [max(0, round(x0)), max(0, round(y0)), min(w, round(x0 + bw)), min(h, round(y0 + bh))],
                        "conf": round(float(score[i]), 3)})
        return out


# ---------------------------------------------------------------- driver
def reading_order(items, rtl=True):
    """Sort boxes into rows (top to bottom), each row right-to-left if rtl (manga) else left-to-right."""
    rows = []
    for g in sorted(items, key=lambda g: g["bbox"][1]):
        cy = (g["bbox"][1] + g["bbox"][3]) / 2
        for r in rows:
            if r["y0"] <= cy <= r["y1"]:  # centre falls inside the row's vertical span
                r["items"].append(g)
                r["y1"] = max(r["y1"], g["bbox"][3])
                break
        else:
            rows.append({"y0": g["bbox"][1], "y1": g["bbox"][3], "items": [g]})
    sign = -1 if rtl else 1
    return [g for r in rows for g in sorted(r["items"], key=lambda g: sign * g["bbox"][0])]


def assign_ids(regions, rtl=True):
    """Give regions reading-order IDs: P1.. for panels, then Pn.m for text inside panel n, T.m for text outside any panel.
    Returns regions sorted by that order."""
    panels = reading_order([g for g in regions if g["type"] == "panel"], rtl)
    texts = [g for g in regions if g["type"] != "panel"]
    for n, p in enumerate(panels, 1):
        p["label"], p["panel"] = f"P{n}", n
    members = {n: [] for n in range(1, len(panels) + 1)}
    members[0] = []
    for t in texts:
        cx, cy = (t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2
        inside = [p for p in panels if p["bbox"][0] <= cx <= p["bbox"][2] and p["bbox"][1] <= cy <= p["bbox"][3]]
        # several panels can contain the centre (nested/overlapping boxes): take the smallest
        n = min(inside, key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]))["panel"] if inside else 0
        members[n].append(t)
    out = []
    for n in range(0, len(panels) + 1):
        if n:
            out.append(panels[n - 1])
        for m, t in enumerate(reading_order(members[n], rtl), 1):
            t["panel"] = n or None
            t["label"] = f"{n}.{m}" if n else f"T.{m}"
            out.append(t)
    return out


def draw(img, regions):
    o = img.copy()
    for g in regions:
        x0, y0, x1, y1 = g["bbox"]
        c = COLORS.get(g["type"], (0, 0, 0))
        cv2.rectangle(o, (x0, y0), (x1, y1), c, 3)
        # big ID on a filled tag. Text tags go just outside the box so they never cover the words;
        # panel tags sit inside the corner.
        label = g["label"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
        tag_h = th + 16
        ty = y0 if g["type"] == "panel" or y0 - tag_h < 0 else y0 - tag_h
        cv2.rectangle(o, (x0, ty), (x0 + tw + 14, ty + tag_h), c, -1)
        cv2.putText(o, label, (x0 + 7, ty + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
    return o
