"""
Real-time eye model fitting on a webcam (or video) stream.

Each frame:
  1. eyes: OpenCV's YuNet face detector gives both eye centres (mode "face"); or a fixed box
     in the middle of the frame, to hold one eye close to the camera (mode "center");
  2. each eye is cropped like the training inputs (iris ~TARGET_IRIS_W of the width, centred
     at TARGET_CENTER, eye line horizontal), predicted in one batch, and rendered;
  3. the rendered outlines are drawn back on the frame, with the predicted laterality; a
     panel on the right shows each network input and its prediction.

Predictions are smoothed over time (exponential moving average of the network outputs and
of the eye positions; `smooth` = weight of the past).
"""

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

import eyemodel3d.renderer.eye_model as em

from . import train as T
from .interop.core import TARGET_CENTER, TARGET_IRIS_W

ROOT = Path(__file__).resolve().parents[2]
YUNET = ROOT / "models/face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
IRIS_PER_IPD = 0.186  # iris diameter / interpupillary distance (~11.7 / 63 mm)
PUPIL = T.NT  # label value of the pupil in the label maps
# outline colours (BGR) per label: sclera, iris, caruncle, pupil
OUTLINE = {1: (255, 255, 255), 2: (255, 255, 0), 3: (255, 0, 255), PUPIL: (0, 255, 255)}


def ensure_yunet(path=YUNET):
    path = Path(path)
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


class EyeTracker:
    def __init__(self, ckpt, dev=None, mode="face", smooth=0.5, center_frac=0.6,
                 min_face_score=0.8):
        self.dev = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net, self.space, self.fixed, self.meta = T.load_model(ckpt, self.dev)
        self.net.amp = self.dev.type == "cuda"
        self.H, self.W = self.meta["size"]
        self.predicts_pupil = "r_pupil" not in self.fixed
        self.mode, self.smooth, self.center_frac = mode, smooth, center_frac
        self.detector = cv2.FaceDetectorYN.create(str(ensure_yunet()), "", (320, 320),
                                                  min_face_score)
        self.det_size = None
        self._eyes = None  # smoothed eye centres (2, 2)
        self._z = None  # smoothed network outputs
        self.mean = T.IMAGENET_MEAN.to(self.dev)
        self.std = T.IMAGENET_STD.to(self.dev)

    # ------------------------------------------------------------------ framing
    def eye_centres(self, frame):
        """(2, 2) eye centres (image left first), smoothed, or None if no face."""
        h, w = frame.shape[:2]
        if self.det_size != (w, h):
            self.detector.setInputSize((w, h))
            self.det_size = (w, h)
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            self._eyes = None
            return None
        f = max(faces, key=lambda f: f[2] * f[3])  # largest face
        eyes = np.array([f[4:6], f[6:8]], np.float32)
        eyes = eyes[np.argsort(eyes[:, 0])]
        if self._eyes is not None:
            eyes = self.smooth * self._eyes + (1 - self.smooth) * eyes
        self._eyes = eyes
        return eyes

    def transforms(self, frame):
        """Affine maps frame -> network input (2 x 3), one per eye."""
        H, W = self.H, self.W
        if self.mode == "center":
            h, w = frame.shape[:2]
            ch = self.center_frac * h
            s = H / ch
            cx, cy = w / 2, h / 2
            return [np.array([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]], np.float32)]
        eyes = self.eye_centres(frame)
        if eyes is None:
            return []
        d = eyes[1] - eyes[0]
        ipd = float(np.hypot(*d))
        angle = math.degrees(math.atan2(d[1], d[0]))  # eye line tilt (image y down)
        s = TARGET_IRIS_W * W / (IRIS_PER_IPD * ipd)
        target = np.array([TARGET_CENTER[0] * W, TARGET_CENTER[1] * H], np.float32)
        Ms = []
        for e in eyes:
            M = cv2.getRotationMatrix2D((float(e[0]), float(e[1])), angle, s)
            M[:, 2] += target - e
            Ms.append(M.astype(np.float32))
        return Ms

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, frame):
        """frame: BGR uint8. Returns a list of per-eye dicts: M, crop (RGB), lbl, lat, p_od,
        vals (natural units)."""
        Ms = self.transforms(frame)
        if not Ms:
            self._z = None
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crops = [cv2.warpAffine(rgb, M, (self.W, self.H), flags=cv2.INTER_LINEAR) for M in Ms]
        x = torch.from_numpy(np.stack(crops)).to(self.dev).permute(0, 3, 1, 2).float() / 255
        x = (x - self.mean) / self.std
        z, lat = self.net(x)
        out = torch.cat([z, lat[:, None]], 1)
        if self._z is not None and self._z.shape == out.shape:
            out = self.smooth * self._z + (1 - self.smooth) * out
        self._z = out
        z, lat = out[:, :-1], out[:, -1]
        v = self.space.values(z)
        m = T.predicted_medial(lat, self.fixed)
        probs, pp = T.render_batch(self.space.as_dict(v), self.fixed, m, self.H, self.W,
                                   self.meta["tau_min"], pupil=True)
        lbl = probs.argmax(-1).cpu().numpy().astype(np.uint8)
        if self.predicts_pupil:
            lbl[((pp > 0.5).cpu().numpy()) & (lbl == T.IRIS)] = PUPIL
        p_od = torch.sigmoid(lat).cpu().numpy()
        vals = v.cpu().numpy()
        return [
            dict(M=M, crop=c, lbl=l, lat="OD" if mi > 0 else "OS", p_od=float(p),
                 vals=dict(zip(self.space.names, vv)))
            for M, c, l, mi, p, vv in zip(Ms, crops, lbl, m.tolist(), p_od, vals)
        ]

    # ------------------------------------------------------------------ drawing
    def draw(self, frame, results, fps=None, thumbs=True):
        out = frame.copy()
        panel = np.full((frame.shape[0], 2 * self.W, 3), 40, np.uint8)
        for k, r in enumerate(results):
            Minv = cv2.invertAffineTransform(r["M"])
            for c, col in OUTLINE.items():
                mask = (r["lbl"] == c).astype(np.uint8)
                if not mask.any():
                    continue
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = [cv2.transform(cn.astype(np.float32), Minv).astype(np.int32)
                        for cn in cnts]
                cv2.polylines(out, cnts, True, col, 1, cv2.LINE_AA)
            # laterality label above the eye
            ex, ey = cv2.transform(
                np.array([[[TARGET_CENTER[0] * self.W, 0.15 * self.H]]], np.float32), Minv
            )[0, 0]
            cv2.putText(out, f"{r['lat']} {r['p_od']:.2f}", (int(ex) - 30, int(ey)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if self.mode == "center":
                box = cv2.transform(np.array([[[0, 0], [self.W, 0], [self.W, self.H],
                                               [0, self.H]]], np.float32), Minv)
                cv2.polylines(out, [box.astype(np.int32)], True, (0, 200, 0), 1)
            if thumbs:  # network input | prediction, in the side panel
                pred = (T.colorize(np.where(r["lbl"] == PUPIL, T.IRIS, r["lbl"])) * 255)
                pred = pred.astype(np.uint8)
                pred[r["lbl"] == PUPIL] = (em.COLORS[em.CLASSES.index("pupil")].numpy() * 255)
                tile = np.concatenate([r["crop"], pred], 1)[..., ::-1]
                y0 = k * (self.H + 24)
                if y0 + self.H + 20 <= panel.shape[0]:
                    panel[y0:y0 + self.H] = tile
                    v = r["vals"]
                    txt = f"{r['lat']}  P(OD) {r['p_od']:.2f}"
                    if "r_pupil" in v and "r_limbus" in v:
                        txt += f"  pupil/iris {v['r_pupil'] / v['r_limbus']:.2f}"
                    cv2.putText(panel, txt, (4, y0 + self.H + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (220, 220, 220), 1, cv2.LINE_AA)
        if fps is not None:
            cv2.putText(out, f"{fps:.1f} fps", (out.shape[1] - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if not results:
            cv2.putText(out, "no face" if self.mode == "face" else "", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return np.concatenate([out, panel], 1) if thumbs else out


class LiveLoop:
    """Read frames from `source` (camera index or video file), process them in a background
    thread, and call `show(jpeg_bytes, results)` for every processed frame and
    `on_end(error_or_None)` when the loop stops."""

    def __init__(self, tracker, source=0, show=None, width=640, mirror=False, loop_file=True,
                 on_end=None):
        self.tracker, self.source, self.show, self.on_end = tracker, source, show, on_end
        self.width, self.mirror, self.loop_file = width, mirror, loop_file
        self.running = False
        self.thread = None
        self.fps = 0.0
        self.error = None
        self.last = None  # (frame, results)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video source {self.source!r}")
            t_prev = time.perf_counter()
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    if self.loop_file and not isinstance(self.source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                if self.width and frame.shape[1] != self.width:
                    s = self.width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                results = self.tracker.predict(frame)
                t = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(t - t_prev, 1e-6)
                t_prev = t
                img = self.tracker.draw(frame, results, self.fps)
                self.last = (frame, results)
                if self.show is not None:
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self.show(buf.tobytes(), results)
        except Exception as exc:  # shown by the notebook
            self.error = exc
        finally:
            cap.release()
            self.running = False
            if self.on_end is not None:
                self.on_end(self.error)
