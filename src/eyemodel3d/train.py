"""
Train an image -> eye-model-parameter network supervised by semantic masks.

    image --CNN--> bounded parameters + laterality --differentiable renderer--> soft masks
                                                                                    |
                                                         loss vs annotated mask <---+

Data: periorbital crops showing BOTH eyes (celeb_final_data, cfd_final_data, or
combined_final_data = both):
    DATA/<...>images/<name>.(jpg|png|...)
    DATA/<...>masks/<name>.png      single-channel label image, same size as the image
Each image is split at its vertical midline into two samples: the image-left half is a
right eye (OD, caruncle on the image right, eye_model `medial = +1`), the image-right half
a left eye (OS, `medial = -1`).

Laterality is predicted by the network: both lateralities are rendered and the mask loss
is their mixture weighted by the predicted probability, plus a small cross-entropy on the
known side (--w-lat).

Close-ups: with probability --p-zoom, a training sample is a random zoom into the
full-resolution half image, up to an iris --zoom-max times wider than in the standard
framing (the eye can then be wider than the image), limited so that the source is never
upsampled more than --max-upsample times. A zoom of an image is exactly a change of focal
length (plus a principal-point shift), so f is predicted (log scale) and D is fixed.

--anchor-range: where the iris centre lands in a zoomed window (fractions of its size);
values beyond [0, 1] put the iris partly out of the frame, as in slit-lamp close-ups.
--p-fov: limited field of view. The image is blacked out outside a random rectangle or
disc around the iris (slit-lamp aperture, close-ups) while the mask target is KEPT
everywhere: the network learns to complete the eye beyond what it sees.

Zoom consistency (--unlabeled ROOT, interop layout ROOT/<dataset>/<image>): unlabeled
photos give pairs of views of the same image: A in the standard framing (iris placed with
the rough per-dataset positions of eyemodel3d.interop), B a random zoom/shift of it
(iris anywhere in --anchor-range, strong augmentation, limited field of view). Both views
come from the source pixels, so B is sharp. A zoom/shift is exactly a change of camera:
    pB = z * pA + b   =>   f_B = z f_A,   cx_B = z cx_A + bx  (cx = W/2 + tx W, same in y)
The network in eval mode (teacher, no gradient) predicts A; its parameters, with the
camera moved to B, are rendered as a pseudo mask for B, and the network (student) is
trained on B with the mask loss (and the teacher's laterality). Only samples with a
confident teacher laterality are used (--cons-lat-conf). This teaches close-ups and
off-centre framings on the photos of the target domains without annotations.

Classes compared by the loss (TARGET_CLASSES): lid/skin (= everything that is not the
eye: skin, brows, lid folds), sclera, iris (the model's iris + pupil are summed: the
annotation has no pupil), caruncle. Edit LABEL_MAP for another label convention.

Pupil (--pupil): with masks where detected pupils are drawn as label 6 (python -m
eyemodel3d.pupil --write-masks), an extra loss compares the rendered pupil with them inside
the annotated iris, and r_pupil, dx, dy are predicted. Eyes without a detected pupil only
get the iris loss.

Usage (from the repository root, with the package installed or PYTHONPATH=src):
    python -m eyemodel3d.train --data DATA --out runs/exp1
    python -m eyemodel3d.train --data DATA --eval runs/exp1/best.pt --refine 100 [--zoom-eval]

The evaluation writes to <checkpoint dir>/analysis[_zoom]/:
    per_image.csv      Dice per class, laterality, failure category, predicted parameters
    worst_XX.png       worst samples: input | annotation | prediction | refined | errors
    params_hist.png    distribution of each predicted parameter against its bounds
    summary.json
Failure categories (with --refine N): starting from the network's prediction, the
parameters are optimized directly against the annotation for N steps.
    ok       network prediction already good (mean foreground Dice >= --ok-thresh)
    network  refinement fixes it: the eye model CAN represent this image, the network missed
    model    refinement does not fix it: the model cannot represent it (or a bad local minimum)
"""

import argparse
import csv
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import eyemodel3d.renderer.eye_model as em

# ----------------------------------------------------------------------------- EDIT ME
TARGET_CLASSES = ["lid/skin", "sclera", "iris", "caruncle"]
# Model class (em.CLASSES order: lid/skin, sclera, iris, pupil, caruncle) -> target class.
# Probabilities of model classes sharing a target are summed (pupil counts as iris).
MODEL_TO_TARGET = [0, 1, 2, 2, 3]
# Label value in the mask files -> target class. Values not listed are ignored by the loss.
#   periorbital_dataset: 0 skin, 1 brow, 2 sclera, 3 iris (incl. pupil), 4 caruncle,
#                        5 upper-lid fold, 6 pupil (detected by eyemodel3d.pupil)
LABEL_MAP = {0: 0, 1: 0, 5: 0, 2: 1, 3: 2, 4: 3, 6: 2}
IRIS = 2  # target class used to anchor zooms
IGNORE = 255
NT = len(TARGET_CLASSES)
# --pupil: pupil pixels (label 6) get this extra target value. The class losses count them
# as iris; the pupil loss compares them with the rendered pupil. Eyes without detected pupil
# (label 3 only) get no pupil loss.
PUPIL_LABEL, PUPIL_T = 6, NT
PUPIL_PARAMS = ["r_pupil", "dx", "dy"]  # predicted with --pupil
EYE_FRAC = 0.42  # typical fissure width / half-image width in the data (sets f's default)

# Parameters NOT predicted by default (kept fixed):
#   D              degenerate with f; f is predicted instead because a zoom is a change of f
#   roll           nearly degenerate with canthal tilt (h_left vs h_right): the limbus is a circle
#   k_up, k_low    lid/gaze coupling, not identifiable from single images
#   r_pupil, dx, dy  no pupil in the annotation
DEFAULT_FIXED = ["D", "roll", "k_up", "k_low", "r_pupil", "dx", "dy"]
LOG_PARAMS = ["f"]  # predicted on a log scale

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def default_focal(W, eye_frac=EYE_FRAC):
    """Focal length (px) for which the default eye spans `eye_frac` of the render width."""
    phi = np.radians(em.DEFAULTS["phi_right"])
    half_mm = em.R_LID * np.sin(phi)
    dist = em.DEFAULTS["D"] - em.R_LID * np.cos(phi)
    return float(eye_frac * W / 2 * dist / half_mm)


def default_bounds(W, zoom_max):
    """Training bounds that differ from eye_model.PARAM_SPECS: name -> (lo, hi, default)."""
    f0 = default_focal(W)
    return {
        "f": (0.6 * f0, 1.5 * zoom_max * f0, f0),
        # in a zoom the eyeball centre can be far from the image centre
        "tx": (-1.2, 1.2, 0.0),
        "ty": (-1.2, 1.2, 0.0),
        # The visible eye cannot be wider than the eyeball (2 R_EYE = 24 mm), so the
        # fissure / iris width ratio is capped near R_EYE / r_limbus: 2.3 at 4.5 mm while
        # wide-open eyes in the data reach 2.6-2.7. With f free, a smaller iris is the same
        # as a bigger eyeball. Floor 3.0 allows up to ~3.4; default where training settled.
        "r_limbus": (3.0, 7.0, 4.5),
        # lids retracted or pulled away (e.g. lower lid pulled down to show the conjunctiva)
        "open_up": (-5.0, 75.0, em.PARAM_SPECS["open_up"][3]),
        "open_low": (-5.0, 75.0, em.PARAM_SPECS["open_low"][3]),
    }


# ----------------------------------------------------------------------------- data
def find_subdir(root, key):
    root = Path(root)
    if (root / key).is_dir():
        return root / key
    hits = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(key)]
    if len(hits) != 1:
        raise SystemExit(f"Need exactly one folder named '{key}' or '*{key}' in {root}")
    return hits[0]


def list_pairs(root, limit=None):
    img_dir, mask_dir = find_subdir(root, "images"), find_subdir(root, "masks")
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMG_EXT:
            m = mask_dir / (img.stem + ".png")
            if m.exists():
                pairs.append((img, m))
    if not pairs:
        raise SystemExit(f"No image/mask pairs found in {img_dir} and {mask_dir}")
    return pairs[:limit] if limit else pairs


def _crop_resize(img, mask, box, W, H):
    """Crop box (x0, y0, x1, y1) in source pixels, possibly outside the image, and resize
    to W x H. Outside the image: black in img, IGNORE in mask."""
    box = tuple(int(round(b)) for b in box)
    img = img.crop(box).resize((W, H), Image.BILINEAR)
    # PIL pads crops with 0: shift labels by one so that padding maps back to IGNORE
    m = Image.fromarray(((np.asarray(mask, np.int64) + 1) % 256).astype(np.uint8))
    m = np.asarray(m.crop(box).resize((W, H), Image.NEAREST), np.int64)
    return img, Image.fromarray(((m - 1) % 256).astype(np.uint8))


def strong_aug(x, img):
    """Photometric and degradation augmentations of a (3, H, W) image in [0, 1], aimed at
    the other capture setups (slit lamp, phone, low quality). The mask is unchanged:
    occlusions and glare must be seen through, as the shape prior allows."""
    _, H, W = x.shape
    r = random.random
    if r() < 0.3:  # JPEG artefacts
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(15, 70))
        x = TF.to_tensor(Image.open(buf).convert("RGB"))
    if r() < 0.2:  # low resolution
        f = random.uniform(0.25, 0.6)
        small = F.interpolate(x[None], scale_factor=f, mode="bilinear", antialias=True)
        x = F.interpolate(small, size=(H, W), mode="bilinear")[0]
    if r() < 0.3:  # defocus
        s = random.uniform(0.3, 1.5)
        x = TF.gaussian_blur(x, 2 * math.ceil(2 * s) + 1, s)
    if r() < 0.5:  # colour cast (illumination, white balance)
        x = x * torch.empty(3, 1, 1).uniform_(0.7, 1.3)
    if r() < 0.3:  # gamma / exposure
        x = x.clamp(0, 1) ** random.uniform(0.6, 1.6)
    if r() < 0.1:
        x = x.mean(0, keepdim=True).expand(3, H, W)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    if r() < 0.3:  # vignetting (slit-lamp / close-up lighting)
        cx, cy = random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        x = x * (1 - random.uniform(0.3, 0.9) * (d2 / d2.max()) ** random.uniform(1, 2))
    if r() < 0.4:  # specular highlights
        for _ in range(random.randint(1, 3)):
            cx, cy = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
            ax, ay = random.uniform(0.01, 0.08), random.uniform(0.01, 0.08)
            d2 = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
            x = x + random.uniform(0.5, 1.0) * torch.exp(-d2 * 2)
    if r() < 0.3:  # occluding blobs (lashes, fingers, instruments): cutout
        for _ in range(random.randint(1, 2)):
            h, w = round(random.uniform(0.1, 0.3) * H), round(random.uniform(0.1, 0.3) * W)
            y0, x0 = random.randint(0, H - h), random.randint(0, W - w)
            x = x.clone()
            x[:, y0 : y0 + h, x0 : x0 + w] = torch.rand(3, 1, 1)
    if r() < 0.3:  # sensor noise
        x = x + random.uniform(0.01, 0.06) * torch.randn_like(x)
    return x.clamp(0, 1)


def fov_aug(x, centre):
    """Limited field of view: black outside a random rectangle or disc (slit-lamp aperture)
    that contains `centre` = (x, y) in pixels (the iris centre). x: (3, H, W) in [0, 1]."""
    _, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) + 0.5,
        torch.arange(W, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    cx, cy = centre
    if random.random() < 0.4:  # disc around the iris
        r = random.uniform(0.3, 0.6) * max(H, W)
        ox, oy = cx + random.uniform(-0.4, 0.4) * r, cy + random.uniform(-0.4, 0.4) * r
        keep = (xx - ox) ** 2 + (yy - oy) ** 2 <= r * r
    else:  # rectangle with the iris centre somewhere inside
        w, h = random.uniform(0.35, 0.9) * W, random.uniform(0.35, 0.9) * H
        x0, y0 = cx - random.uniform(0.1, 0.9) * w, cy - random.uniform(0.1, 0.9) * h
        keep = (xx >= x0) & (xx < x0 + w) & (yy >= y0) & (yy < y0 + h)
    return x * keep


def label_centre(y):
    """Iris centre (x, y) in pixels of a target label map, or the image centre."""
    ys, xs = torch.nonzero((y == IRIS) | (y == PUPIL_T), as_tuple=True)
    if len(xs) < 5:
        return y.shape[1] / 2, y.shape[0] / 2
    return xs.float().mean().item() + 0.5, ys.float().mean().item() + 0.5


class EyeMaskDataset(torch.utils.data.Dataset):
    """Two samples per image: the image-left half (OD, medial +1) and the image-right half
    (OS, medial -1), each framed like a letterbox into H x W (padding is IGNORE).

    augment: flips (swap laterality), random zooms, small rotations/shifts, colour jitter.
    aug="strong" adds photometric/degradation augmentations (see strong_aug).
    zoom_eval: deterministic random zoom for every sample (to evaluate close-ups)."""

    def __init__(
        self,
        pairs,
        H,
        W,
        augment=False,
        p_zoom=0.5,
        zoom_max=6.0,
        max_upsample=4.0,
        zoom_eval=False,
        seed=0,
        aug="basic",
        pupil=False,
        anchor=(0.15, 0.85),
        p_fov=0.0,
    ):
        self.pairs, self.H, self.W = pairs, H, W
        self.augment, self.zoom_eval, self.seed = augment, zoom_eval, seed
        self.strong = aug == "strong"
        self.anchor, self.p_fov = tuple(anchor), p_fov
        self.p_zoom, self.zoom_max, self.max_upsample = p_zoom, zoom_max, max_upsample
        self.ramp = 1.0  # training curriculum: scales p_zoom and the zoom range (0..1)
        self.jitter = torchvision.transforms.ColorJitter(0.3, 0.3, 0.3, 0.03)
        self.lut = np.full(256, IGNORE, np.uint8)
        for k, v in LABEL_MAP.items():
            self.lut[k] = v
        if pupil:
            self.lut[PUPIL_LABEL] = PUPIL_T

    def __len__(self):
        return 2 * len(self.pairs)

    def load_half(self, i):
        """Full-resolution half image, its target-label mask, and medial (+1 OD, -1 OS)."""
        ip, mp = self.pairs[i // 2]
        side = i % 2
        img = Image.open(ip).convert("RGB")
        raw = np.array(Image.open(mp))
        if raw.ndim == 3:
            raw = raw[..., 0]
        mask = Image.fromarray(self.lut[raw.astype(np.int64).clip(0, 255)])
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)
        w, h = img.size
        hw = w // 2
        box = (0, 0, hw, h) if side == 0 else (w - hw, 0, w, h)
        name = f"{ip.stem}_{'OD' if side == 0 else 'OS'}"
        return img.crop(box), mask.crop(box), 1.0 if side == 0 else -1.0, name

    def frame(self, img, mask, rng, zoom):
        """Window into the half image: the letterbox frame, or a random zoom into it."""
        w, h = img.size
        s = min(self.W / w, self.H / h)  # letterbox scale (output px per source px)
        fw, fh = self.W / s, self.H / s  # letterbox frame in source px
        if not zoom:
            x0, y0 = (w - fw) / 2, (h - fh) / 2
            return _crop_resize(img, mask, (x0, y0, x0 + fw, y0 + fh), self.W, self.H)

        zmax = min(1 + (self.zoom_max - 1) * self.ramp, self.max_upsample / s)
        z = math.exp(rng.uniform(0, math.log(zmax))) if zmax > 1 else 1.0
        ww, wh = fw / z, fh / z
        m = np.asarray(mask)
        ys, xs = np.nonzero((m == IRIS) | (m == PUPIL_T))
        if len(xs) < 5:
            ys, xs = np.nonzero((m != 0) & (m != IGNORE))
        ax, ay = (xs.mean(), ys.mean()) if len(xs) else (w / 2, h / 2)
        # the anchor (iris centre) lands at a random place in the window (anchor range;
        # beyond [0, 1] the iris is partly out of the frame)
        x0 = ax - rng.uniform(*self.anchor) * ww
        y0 = ay - rng.uniform(*self.anchor) * wh
        return _crop_resize(img, mask, (x0, y0, x0 + ww, y0 + wh), self.W, self.H)

    def __getitem__(self, i):
        img, mask, medial, name = self.load_half(i)
        if self.zoom_eval:
            rng = random.Random(self.seed * 100003 + i)
            img, mask = self.frame(img, mask, rng, zoom=True)
        else:
            zoom = self.augment and random.random() < self.p_zoom * self.ramp
            img, mask = self.frame(img, mask, random, zoom)

        if self.augment:
            if random.random() < 0.5:  # a mirrored right eye looks like a left eye
                img, mask, medial = TF.hflip(img), TF.hflip(mask), -medial
            angle = random.uniform(-15, 15)
            ds, dt = (0.2, 0.1) if self.strong else (0.1, 0.05)
            scale = random.uniform(1 - ds, 1 + ds)
            shift = [
                round(random.uniform(-dt, dt) * self.W),
                round(random.uniform(-dt, dt) * self.H),
            ]
            img = TF.affine(
                img,
                angle,
                shift,
                scale,
                [0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle,
                shift,
                scale,
                [0.0, 0.0],
                interpolation=TF.InterpolationMode.NEAREST,
                fill=IGNORE,
            )
            img = self.jitter(img)

        x = TF.to_tensor(img)
        if self.augment and self.strong:
            x = strong_aug(x, img)
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        if self.augment and random.random() < self.p_fov:  # target kept everywhere
            x = fov_aug(x, label_centre(y))
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, y, torch.tensor(medial), name


class ConsistencyPairs(torch.utils.data.Dataset):
    """Unlabeled photos (interop records) -> (xA, xB, rel): view A in the standard framing
    (small scale/shift jitter, mild colour jitter), view B a random zoom of the source with
    the iris centre at a random place (anchor range), strong augmentation and limited field
    of view. rel = (z, bx, by): pixel coordinates satisfy pB = z * pA + (bx, by)."""

    def __init__(self, records, H, W, zoom_max=3.0, max_upsample=4.0, anchor=(0.15, 0.85),
                 p_fov=0.0, aug="strong"):
        self.records, self.H, self.W = records, H, W
        self.zoom_max, self.max_upsample = zoom_max, max_upsample
        self.anchor, self.p_fov, self.strong = tuple(anchor), p_fov, aug == "strong"
        self.jitter = torchvision.transforms.ColorJitter(0.3, 0.3, 0.3, 0.03)
        self.weak = torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.0)

    def __len__(self):
        return len(self.records)

    def warp(self, img, s, b):
        """Output pixel p = s * source pixel + b (per axis), W x H, black outside."""
        W0, H0 = img.size
        r = min(1.0, s)  # downscale with antialiasing first, then upsample/shift exactly
        if r < 1.0:
            img = img.resize((max(1, round(W0 * r)), max(1, round(H0 * r))), Image.LANCZOS,
                             reducing_gap=3.0)
        rx, ry = img.size[0] / W0, img.size[1] / H0
        # PIL: input = (a x + b y + c, d x + e y + f) for output (x, y)
        coef = (rx / s, 0, -b[0] * rx / s, 0, ry / s, -b[1] * ry / s)
        return img.transform((self.W, self.H), Image.AFFINE, coef, Image.BILINEAR,
                             fillcolor=(0, 0, 0))

    def __getitem__(self, i):
        rec = self.records[i]
        from .interop.core import TARGET_CENTER, TARGET_IRIS_W

        img = Image.open(rec.path).convert("RGB")
        W0, H0 = img.size
        cx, cy, iw = rec.iris_cx * W0, rec.iris_cy * H0, rec.iris_w * W0
        if random.random() < 0.5:
            img, cx = img.transpose(Image.FLIP_LEFT_RIGHT), W0 - cx
        H, W = self.H, self.W
        u = random.uniform
        sA = TARGET_IRIS_W * W / iw * math.exp(u(-0.15, 0.15))
        bA = ((TARGET_CENTER[0] + u(-0.05, 0.05)) * W - sA * cx,
              (TARGET_CENTER[1] + u(-0.05, 0.05)) * H - sA * cy)
        zmax = min(self.zoom_max, max(1.0, self.max_upsample / sA))
        z = math.exp(u(0, math.log(zmax)))
        sB = z * sA
        bB = (u(*self.anchor) * W - sB * cx, u(*self.anchor) * H - sB * cy)

        a = self.weak(self.warp(img, sA, bA))
        b = self.jitter(self.warp(img, sB, bB))
        xA, xB = TF.to_tensor(a), TF.to_tensor(b)
        if self.strong:
            xB = strong_aug(xB, b)
        if random.random() < self.p_fov:
            xB = fov_aug(xB, (sB * cx + bB[0], sB * cy + bB[1]))
        rel = torch.tensor([z, bB[0] - z * bA[0], bB[1] - z * bA[1]], dtype=torch.float32)
        return (xA - IMAGENET_MEAN) / IMAGENET_STD, (xB - IMAGENET_MEAN) / IMAGENET_STD, rel


def transfer_camera(vals, rel, H, W):
    """Parameters predicted on view A -> the same eye seen in view B (pB = z pA + b)."""
    z, bx, by = rel.unbind(1)
    out = dict(vals)
    out["f"] = vals["f"] * z
    out["tx"] = (z * (W / 2 + vals["tx"] * W) + bx - W / 2) / W
    out["ty"] = (z * (H / 2 + vals["ty"] * H) + by - H / 2) / H
    return out


# ----------------------------------------------------------------------------- model
class ParamSpace:
    """Maps unconstrained outputs z to bounded parameter values in natural units.
    bounds: optional {name: (lo, hi, default)} overriding eye_model.PARAM_SPECS.
    log: names mapped on a log scale (lo > 0)."""

    def __init__(self, names, bounds=None, log=()):
        self.names = list(names)
        bounds = bounds or {}
        spec = [bounds.get(n, em.PARAM_SPECS[n][1:]) for n in self.names]
        self.lo = torch.tensor([s[0] for s in spec], dtype=torch.float32)
        self.hi = torch.tensor([s[1] for s in spec], dtype=torch.float32)
        self.default = torch.tensor([s[2] for s in spec], dtype=torch.float32)
        self.is_log = torch.tensor([n in log for n in self.names])

    def to(self, device):
        self.lo, self.hi, self.default, self.is_log = (
            t.to(device) for t in (self.lo, self.hi, self.default, self.is_log)
        )
        return self

    def _t(self, v):  # to the space in which the mapping is linear
        safe = torch.where(self.is_log, v, torch.ones_like(v)).clamp(min=1e-6)
        return torch.where(self.is_log, torch.log(safe), v)

    def normalize(self, v):
        """Position of v between its bounds, in [0, 1]."""
        lo, hi = self._t(self.lo), self._t(self.hi)
        return (self._t(v) - lo) / (hi - lo)

    def values(self, z):
        lo, hi = self._t(self.lo), self._t(self.hi)
        t = lo + (hi - lo) * torch.sigmoid(z)
        # exp only where used: exp of a large linear value would overflow and poison grads
        return torch.where(self.is_log, torch.exp(torch.where(self.is_log, t, 0.0)), t)

    def logits(self, v):
        return torch.logit(self.normalize(v).clamp(1e-4, 1 - 1e-4))

    def as_dict(self, v):
        return {n: v[:, i] for i, n in enumerate(self.names)}


def make_backbone(name, pretrained=True, img_size=None, drop_path=0.0):
    """(module returning (B, F) features, F). "timm:<model>" for a timm model (ViT, Swin,
    ConvNeXt, ...: see timm.list_models), otherwise a torchvision classifier."""
    if name.startswith("timm:"):
        import timm

        kw = dict(pretrained=pretrained, num_classes=0)
        if drop_path:
            kw["drop_path_rate"] = drop_path
        try:  # transformers take the input size (position embeddings, windows)
            net = timm.create_model(name[5:], img_size=tuple(img_size), **kw)
        except TypeError:  # CNNs do not
            net = timm.create_model(name[5:], **kw)
        return net, net.num_features
    ctor = getattr(torchvision.models, name)
    try:
        net = ctor(weights="DEFAULT" if pretrained else None)
    except Exception as exc:  # e.g. offline
        print(f"Could not load pretrained weights ({exc}); using random init.")
        net = ctor(weights=None)
    feat = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat


class ParamNet(nn.Module):
    """backbone -> parameter logits z and laterality logit. net_input (H, W): the image is
    resized to this size before the backbone (e.g. 224 x 288 for a ViT pretrained at 224),
    None to keep the data size."""

    def __init__(
        self, space, backbone="resnet34", pretrained=True, net_input=None, drop_path=0.0
    ):
        super().__init__()
        self.net_input = tuple(net_input) if net_input else None
        self.amp = False  # bf16 autocast of the backbone only (set by train)
        net, feat = make_backbone(backbone, pretrained, self.net_input, drop_path)
        self.backbone = net
        self.head = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, len(space.names)),
        )
        self.lat_head = nn.Linear(feat, 1)  # laterality logit (+ = OD, medial +1)
        # Start every prediction at the average eye: zero last layer, bias = logit(default)
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(space.logits(space.default))

    def forward(self, x):
        if self.net_input and tuple(x.shape[-2:]) != self.net_input:
            x = F.interpolate(x, size=self.net_input, mode="bilinear", align_corners=False)
        with torch.autocast(x.device.type, torch.bfloat16, enabled=self.amp):
            f = self.backbone(x)
        f = f.float()  # heads (and the renderer) in fp32
        return self.head(f), self.lat_head(f).squeeze(-1)


def load_model(path, dev="cpu"):
    """Checkpoint -> (net, space, fixed, meta). Also reads checkpoints from before
    laterality was predicted (they have a fixed `medial`)."""
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    meta = ckpt["meta"]
    space = ParamSpace(meta["names"], meta.get("bounds"), meta.get("log", ()))
    net = ParamNet(space, meta["backbone"], False, meta.get("net_input"))
    missing, _ = net.load_state_dict(ckpt["model"], strict=False)
    if any(not k.startswith("lat_head") for k in missing):
        raise RuntimeError(f"checkpoint is missing weights: {missing}")
    net.to(dev).eval()
    space.to(dev)
    fixed = {k: torch.tensor(v, device=dev) for k, v in meta["fixed"].items()}
    meta = dict(meta, epoch=ckpt.get("epoch"), val_dice=ckpt.get("val_dice"))
    return net, space, fixed, meta


def predicted_medial(lat_logit, fixed):
    """+1 / -1 per sample: the network's laterality, unless the checkpoint fixed it."""
    if "medial" in fixed:
        return fixed["medial"].expand(lat_logit.shape[0])
    return torch.where(lat_logit >= 0, 1.0, -1.0).to(lat_logit)


# ----------------------------------------------------------------------------- render + loss
def render_batch(vals, fixed, medial, H, W, tau, pupil=False):
    """Renders a batch and returns TARGET_CLASSES probabilities (B, H, W, NT)
    (pupil=True: and the rendered pupil probability (B, H, W))."""
    B = medial.shape[0]
    leaves = {k: v.expand(B) for k, v in fixed.items()}
    leaves.update(vals)
    leaves["medial"] = medial
    probs = em.render(*em.split_params(leaves), H=H, W=W, tau=tau)
    M = F.one_hot(torch.tensor(MODEL_TO_TARGET), NT).to(probs)
    if pupil:
        return probs @ M, probs[..., em.CLASSES.index("pupil")]
    return probs @ M


def as_classes(target):
    """Labels with pupil pixels counted as iris (for the class losses and Dice)."""
    return torch.where(target == PUPIL_T, torch.full_like(target, IRIS), target)


def pupil_loss(pp, target):
    """Per-sample (B,) Dice + BCE of the rendered pupil probability pp (B, H, W) inside the
    annotated iris; 0 for eyes without a detected pupil."""
    region = ((target == IRIS) | (target == PUPIL_T)).float()
    t = (target == PUPIL_T).float()
    known = (t.sum((1, 2)) > 0).float()
    p = pp * region
    dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    bce = -(t * pp.clamp(min=1e-6).log() + (1 - t) * (1 - pp).clamp(min=1e-6).log())
    bce = (bce * region).sum((1, 2)) / region.sum((1, 2)).clamp(min=1)
    return known * (dice + bce)


def mask_loss(probs, target, w_ce=1.0, w_dice=1.0, w_mom=1.0):
    """Per-sample loss (B,). probs (B, H, W, C) soft probabilities, target (B, H, W) labels.
    The moment term (area + centroid per class) gives gradients even when a predicted
    region does not overlap its target yet."""
    B, H, W, C = probs.shape
    target = as_classes(target)
    valid = (target != IGNORE).float()
    onehot = F.one_hot(target.clamp(max=C - 1), C).float() * valid[..., None]
    nvalid = valid.sum((1, 2)).clamp(min=1)

    ce = -(onehot * probs.clamp(min=1e-6).log()).sum((1, 2, 3)) / nvalid
    p = probs * valid[..., None]
    inter = (p * onehot).sum((1, 2))
    psum, tsum = p.sum((1, 2)), onehot.sum((1, 2))
    dice = (1 - (2 * inter + 1) / (psum + tsum + 1)).mean(1)
    loss = w_ce * ce + w_dice * dice

    if w_mom > 0:
        ys = torch.linspace(0, 1, H, device=probs.device)[None, :, None, None]
        xs = torch.linspace(0, 1, W, device=probs.device)[None, None, :, None]

        def moments(m):
            a = m.sum((1, 2))
            return (
                a / nvalid[:, None],
                (m * xs).sum((1, 2)) / (a + 1e-3),
                (m * ys).sum((1, 2)) / (a + 1e-3),
            )

        ap, xp, yp = moments(p)
        at, xt, yt = moments(onehot)
        present = (tsum > 0).float()
        mom = ((ap - at).abs() + present * ((xp - xt).abs() + (yp - yt).abs())).mean(1)
        loss = loss + w_mom * mom
    return loss


def mixture_loss(vals, fixed, lat_logit, target, H, W, tau, args):
    """Renders both lateralities; per-sample mask loss weighted by the predicted
    probability of each. Returns the loss (B,) and the probs of the more likely side."""
    B = lat_logit.shape[0]
    ones = torch.ones(B, device=lat_logit.device)
    p_od = torch.sigmoid(lat_logit)
    pupil = getattr(args, "pupil", False)
    pr_od = render_batch(vals, fixed, ones, H, W, tau, pupil)
    pr_os = render_batch(vals, fixed, -ones, H, W, tau, pupil)
    if pupil:
        (pr_od, pp_od), (pr_os, pp_os) = pr_od, pr_os
    l_od = mask_loss(pr_od, target, args.w_ce, args.w_dice, args.w_mom)
    l_os = mask_loss(pr_os, target, args.w_ce, args.w_dice, args.w_mom)
    if pupil:
        l_od = l_od + args.w_pupil * pupil_loss(pp_od, target)
        l_os = l_os + args.w_pupil * pupil_loss(pp_os, target)
    probs = torch.where((p_od >= 0.5)[:, None, None, None], pr_od, pr_os)
    return p_od * l_od + (1 - p_od) * l_os, probs


def consistency_loss(net, space, fixed, xA, xB, rel, H, W, tau, args):
    """Zoom consistency on unlabeled pairs (see the module docstring). Returns the mean
    loss over samples with a confident teacher laterality, and their fraction."""
    was_training = net.training
    net.eval()
    with torch.no_grad():
        zA, latA = net(xA)
        vB = transfer_camera(space.as_dict(space.values(zA)), rel, H, W)
        mA = predicted_medial(latA, fixed)
        probs, pp = render_batch(vB, fixed, mA, H, W, args.tau_min, pupil=True)
        target = probs.argmax(-1)
        if args.pupil:
            target[(pp > 0.5) & (target == IRIS)] = PUPIL_T
        conf = (torch.sigmoid(latA) - 0.5).abs() * 2 >= args.cons_lat_conf
    net.train(was_training)
    zB, latB = net(xB)
    vals = space.as_dict(space.values(zB))
    loss, _ = mixture_loss(vals, fixed, latB, target, H, W, tau, args)
    loss = loss + args.w_lat * F.binary_cross_entropy_with_logits(
        latB, (mA > 0).float(), reduction="none"
    )
    w = conf.float()
    return (loss * w).sum() / w.sum().clamp(min=1), w.mean().item()


def prior_loss(space, v):
    return ((space.normalize(v) - space.normalize(space.default)) ** 2).mean(1)


@torch.no_grad()
def hard_dice(probs, target):
    """(B, C) Dice of the argmax prediction; NaN where a class is absent in both."""
    pred = probs.argmax(-1)
    target = as_classes(target)
    valid = target != IGNORE
    out = torch.full((probs.shape[0], NT), float("nan"), device=probs.device)
    for c in range(NT):
        pc, tc = (pred == c) & valid, (target == c) & valid
        den = (pc.sum((1, 2)) + tc.sum((1, 2))).float()
        d = 2 * (pc & tc).sum((1, 2)).float() / den.clamp(min=1)
        out[:, c] = torch.where(den > 0, d, torch.full_like(d, float("nan")))
    return out


def tau_at(step, total, t_max, t_min, frac=0.6):
    """Geometric annealing of edge softness over the first `frac` of training."""
    r = min(1.0, step / max(1.0, frac * total))
    return t_max * (t_min / t_max) ** r


def build_fixed(names, overrides, dev):
    fixed = {}
    for n in names:
        if n not in em.PARAM_SPECS:
            raise SystemExit(f"Unknown parameter in --fixed: {n}")
        fixed[n] = float(em.PARAM_SPECS[n][3])
    for item in overrides:
        k, v = item.split("=")
        if k not in fixed:
            raise SystemExit(f"--set {k}: only fixed parameters can be set")
        fixed[k] = float(v)
    return {k: torch.tensor(v, device=dev) for k, v in fixed.items()}


# ----------------------------------------------------------------------------- training
@torch.no_grad()
def validate(net, space, fixed, loader, H, W, tau, dev):
    """Mean foreground Dice, per-class Dice, laterality accuracy, pupil Dice (inside the
    annotated iris, eyes with a detected pupil; NaN if there are none)."""
    net.eval()
    dices, correct, pdice = [], [], []
    for x, y, med, _ in loader:
        x, y, med = x.to(dev), y.to(dev), med.to(dev)
        z, lat_logit = net(x)
        m = predicted_medial(lat_logit, fixed)
        probs, pp = render_batch(space.as_dict(space.values(z)), fixed, m, H, W, tau, True)
        dices.append(hard_dice(probs, y).cpu())
        correct.append((m == med).cpu())
        t = y == PUPIL_T
        known = t.any(-1).any(-1)
        if known.any():
            region = (y == IRIS) | t
            pc = (pp > 0.5) & region
            inter = (pc & t).sum((1, 2)).float()
            d = 2 * inter / (pc.sum((1, 2)) + t.sum((1, 2))).clamp(min=1).float()
            pdice.append(d[known].cpu())
    d = torch.cat(dices)
    return (
        torch.nanmean(d[:, 1:]).item(),
        torch.nanmean(d, 0).tolist(),
        torch.cat(correct).float().mean().item(),
        torch.cat(pdice).mean().item() if pdice else float("nan"),
    )


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    H, W = args.size

    # Split by image, so both eyes of a face stay on the same side of the split
    pairs = list_pairs(args.data, args.max_images)
    if args.val_data:
        train_pairs, val_pairs = pairs, list_pairs(args.val_data)
    else:
        idx = list(range(len(pairs)))
        random.Random(args.seed).shuffle(idx)
        nv = max(1, round(args.val_frac * len(pairs)))
        val_pairs = [pairs[i] for i in idx[:nv]]
        train_pairs = [pairs[i] for i in idx[nv:]]
    print(
        f"{len(train_pairs)} training / {len(val_pairs)} validation images "
        f"(x2 eyes), device {dev}"
    )

    fixed_names = [n for n in args.fixed if not (args.pupil and n in PUPIL_PARAMS)]
    fixed = build_fixed(fixed_names, args.set, dev)
    names = [n for n in em.PARAM_SPECS if n not in fixed]
    bounds = {k: v for k, v in default_bounds(W, args.zoom_max).items() if k in names}
    log = [n for n in LOG_PARAMS if n in names]
    space = ParamSpace(names, bounds, log)
    net = ParamNet(
        space, args.backbone, not args.no_pretrained, args.net_input, args.drop_path
    ).to(dev)
    net.amp = args.amp and dev.type == "cuda"
    print(
        f"backbone {args.backbone}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M "
        f"parameters, input {net.net_input or (H, W)}, augmentation {args.aug}"
    )
    space.to(dev)
    if args.init:  # previous run: whole network if it predicts the same parameters
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        sd, same = ck["model"], ck.get("meta", {}).get("names") == names
        if not same:  # heads differ when the parameters differ
            sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        net.load_state_dict(sd, strict=False)
        print(f"{'network' if same else 'backbone'} initialized from {args.init}")

    zkw = dict(
        p_zoom=args.p_zoom,
        zoom_max=args.zoom_max,
        max_upsample=args.max_upsample,
        anchor=tuple(args.anchor_range),
    )
    kw = dict(
        batch_size=args.batch, num_workers=args.workers, pin_memory=dev.type == "cuda"
    )
    ds_tr = EyeMaskDataset(
        train_pairs, H, W, augment=True, aug=args.aug, pupil=args.pupil, p_fov=args.p_fov,
        **zkw
    )
    unl_iter = None
    if args.unlabeled:
        if "f" not in names or "tx" not in names or "ty" not in names:
            raise SystemExit("--unlabeled needs f, tx and ty predicted (not --fixed)")
        from .interop.core import list_images, load_overrides

        ov = load_overrides(args.unlabeled_overrides) if args.unlabeled_overrides else None
        recs = list_images(args.unlabeled, args.unlabeled_datasets, ov)
        ds_u = ConsistencyPairs(
            recs, H, W, args.cons_zoom_max, args.max_upsample, args.anchor_range,
            args.p_fov, args.aug,
        )
        dl_u = torch.utils.data.DataLoader(
            ds_u, batch_size=args.cons_batch or args.batch, shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=dev.type == "cuda",
            persistent_workers=args.workers > 0,
        )

        def cycle():
            while True:
                yield from dl_u

        unl_iter = cycle()
        print(f"zoom consistency on {len(recs)} unlabeled photos from {args.unlabeled}")
    dl_tr = torch.utils.data.DataLoader(
        ds_tr,
        shuffle=True,
        drop_last=True,
        **kw,
    )
    dl_va = torch.utils.data.DataLoader(EyeMaskDataset(val_pairs, H, W, pupil=args.pupil, **zkw), **kw)
    dl_vz = torch.utils.data.DataLoader(
        EyeMaskDataset(val_pairs, H, W, zoom_eval=True, pupil=args.pupil, **zkw), **kw
    )

    opt = torch.optim.AdamW(
        [
            {"params": net.backbone.parameters(), "lr": args.lr},
            {
                "params": list(net.head.parameters()) + list(net.lat_head.parameters()),
                "lr": args.lr * 10,
            },
        ],
        weight_decay=args.wd,
    )
    # linear warmup then cosine decay, per step (transformers need the warmup)
    total_steps = args.epochs * len(dl_tr)
    warm = args.warmup * len(dl_tr)

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(
        version=2,
        names=names,
        bounds={k: list(v) for k, v in bounds.items()},
        log=log,
        fixed={k: float(v) for k, v in fixed.items()},
        size=[H, W],
        backbone=args.backbone,
        net_input=list(args.net_input) if args.net_input else None,
        aug=args.aug,
        pupil=args.pupil,
        data=str(args.data),
        val=[p[0].name for p in val_pairs],
        label_map={str(k): v for k, v in LABEL_MAP.items()},
        target_classes=TARGET_CLASSES,
        tau_min=args.tau_min,
        zoom=zkw,
        p_fov=args.p_fov,
        init=str(args.init) if args.init else None,
        consistency=dict(
            unlabeled=str(args.unlabeled),
            datasets=args.unlabeled_datasets,
            w=args.w_cons,
            zoom_max=args.cons_zoom_max,
            start=args.cons_start,
            lat_conf=args.cons_lat_conf,
        )
        if args.unlabeled
        else None,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    best = -1.0
    for ep in range(args.epochs):
        tau = tau_at(ep, args.epochs, args.tau_max, args.tau_min)
        # zoom curriculum; workers copy the dataset at each epoch, so this is picked up
        ds_tr.ramp = min(1.0, ep / max(1.0, args.zoom_warmup * args.epochs))
        net.train()
        total, count = 0.0, 0
        c_tot, c_conf, c_n = 0.0, 0.0, 0
        # consistency weight: 0 before --cons-start (fraction of training), then ramps up
        # linearly over 5 epochs
        s0 = args.cons_start * args.epochs
        w_cons = args.w_cons * min(1.0, max(0.0, (ep - s0 + 1) / 5)) if unl_iter else 0.0
        bar = tqdm(dl_tr, desc=f"epoch {ep}/{args.epochs}", leave=False)
        for x, y, med, _ in bar:
            x, y, med = x.to(dev), y.to(dev), med.to(dev)
            z, lat_logit = net(x)
            v = space.values(z)
            loss, _ = mixture_loss(space.as_dict(v), fixed, lat_logit, y, H, W, tau, args)
            loss = (
                loss.mean()
                + args.w_prior * prior_loss(space, v).mean()
                + args.w_lat
                * F.binary_cross_entropy_with_logits(lat_logit, (med > 0).float())
            )
            if w_cons > 0:
                xA, xB, rel = (t.to(dev, non_blocking=True) for t in next(unl_iter))
                lc, frac = consistency_loss(net, space, fixed, xA, xB, rel, H, W, tau, args)
                loss = loss + w_cons * lc
                c_tot, c_conf, c_n = c_tot + lc.item(), c_conf + frac, c_n + 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
            bar.set_postfix(loss=f"{total / count:.4f}", tau=f"{tau:.3f}")

        fg, per_class, lat_acc, pup = validate(
            net, space, fixed, dl_va, H, W, args.tau_min, dev
        )
        fg_z, _, lat_acc_z, _ = validate(net, space, fixed, dl_vz, H, W, args.tau_min, dev)
        if args.pupil:
            per_class = per_class + [pup]
        names_pc = TARGET_CLASSES + ["pupil"]
        pc = "  ".join(f"{c}={d:.3f}" for c, d in zip(names_pc, per_class))
        print(
            f"epoch {ep:3d}  tau {tau:.3f}  zoom ramp {ds_tr.ramp:.2f}  train loss {total / max(count, 1):.4f}  "
            f"val Dice fg {fg:.4f} [{pc}]  zoomed {fg_z:.4f}  "
            f"laterality acc {lat_acc:.3f} / zoomed {lat_acc_z:.3f}"
            + (
                f"  consistency {c_tot / c_n:.4f} (w {w_cons:.2f}, "
                f"confident {c_conf / c_n:.2f})"
                if c_n
                else ""
            ),
            flush=True,
        )
        score = (fg + fg_z) / 2
        ckpt = dict(
            model=net.state_dict(),
            meta=meta,
            epoch=ep,
            val_dice=fg,
            val_dice_zoom=fg_z,
            lat_acc=lat_acc,
        )
        torch.save(ckpt, out / "last.pt")
        if score > best:
            best = score
            torch.save(ckpt, out / "best.pt")
    print(f"best val Dice (mean of standard and zoomed) {best:.4f}; checkpoints in {out}")


# ----------------------------------------------------------------------------- analysis
def refine(space, z0, fixed, medial, y, H, W, args):
    """Optimize parameters directly against the annotation, starting from the prediction."""
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    t0 = max(0.15, args.tau_min)
    with torch.enable_grad():
        for it in range(args.refine):
            tau = t0 * (args.tau_min / t0) ** (it / max(1, args.refine - 1))
            probs = render_batch(
                space.as_dict(space.values(z)), fixed, medial, H, W, tau
            )
            loss = mask_loss(probs, y, args.w_ce, args.w_dice, args.w_mom).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return z.detach(), render_batch(
            space.as_dict(space.values(z)), fixed, medial, H, W, args.tau_min
        )


def colorize(lbl):
    # target colour = colour of the first model class mapped to it
    first = [MODEL_TO_TARGET.index(t) for t in range(NT)]
    pal = np.vstack([em.COLORS[first].numpy(), [[0.5, 0.5, 0.5]]])  # last: ignore
    idx = np.where(lbl == IGNORE, len(pal) - 1, lbl)
    return pal[idx]


def analyze(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, space, fixed, meta = load_model(args.eval, dev)
    H, W = meta["size"]

    if args.val_data:
        pairs = list_pairs(args.val_data)
    else:
        keep = set(meta["val"])
        pairs = [p for p in list_pairs(args.data) if p[0].name in keep]
    zkw = meta.get("zoom", {})
    loader = torch.utils.data.DataLoader(
        EyeMaskDataset(pairs, H, W, zoom_eval=args.zoom_eval, **zkw),
        batch_size=args.batch,
        num_workers=args.workers,
    )
    out = Path(args.eval).parent / ("analysis_zoom" if args.zoom_eval else "analysis")
    out.mkdir(exist_ok=True)

    rows, worst, all_vals = [], [], []
    for x, y, med, names in tqdm(loader, desc="analysis"):
        x, y, med = x.to(dev), y.to(dev), med.to(dev)
        with torch.no_grad():
            z, lat_logit = net(x)
            m_used = predicted_medial(lat_logit, fixed)
            probs = render_batch(
                space.as_dict(space.values(z)), fixed, m_used, H, W, args.tau_min
            )
        d_pred = hard_dice(probs, y)
        if args.refine > 0:
            # refine with the true laterality: diagnoses the shape, not the side
            z_ref, probs_ref = refine(space, z, fixed, med, y, H, W, args)
            d_ref = hard_dice(probs_ref, y)
            vr = space.values(z_ref)
        v = space.values(z)
        all_vals.append(v.cpu())
        u = torch.sigmoid(z)

        for i in range(x.shape[0]):
            mean_pred = float(np.nanmean(d_pred[i, 1:].cpu().numpy()))
            row = dict(
                name=names[i],
                laterality="OD" if med[i] > 0 else "OS",
                laterality_pred="OD" if m_used[i] > 0 else "OS",
                p_od=round(float(torch.sigmoid(lat_logit[i])), 4),
                mean_fg_pred=round(mean_pred, 4),
            )
            for c, cname in enumerate(TARGET_CLASSES):
                row[f"dice_pred_{cname}"] = round(float(d_pred[i, c]), 4)
            if args.refine > 0:
                mean_ref = float(np.nanmean(d_ref[i, 1:].cpu().numpy()))
                row["mean_fg_refined"] = round(mean_ref, 4)
                for c, cname in enumerate(TARGET_CLASSES):
                    row[f"dice_refined_{cname}"] = round(float(d_ref[i, c]), 4)
                row["category"] = (
                    "ok"
                    if mean_pred >= args.ok_thresh
                    else "network"
                    if mean_ref >= args.ok_thresh
                    else "model"
                )
            else:
                row["category"] = "ok" if mean_pred >= args.ok_thresh else "fail"
            row["at_bound"] = " ".join(
                n for j, n in enumerate(space.names) if u[i, j] < 0.02 or u[i, j] > 0.98
            )
            for j, n in enumerate(space.names):
                row[f"p_{n}"] = round(float(v[i, j]), 4)
            if args.refine > 0:
                for j, n in enumerate(space.names):
                    row[f"r_{n}"] = round(float(vr[i, j]), 4)
            rows.append(row)

            worst.append(
                dict(
                    score=mean_pred,
                    name=names[i],
                    row=row,
                    img=(x[i].cpu() * IMAGENET_STD + IMAGENET_MEAN)
                    .clamp(0, 1)
                    .permute(1, 2, 0)
                    .numpy(),
                    gt=y[i].cpu().numpy().astype(np.uint8),
                    pred=probs[i].argmax(-1).cpu().numpy().astype(np.uint8),
                    ref=probs_ref[i].argmax(-1).cpu().numpy().astype(np.uint8)
                    if args.refine > 0
                    else None,
                )
            )
        worst = sorted(worst, key=lambda e: e["score"])[: args.worst]

    # per-image CSV
    with open(out / "per_image.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # worst-case panels, 8 samples per figure
    cols = (
        ["image", "annotation", "prediction"]
        + (["refined"] if args.refine > 0 else [])
        + ["errors"]
    )
    for k in range(0, len(worst), 8):
        chunk = worst[k : k + 8]
        fig, axs = plt.subplots(
            len(chunk),
            len(cols),
            figsize=(3.2 * len(cols), 2.6 * len(chunk)),
            squeeze=False,
        )
        for r, e in enumerate(chunk):
            err = ((e["pred"] != e["gt"]) & (e["gt"] != IGNORE)).astype(float)
            panels = [e["img"], colorize(e["gt"]), colorize(e["pred"])]
            if args.refine > 0:
                panels.append(colorize(e["ref"]))
            panels.append(err)
            for c, (a, p) in enumerate(zip(axs[r], panels)):
                a.imshow(p, cmap="magma" if cols[c] == "errors" else None)
                a.set_axis_off()
                if r == 0:
                    a.set_title(cols[c])
            info = (
                f"{e['name']}  pred {e['row']['laterality_pred']}\n"
                f"Dice {e['row']['mean_fg_pred']:.2f}"
            )
            if args.refine > 0:
                info += f" / ref {e['row']['mean_fg_refined']:.2f}  [{e['row']['category']}]"
            axs[r][0].text(0, -4, info, fontsize=7, va="bottom")
        plt.tight_layout()
        plt.savefig(out / f"worst_{k // 8 + 1:02d}.png", dpi=90)
        plt.close(fig)

    # parameter distributions vs bounds (spikes at a bound = saturated / degenerate parameter)
    V = torch.cat(all_vals).numpy()
    n = len(space.names)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.3 * nrow), squeeze=False)
    for j, name in enumerate(space.names):
        a = axs[j // ncol][j % ncol]
        lo, hi, d = float(space.lo[j]), float(space.hi[j]), float(space.default[j])
        if space.is_log[j]:
            a.hist(V[:, j], bins=np.geomspace(lo, hi, 31), color="#4a78b5")
            a.set_xscale("log")
        else:
            a.hist(V[:, j], bins=30, range=(lo, hi), color="#4a78b5")
        a.axvline(d, color="k", lw=0.8, ls="--")
        a.set_title(name, fontsize=9)
        a.tick_params(labelsize=7)
    for j in range(n, nrow * ncol):
        axs[j // ncol][j % ncol].set_axis_off()
    plt.tight_layout()
    plt.savefig(out / "params_hist.png", dpi=90)
    plt.close(fig)

    # summary
    cats, bound_counts = {}, {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        for nm in r["at_bound"].split():
            bound_counts[nm] = bound_counts.get(nm, 0) + 1
    summary = dict(
        n_samples=len(rows),
        zoomed=args.zoom_eval,
        mean_fg_pred=float(np.nanmean([r["mean_fg_pred"] for r in rows])),
        per_class_pred={
            c: float(np.nanmean([r[f"dice_pred_{c}"] for r in rows]))
            for c in TARGET_CLASSES
        },
        laterality_acc=float(
            np.mean([r["laterality"] == r["laterality_pred"] for r in rows])
        ),
        categories=cats,
        params_at_bound=dict(sorted(bound_counts.items(), key=lambda kv: -kv[1])),
    )
    if args.refine > 0:
        summary["mean_fg_refined"] = float(
            np.nanmean([r["mean_fg_refined"] for r in rows])
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"analysis written to {out}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data", required=True, help="folder with *images/ and *masks/ subfolders"
    )
    ap.add_argument(
        "--val-data",
        help="separate validation folder (default: random split of --data)",
    )
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument(
        "--max-images", type=int, help="use only the first N images (quick tests)"
    )
    ap.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[128, 160],
        metavar=("H", "W"),
        help="render / network input size of one eye (half image)",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--lr", type=float, default=1e-4, help="backbone lr (heads use 10x)"
    )
    ap.add_argument(
        "--backbone",
        default="resnet34",
        help="torchvision name, or timm:<name>, e.g. "
        "timm:swin_tiny_patch4_window7_224.ms_in1k, "
        "timm:vit_small_patch16_224.augreg_in21k_ft_in1k",
    )
    ap.add_argument(
        "--net-input",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="resize the image to this size inside the network (e.g. 224 288 for "
        "ViT/Swin pretrained at 224; default: --size)",
    )
    ap.add_argument("--drop-path", type=float, default=0.0, help="timm stochastic depth")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay (ViT: 0.05)")
    ap.add_argument("--warmup", type=int, default=0, help="lr warmup epochs")
    ap.add_argument(
        "--no-amp", dest="amp", action="store_false", help="disable bf16 backbone"
    )
    ap.add_argument(
        "--aug",
        choices=["basic", "strong"],
        default="basic",
        help="strong: + JPEG, blur, low-res, colour cast, gamma, vignetting, glare, "
        "cutout, noise",
    )
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument(
        "--fixed",
        nargs="*",
        default=DEFAULT_FIXED,
        help="parameters kept fixed instead of predicted",
    )
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="NAME=VALUE",
        help="value of a fixed parameter (default: eye_model default)",
    )
    ap.add_argument(
        "--p-zoom", type=float, default=0.5, help="fraction of zoomed training samples"
    )
    ap.add_argument(
        "--zoom-max",
        type=float,
        default=6.0,
        help="max zoom vs the standard framing (6: iris ~1.2x the image width)",
    )
    ap.add_argument(
        "--zoom-warmup",
        type=float,
        default=0.3,
        help="fraction of training over which zooms ramp in (none at epoch 0)",
    )
    ap.add_argument(
        "--init", help="checkpoint whose backbone initializes the network"
    )
    ap.add_argument(
        "--max-upsample",
        type=float,
        default=4.0,
        help="never enlarge source pixels more than this (limits zoom on small images)",
    )
    ap.add_argument(
        "--pupil",
        action="store_true",
        help="supervise the pupil with the detected pupils (mask label 6, see "
        "eyemodel3d.pupil) and predict r_pupil, dx, dy",
    )
    ap.add_argument("--w-pupil", type=float, default=1.0, help="pupil loss weight")
    ap.add_argument(
        "--anchor-range",
        type=float,
        nargs=2,
        default=[0.15, 0.85],
        metavar=("LO", "HI"),
        help="where the iris centre lands in zoomed windows (fraction of the window); "
        "e.g. -0.1 1.1 to also get irises cut by the frame",
    )
    ap.add_argument(
        "--p-fov",
        type=float,
        default=0.0,
        help="probability of a limited field of view (black outside a rectangle/disc "
        "around the iris; target kept)",
    )
    ap.add_argument(
        "--unlabeled",
        help="interop-style folder (ROOT/<dataset>/<images>) for the zoom-consistency loss",
    )
    ap.add_argument("--unlabeled-datasets", nargs="*", help="subfolders to use (default: all)")
    ap.add_argument(
        "--unlabeled-overrides", help="interop overrides CSV (iris position/width per photo)"
    )
    ap.add_argument("--w-cons", type=float, default=0.5, help="zoom-consistency weight")
    ap.add_argument(
        "--cons-zoom-max", type=float, default=3.0, help="max zoom of view B vs view A"
    )
    ap.add_argument(
        "--cons-start",
        type=float,
        default=0.2,
        help="fraction of training before the consistency loss starts (0 with --init)",
    )
    ap.add_argument(
        "--cons-lat-conf",
        type=float,
        default=0.6,
        help="use pairs whose teacher laterality |2 p(OD) - 1| is at least this",
    )
    ap.add_argument("--cons-batch", type=int, help="unlabeled batch size (default: --batch)")
    ap.add_argument("--w-ce", type=float, default=1.0)
    ap.add_argument("--w-dice", type=float, default=1.0)
    ap.add_argument(
        "--w-mom", type=float, default=1.0, help="area/centroid moment loss"
    )
    ap.add_argument(
        "--w-prior", type=float, default=0.01, help="pull toward default parameters"
    )
    ap.add_argument(
        "--w-lat",
        type=float,
        default=0.1,
        help="cross-entropy on the known laterality (0: learned from the masks only)",
    )
    ap.add_argument(
        "--tau-max", type=float, default=0.5, help="initial edge softness (mm)"
    )
    ap.add_argument(
        "--tau-min", type=float, default=0.05, help="final edge softness (mm)"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--eval", help="checkpoint to analyze instead of training")
    ap.add_argument(
        "--zoom-eval", action="store_true", help="analyze random zooms of the val set"
    )
    ap.add_argument(
        "--refine",
        type=int,
        default=0,
        help="test-time optimization steps against the annotation (failure diagnosis)",
    )
    ap.add_argument(
        "--worst", type=int, default=32, help="number of worst samples to plot"
    )
    ap.add_argument(
        "--ok-thresh",
        type=float,
        default=0.85,
        help="mean foreground Dice counted as ok",
    )
    args = ap.parse_args()
    analyze(args) if args.eval else train(args)


if __name__ == "__main__":
    main()
