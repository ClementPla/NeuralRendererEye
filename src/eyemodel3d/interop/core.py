"""
Read the interop eye photos and frame them for the network (H x W, e.g. 128 x 160).

Two framings (`mode` of prepare):

  "letterbox" (default)  the whole photo, scaled to fit and padded with black. For models
                         trained with zoom augmentation (train.py --p-zoom), which handle
                         close-ups where the iris is up to ~1.2x the image width.
  "iris"                 the standard training framing: scaled so that the iris width is
                         TARGET_IRIS_W * W and placed so that its centre lands at
                         TARGET_CENTER. Needs the iris centre and width in each photo, from
                         rough per-dataset defaults (DATASETS) corrected per image with an
                         overrides CSV (write_template / load_overrides). For models trained
                         without zooms.

mirror_os=True mirrors left eyes so they look like right eyes: only needed for older
checkpoints that do not predict laterality (they have a fixed `medial`). The laterality
parsed from file names is otherwise only a reference to check the network's prediction.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path("/home/clement/Documents/data/EyeSegmentation/interop")
DEFAULT_SIZE = (128, 160)  # (H, W) of train.py; use the checkpoint's meta["size"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

# Iris in the training inputs (measured on celeb_final_data, median over ~1000 eyes):
TARGET_IRIS_W = 0.20  # iris width / image width (5-95%: 0.17-0.23)
TARGET_CENTER = (0.48, 0.67)  # iris centre (x, y) as fractions of the image


def _lat_token(pattern, flip=False):
    """Laterality from a regex whose first group is od/os/r/l (case-insensitive).
    The result is the laterality the eye LOOKS like (OD: caruncle on the image right);
    flip=True for datasets whose images are mirrored relative to their labels."""
    rx = re.compile(pattern, re.IGNORECASE)

    def parse(name):
        m = rx.search(name)
        if not m:
            return "unknown"
        od = m.group(1).lower() in ("od", "r")
        return "OD" if od != flip else "OS"

    return parse


def _unknown(name):
    return "unknown"


@dataclass
class DatasetInfo:
    """Rough defaults for one source folder. iris_w and center are fractions of the
    source image width (and height for center y)."""

    iris_w: float
    center: tuple
    laterality: callable = _unknown
    note: str = ""


# Visual estimates from a few images per folder: correct outliers with an overrides CSV.
DATASETS = {
    "eom": DatasetInfo(0.30, (0.50, 0.55), _lat_token(r"_(od|os)_"), "slit-lamp disc"),
    "eyepacs": DatasetInfo(0.27, (0.50, 0.50), note="external eye photo, laterality unknown"),
    "eyes_defy_anemia": DatasetInfo(
        0.60, (0.50, 0.25), note="lower lid pulled down, iris at the top; laterality unknown"
    ),
    "iol": DatasetInfo(0.40, (0.55, 0.50), _lat_token(r"_(od|os)_"), "slit-lamp disc"),
    "mobius": DatasetInfo(0.23, (0.45, 0.50), _lat_token(r"_([LR])[slru]_")),
    "poly": DatasetInfo(0.40, (0.50, 0.50), _lat_token(r"_(od|os)_"), "slit-lamp disc"),
    # sbvpi images are mirrored relative to their R/L names: "R" photos have the caruncle
    # on the image LEFT (checked on 34R_s_7, 38R_s_4, 18L_s_3, 43L_u_2)
    "sbvpi": DatasetInfo(
        0.25, (0.50, 0.50), _lat_token(r"^\d+([LR])_", flip=True), "mirrored vs names"
    ),
    "slid": DatasetInfo(0.50, (0.50, 0.50), note="iris close-up, laterality unknown"),
}
GENERIC = DatasetInfo(0.30, (0.50, 0.50))


@dataclass
class Record:
    path: Path
    dataset: str
    laterality: str  # "OD", "OS" or "unknown" (unknown is not mirrored)
    iris_cx: float  # iris centre / source width
    iris_cy: float  # iris centre / source height
    iris_w: float  # iris width / source width

    @property
    def key(self):
        return f"{self.dataset}/{self.path.name}"


@dataclass
class Prepared:
    """A reframed image plus what is needed to map results back to the source photo."""

    image: Image.Image  # RGB, W x H
    valid: np.ndarray  # (H, W) bool, False on padding
    scale: float  # output px per source px
    offset: tuple  # (ox, oy): output = scale * (mirrored source) + offset
    mirrored: bool
    source_size: tuple  # (W0, H0)
    record: Record = field(repr=False, default=None)


# ----------------------------------------------------------------------------- reading
def list_images(root=DEFAULT_ROOT, datasets=None, overrides=None):
    """Records for every image under root/<dataset>/, with defaults then overrides applied.
    datasets: optional list of folder names to keep. overrides: dict from load_overrides."""
    root = Path(root)
    overrides = overrides or {}
    recs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if datasets and d.name not in datasets:
            continue
        info = DATASETS.get(d.name, GENERIC)
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            r = Record(
                f, d.name, info.laterality(f.name), info.center[0], info.center[1], info.iris_w
            )
            for k, v in overrides.get(r.key, {}).items():
                setattr(r, k, v)
            recs.append(r)
    return recs


def load_overrides(path):
    """CSV with columns dataset,name and any of iris_cx,iris_cy,iris_w,laterality.
    Empty cells keep the default. Returns {"dataset/name": {field: value}}."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            o = {}
            for k in ("iris_cx", "iris_cy", "iris_w"):
                if row.get(k, "").strip():
                    o[k] = float(row[k])
            lat = row.get("laterality", "").strip().upper()
            if lat:
                o["laterality"] = {"R": "OD", "L": "OS"}.get(lat, lat)
            out[f"{row['dataset']}/{row['name']}"] = o
    return out


def write_template(records, path):
    """Write the current values of every record as an editable overrides CSV."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "name", "laterality", "iris_cx", "iris_cy", "iris_w"])
        for r in records:
            w.writerow([r.dataset, r.path.name, r.laterality, r.iris_cx, r.iris_cy, r.iris_w])


# ----------------------------------------------------------------------------- reframing
def prepare(
    record,
    size=DEFAULT_SIZE,
    mode="letterbox",
    mirror_os=False,
    target_iris_w=TARGET_IRIS_W,
    target_center=TARGET_CENTER,
):
    """Load a record's photo and frame it to `size` = (H, W); see the module docstring."""
    H, W = size
    img = Image.open(record.path).convert("RGB")
    W0, H0 = img.size
    cx, cy = record.iris_cx, record.iris_cy
    mirrored = mirror_os and record.laterality == "OS"
    if mirrored:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        cx = 1 - cx

    if mode == "letterbox":
        s = min(W / W0, H / H0)
        ox, oy = (W - s * W0) / 2, (H - s * H0) / 2
    elif mode == "iris":
        s = target_iris_w * W / (record.iris_w * W0)
        ox = target_center[0] * W - s * cx * W0
        oy = target_center[1] * H - s * cy * H0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    # Resize first (antialiased), then paste at the integer offset
    nw, nh = max(1, round(W0 * s)), max(1, round(H0 * s))
    small = img.resize((nw, nh), Image.LANCZOS, reducing_gap=3.0)
    px, py = round(ox), round(oy)
    canvas = Image.new("RGB", (W, H))
    canvas.paste(small, (px, py))
    valid = np.zeros((H, W), bool)
    valid[max(0, py) : max(0, py + nh), max(0, px) : max(0, px + nw)] = True
    return Prepared(canvas, valid, nw / W0, (px, py), mirrored, (W0, H0), record)


def to_source(label_map, prep, fill=0):
    """Map an (H, W) label map from the prepared frame back onto the source photo
    (W0 x H0), e.g. to overlay a prediction on the original image."""
    W0, H0 = prep.source_size
    s, (ox, oy) = prep.scale, prep.offset
    lbl = Image.fromarray(np.asarray(label_map, np.uint8))
    # output (source) pixel x -> prepared pixel s * x + ox
    back = lbl.transform(
        (W0, H0), Image.AFFINE, (s, 0, ox, 0, s, oy), Image.NEAREST, fillcolor=fill
    )
    if prep.mirrored:
        back = back.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(back)


# ----------------------------------------------------------------------------- torch
def to_tensor(prep):
    """Network input (3, H, W), normalized as in train.py."""
    import torch
    import torchvision.transforms.functional as TF

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (TF.to_tensor(prep.image) - mean) / std


def make_dataset(records, size=DEFAULT_SIZE, mode="letterbox", mirror_os=False):
    """torch Dataset yielding (x, valid, key); x is normalized as in train.py."""
    import torch

    class InteropDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, i):
            p = prepare(records[i], size, mode, mirror_os)
            return to_tensor(p), torch.from_numpy(p.valid), records[i].key

    return InteropDataset()


# ----------------------------------------------------------------------------- checking
def preview_sheet(records, out, size=DEFAULT_SIZE, mode="letterbox", mirror_os=False, ncol=5):
    """Contact sheet of prepared images. In the "iris" framing, the dashed circle is where
    the iris should be; if an image's iris is not on it, fix it in the overrides CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = size
    n = len(records)
    nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), squeeze=False)
    for a in axs.flat:
        a.set_axis_off()
    for a, r in zip(axs.flat, records):
        a.imshow(prepare(r, size, mode, mirror_os).image)
        if mode == "iris":
            a.add_patch(
                plt.Circle(
                    (TARGET_CENTER[0] * W, TARGET_CENTER[1] * H),
                    TARGET_IRIS_W * W / 2,
                    fill=False,
                    ls="--",
                    lw=1,
                    color="y",
                )
            )
        a.set_title(f"{r.key}\n{r.laterality}", fontsize=6)
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close(fig)
