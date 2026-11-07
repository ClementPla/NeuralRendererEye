"""
Automatic pupil detection inside the annotated iris (pseudo-labels for training).

For each eye (each half of a periorbital image):
  1. iris circle: least-squares circle through the iris/sclera boundary of the mask (the
     iris/lid boundary is ignored: lids cut the circle), with a fallback on the iris'
     horizontal extent;
  2. the iris neighbourhood is resampled so that the iris radius is SCALE px;
  3. candidate pupil circles: cv2.HoughCircles on the (blurred) red channel, plus a dense
     grid of centres near the iris centre, each with every radius in [R_MIN, R_MAX] x iris
     radius;
  4. each candidate is scored by the contrast of its edge (as Daugman's operator): mean
     intensity of a thin band just outside minus a band just inside, over iris pixels
     only, relative to the outside band (Weber contrast), with highlights (corneal
     reflections) left out. The best candidate is the pupil; `score` is its contrast
     (low = doubtful: dark iris, blur, closed eye).

    python -m eyemodel3d.pupil --data DATA --out pupils.csv [--sheet sheet.png]
    python -m eyemodel3d.pupil --data DATA --out pupils.csv --write-masks NEW_DATA

--write-masks copies the dataset with the pupil drawn in the masks (label PUPIL_LABEL,
inside the iris only) for the eyes whose detection passes `accept`; in the other eyes the
iris keeps label 3 (= iris with an unknown pupil).
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IRIS_LABEL, SCLERA_LABEL = 3, 2  # periorbital_dataset labels
PUPIL_LABEL = 6  # new label written by --write-masks
SCALE = 40.0  # iris radius in the working resolution (px)
R_MIN, R_MAX = 0.15, 0.55  # pupil radius / iris radius
MAX_OFFSET = 0.35  # pupil centre distance from the iris centre / iris radius


def fit_circle(x, y):
    """Algebraic (Kasa) least-squares circle: (cx, cy, r)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    (a, c, d), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a / 2, c / 2
    return cx, cy, np.sqrt(max(d + cx**2 + cy**2, 0.0))


def iris_circle(mask):
    """(cx, cy, R) of the iris in mask pixels, or None if there is too little iris."""
    iris = mask == IRIS_LABEL
    if iris.sum() < 20:
        return None
    ys, xs = np.nonzero(iris)
    half_w = (xs.max() - xs.min() + 1) / 2
    fallback = ((xs.max() + xs.min()) / 2, ys.mean(), half_w)
    sclera = cv2.dilate((mask == SCLERA_LABEL).astype(np.uint8), np.ones((3, 3), np.uint8))
    edge = iris & ~cv2.erode(iris.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    by, bx = np.nonzero(edge & sclera.astype(bool))
    if len(bx) >= 8:
        cx, cy, r = fit_circle(bx.astype(float), by.astype(float))
        # accept if consistent with the visible iris
        if 0.8 * half_w <= r <= 1.6 * half_w and abs(cx - fallback[0]) < 0.5 * half_w:
            return cx, cy, r
    return fallback


def _ring_means(d, v, radii, inner=(0.6, 0.95), outer=(1.05, 1.35)):
    """For distances d (sorted) and values v (same order), mean of v in the inner disk and
    in the outer ring of every radius. Returns (inside, ring, n_inside)."""
    cs = np.concatenate([[0.0], np.cumsum(v)])

    def mean(lo, hi):
        i0, i1 = np.searchsorted(d, lo), np.searchsorted(d, hi)
        n = i1 - i0
        return (cs[i1] - cs[i0]) / np.maximum(n, 1), n

    ins, n_in = mean(inner[0] * radii, inner[1] * radii)
    ring, n_out = mean(outer[0] * radii, outer[1] * radii)
    _, n_disk = mean(0.0, 0.95 * radii)
    return ins, ring, n_disk, n_out


def detect(img, mask, box=None):
    """Pupil in one eye. img: RGB array, mask: label array (same size), box: (x0, x1)
    columns of the eye (default: all). Returns a dict in image pixels, or None."""
    if box is not None:
        x0, x1 = box
        img, mask = img[:, x0:x1], mask[:, x0:x1]
    else:
        x0 = 0
    ic = iris_circle(mask)
    if ic is None:
        return None
    icx, icy, R = ic

    # working crop: iris radius -> SCALE px
    s = SCALE / R
    half = 1.3 * R
    M = np.array([[s, 0, -(icx - half) * s], [0, s, -(icy - half) * s]], np.float32)
    size = int(round(2 * half * s))
    red = cv2.warpAffine(img[..., 0].astype(np.float32), M, (size, size), flags=cv2.INTER_CUBIC)
    irm = cv2.warpAffine(
        (mask == IRIS_LABEL).astype(np.uint8), M, (size, size), flags=cv2.INTER_NEAREST
    ).astype(bool)
    c0 = half * s  # iris centre in the crop
    # keep iris pixels away from the limbus (its dark edge would look like a pupil ring)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    irm &= np.hypot(xx - c0, yy - c0) < 0.95 * SCALE
    if irm.sum() < 50:  # mostly hidden
        return None
    # leave out highlights (corneal reflections): bright outliers of the iris
    hi = np.median(red[irm]) + 3 * np.median(np.abs(red[irm] - np.median(red[irm]))) + 20
    use = irm & (red < hi)
    px, py, pv = xx[use], yy[use], red[use]
    n_iris = irm.sum()

    # candidates: Hough circles + dense grid of centres
    radii = np.arange(R_MIN * SCALE, R_MAX * SCALE + 1e-6, 1.0)
    blur = cv2.GaussianBlur(np.clip(red, 0, 255).astype(np.uint8), (0, 0), 1.5)
    hough = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=60, param2=8,
        minRadius=int(radii[0]), maxRadius=int(radii[-1]),
    )
    off = MAX_OFFSET * SCALE
    g = np.arange(-off, off + 1e-6, 1.5)
    centres = [(c0 + gx, c0 + gy) for gx in g for gy in g if gx * gx + gy * gy <= off * off]
    n_hough = 0
    if hough is not None:
        for hx, hy, _ in hough[0]:
            if np.hypot(hx - c0, hy - c0) <= off:
                centres.append((float(hx), float(hy)))
                n_hough += 1

    best = (-np.inf, None, None)
    for k, (cx, cy) in enumerate(centres):
        d = np.hypot(px - cx, py - cy)
        o = np.argsort(d)
        ins, ring, n_disk, n_out = _ring_means(d[o], pv[o], radii)
        # the pupil disk and its edge must be mostly visible (inside the iris annotation)
        frac = n_disk / (np.pi * (0.95 * radii) ** 2)
        frac_out = n_out / (np.pi * (1.35**2 - 1.05**2) * radii**2)
        ok = (frac > 0.5) & (frac_out > 0.4)
        score = np.where(ok, (ring - ins) / (ring + 10.0), -np.inf)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (score[j], (cx, cy, radii[j]), k >= len(centres) - n_hough)
    score, circ, from_hough = best
    if circ is None or not np.isfinite(score):
        return None
    cx, cy, r = circ
    to_img = lambda u, c: (u - half * s) / s + c  # crop -> eye pixels
    return dict(
        iris_cx=icx + x0, iris_cy=icy, iris_r=R,
        pupil_cx=to_img(cx, icx) + x0, pupil_cy=to_img(cy, icy), pupil_r=r / s,
        ratio=r / SCALE, score=float(score), hough=bool(from_hough),
    )


def accept(row, min_score=0.4, min_score_hires=0.2, hires_r=20.0, ratio=(0.18, 0.5)):
    """Is a detection reliable enough to be used as a label? Small irises (celeb, ~12 px
    radius) need more contrast than large ones (CFD, ~33 px); circles at the radius limits
    of the search are usually wrong. Thresholds chosen by eye on contact sheets."""
    sc = row.get("score", float("nan"))
    if not np.isfinite(sc):
        return False
    need = min_score_hires if row["iris_r"] >= hires_r else min_score
    return sc >= need and ratio[0] <= row["ratio"] <= ratio[1]


def eye_boxes(width):
    """Columns of the two eyes of a periorbital image, as train.py splits them."""
    hw = width // 2
    return {"OD": (0, hw), "OS": (width - hw, width)}


def load(ip, mp):
    img = np.array(Image.open(ip).convert("RGB"))
    m = np.array(Image.open(mp))
    m = m[..., 0] if m.ndim == 3 else m
    if m.shape != img.shape[:2]:
        m = np.array(Image.fromarray(m).resize(img.shape[1::-1], Image.NEAREST))
    return img, m


def run(pairs, workers=8):
    """Detect the pupils of every eye: list of row dicts."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        out = list(tqdm(ex.map(_one, pairs, chunksize=8), total=len(pairs), desc="pupils"))
    return [r for rows in out for r in rows]


def _one(pair):
    ip, mp = pair
    img, m = load(ip, mp)
    rows = []
    for side, box in eye_boxes(img.shape[1]).items():
        r = detect(img, m, box)
        rows.append(dict(name=ip.name, side=side, **(r or {"score": float("nan")})))
    return rows


def draw_pupil(mask, row, label=PUPIL_LABEL):
    """Mask with the pupil disk (inside the iris only) set to `label`."""
    m = mask.copy()
    disk = np.zeros(m.shape, np.uint8)
    cv2.circle(
        disk, (round(row["pupil_cx"] * 16), round(row["pupil_cy"] * 16)),
        round(row["pupil_r"] * 16), 1, -1, lineType=cv2.LINE_8, shift=4,
    )
    m[(disk > 0) & (m == IRIS_LABEL)] = label
    return m


def sheet(pairs, rows, out, n=48, ncol=8, seed=0):
    """Contact sheet of random eyes: iris circle (cyan), pupil (yellow), score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {(r["name"], r["side"]): r for r in rows}
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pairs), min(n, len(pairs)), replace=False)
    nrow = (len(pick) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow), squeeze=False)
    for a in axs.flat:
        a.set_axis_off()
    for a, i in zip(axs.flat, pick):
        ip, mp = pairs[i]
        side = "OD" if rng.random() < 0.5 else "OS"
        r = by.get((ip.name, side))
        img, _ = load(ip, mp)
        if not r or not np.isfinite(r.get("score", np.nan)):
            a.set_title(f"{side} none", fontsize=7)
            continue
        R = r["iris_r"]
        x0, x1 = int(r["iris_cx"] - 1.6 * R), int(r["iris_cx"] + 1.6 * R)
        y0, y1 = int(r["iris_cy"] - 1.3 * R), int(r["iris_cy"] + 1.3 * R)
        a.imshow(img, extent=(-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
        a.set_xlim(x0, x1)
        a.set_ylim(y1, y0)
        a.add_patch(plt.Circle((r["iris_cx"], r["iris_cy"]), R, fill=False, color="c", lw=0.8))
        a.add_patch(
            plt.Circle((r["pupil_cx"], r["pupil_cy"]), r["pupil_r"], fill=False, color="y", lw=0.8)
        )
        a.set_title(f"{side} s={r['score']:.2f} p/i={r['ratio']:.2f}", fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    plt.close(fig)


FIELDS = ["name", "side", "score", "hough", "ratio", "iris_cx", "iris_cy", "iris_r",
          "pupil_cx", "pupil_cy", "pupil_r"]


def read_csv(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (r[k] if k in ("name", "side") else
                             (r[k] == "True") if k == "hough" else
                             float(r[k]) if r[k] else float("nan")) for k in FIELDS})
    return rows


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eyemodel3d.train import list_pairs

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="CSV of detections")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--sheet", help="PNG contact sheet of random detections")
    ap.add_argument("--write-masks", help="folder for a copy of the dataset with pupils")
    ap.add_argument("--min-score", type=float, default=0.4,
                    help="contrast needed to keep a pupil (small irises)")
    ap.add_argument("--min-score-hires", type=float, default=0.2,
                    help="contrast needed for irises of radius >= 20 px")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pairs = list_pairs(args.data, args.max_images)
    rows = run(pairs, args.workers)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    keep = lambda r: accept(r, args.min_score, args.min_score_hires)
    sc = np.array([r["score"] for r in rows], float)
    print(f"{len(rows)} eyes, {np.isfinite(sc).sum()} with an iris; "
          f"pupil kept for {np.mean([keep(r) for r in rows]):.1%}; written to {args.out}")
    if args.sheet:
        sheet(pairs, rows, args.sheet)

    if args.write_masks:
        out = Path(args.write_masks)
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(exist_ok=True)
        by = {}
        for r in rows:
            by.setdefault(r["name"], []).append(r)
        for ip, mp in tqdm(pairs, desc="masks"):
            img_dst = out / "images" / ip.name
            if not img_dst.exists():
                img_dst.symlink_to(ip.resolve())
            m = np.array(Image.open(mp))
            m = m[..., 0] if m.ndim == 3 else m
            for r in by.get(ip.name, []):
                if keep(r):
                    m = draw_pupil(m, r)
            Image.fromarray(m.astype(np.uint8)).save(out / "masks" / mp.name)
        print(f"dataset with pupils (label {PUPIL_LABEL}) in {out}")


if __name__ == "__main__":
    main()
