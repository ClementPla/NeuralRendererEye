"""
Reframe the interop photos like the training inputs and save them.

    python -m eyemodel3d.interop --out interop_prepared --ckpt runs/exp/best.pt
    python -m eyemodel3d.interop --out interop_prepared --mode iris --overrides fixes.csv

--mode letterbox (default): whole photo, for models trained with zooms;
--mode iris: iris-based framing (uses --overrides), for models trained without zooms.
--mirror-os: mirror left eyes (only for checkpoints that do not predict laterality).

Writes <out>/<dataset>/<name>.png, <out>/index.csv (values used, scale, offset) and
<out>/preview_<dataset>.png (dashed circle = where the iris should be).
--template FILE writes the current per-image values as an editable overrides CSV.
"""

import argparse
import csv
import json
from pathlib import Path

from .core import (
    DEFAULT_ROOT,
    DEFAULT_SIZE,
    list_images,
    load_overrides,
    prepare,
    preview_sheet,
    write_template,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", help="output folder for the prepared images")
    ap.add_argument("--datasets", nargs="*", help="only these subfolders")
    ap.add_argument("--overrides", help="CSV of per-image corrections")
    ap.add_argument("--template", help="write an editable overrides CSV and stop")
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"))
    ap.add_argument("--ckpt", help="take --size from this checkpoint's meta")
    ap.add_argument("--mode", choices=["letterbox", "iris"], default="letterbox")
    ap.add_argument("--mirror-os", action="store_true", help="mirror left eyes")
    args = ap.parse_args()

    size = tuple(args.size) if args.size else DEFAULT_SIZE
    if args.ckpt:
        import torch

        size = tuple(torch.load(args.ckpt, map_location="cpu", weights_only=False)["meta"]["size"])

    ov = load_overrides(args.overrides) if args.overrides else None
    recs = list_images(args.root, args.datasets, ov)
    print(f"{len(recs)} images, output size H x W = {size[0]} x {size[1]}")

    if args.template:
        write_template(recs, args.template)
        print(f"template written to {args.template}")
        return
    if not args.out:
        ap.error("--out is required (or use --template)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["dataset", "name", "file", "laterality", "mirrored", "iris_cx", "iris_cy",
             "iris_w", "scale", "offset_x", "offset_y", "source_w", "source_h"]
        )
        for r in recs:
            p = prepare(r, size, args.mode, args.mirror_os)
            dst = out / r.dataset / (r.path.stem + ".png")
            dst.parent.mkdir(exist_ok=True)
            p.image.save(dst)
            w.writerow(
                [r.dataset, r.path.name, dst.relative_to(out), r.laterality, p.mirrored,
                 r.iris_cx, r.iris_cy, r.iris_w, round(p.scale, 6), *p.offset, *p.source_size]
            )
    (out / "settings.json").write_text(
        json.dumps(dict(size=size, root=args.root, mode=args.mode, mirror_os=args.mirror_os), indent=2)
    )

    for ds in sorted({r.dataset for r in recs}):
        preview_sheet(
            [r for r in recs if r.dataset == ds],
            out / f"preview_{ds}.png",
            size,
            args.mode,
            args.mirror_os,
        )
    print(f"written to {out}")


if __name__ == "__main__":
    main()
