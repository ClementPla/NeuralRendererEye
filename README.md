# NeuralRendererEye (eyemodel3d)

A network regresses the parameters of an analytic 3D eye model (eyeball, iris, pupil, eyelids,
caruncle, camera, laterality) from a periorbital photo. It is trained from 2D segmentation masks
through a differentiable soft ray-caster. See `docs/literature_review.md`.

## Install

```bash
pip install -e .
```

## Live webcam demo

Open `notebooks/webcam.ipynb`, run all cells and press **Start**. It uses the included ViT-S
model (`runs/vit_s_pupil/best.pt`, predicts the pupil) and OpenCV's YuNet face detector
(`models/`). It runs on CPU too, more slowly. `SOURCE` selects the camera index or a video file.

## Other entry points

- `python -m eyemodel3d.train --help`: training (needs the periorbital dataset).
- `python -m eyemodel3d.pupil --help`: automatic pupil pseudo-labels.
- `notebooks/explorer.ipynb`: interactive effect of each model parameter.
- `notebooks/predictions.ipynb`, `notebooks/interop.ipynb`: predictions on the datasets.
