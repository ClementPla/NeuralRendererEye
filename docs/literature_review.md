# EyeModel3D: related work

**Our method.** A network regresses the parameters of an analytic 3D eye model, plus the eye's laterality, from a single periorbital photo. The model has an eyeball, a limbus and pupil, head-fixed eyelids, a caruncle, and a camera. Training needs no 3D labels. A differentiable soft ray-caster renders class masks from the predicted parameters, and the loss compares them with 2D semantic annotations (CE + Dice + moments). The training data is the periorbital dataset of Nahass et al. (CFD + CelebAMask-HQ, 2842 images).

## 1. Parametric eye models fitted to images

- **Analysis-by-synthesis eye-region fitting.** Wood et al. [1] fit a morphable eye-region model and an anatomical eyeball to a photo by gradient descent on photometric error, then read gaze from the fitted pose. This is the same idea as ours, but it optimizes each image instead of using a network, and it fits appearance rather than masks.
- **Eyeball capture.** Bérard et al. [2, 3] build high-quality and parametric eyeball models (sclera, cornea, iris) for capture in computer graphics. Świrski & Dodgson [4] fit a 3D eyeball to pupil ellipses seen over time.
- **Eyelids.** Wen et al. [5] track 3D eyelids in real time by fitting an eyelid model to semantic eyelid edges detected by a CNN. This is the closest eyelid work: a parametric lid driven by semantic evidence. It optimizes per frame and is tied to a face-tracking system.
- **Synthetic eyes.** UnityEyes [6] samples eye-region model parameters to render millions of training images for gaze estimation, the reverse direction (model → images).

## 2. Networks that predict model parameters through a differentiable renderer

- **Faces.** MoFA [7] and Genova et al. [8] train an encoder to output 3D morphable-model parameters, decoded by a differentiable renderer and supervised by 2D signals only. This is our training scheme.
- **Bodies.** Pavlakos et al. [9] regress SMPL parameters and fine-tune with a loss on the silhouette rendered by a differentiable renderer. That is mask supervision of a parametric model, as in our work.
- **Eyes (closest match).** Popovic et al. [10] predict 3D eyeball parameters with a transformer, project them to pupil and iris masks, and supervise with those masks plus a few 3D gaze labels. They do not model eyelids or a caruncle, and they work on near-eye (headset) video.
- **Soft rendering.** OpenDR [11], the Neural Mesh Renderer [12] and SoftRas [13] make rasterization differentiable. Like SoftRas, our renderer uses sigmoid-softened edges, with a softness `τ` that is annealed during training. Ours is analytic ray casting rather than a mesh rasterizer.

## 3. Shape-aware eye segmentation

- **Intermediate maps.** Deep Pictorial Gaze [14] predicts "gazemaps" (rendered eyeball and iris masks) as an intermediate representation.
- **Hidden structure.** EllSeg [15] segments the *full* pupil and iris ellipses, including the parts hidden by the lids. It shares our motivation that a shape prior is more robust to occlusion than free per-pixel segmentation.
- **Segmentation baselines.** OpenEDS [16] (near-eye sclera/iris/pupil segmentation) and RITnet [17] are standard eye-segmentation benchmarks and models without any geometric model.

## 4. Data and ophthalmic measurement

- **Training data.** The periorbital segmentation dataset of Nahass et al. [18] provides our training masks (iris, sclera, lids, caruncle, brows). It comes with a toolkit that measures periorbital distances (such as margin reflex distance and palpebral fissure) from the predicted masks.
- **Advantage of our model.** Our model gives these measurements as parameters, in mm and degrees up to the overall scale. It extrapolates through occlusions: the iris is a circle on a sphere even where the lids hide it.

## 5. Training choices

- **Zoom as focal length.** We treat a zoom as a change of focal length and predict `f` on a log scale. This relates to work that makes networks aware of camera intrinsics, e.g. CAM-Convs [19].
- **Backbones.** The transformer backbones are ViT [20] and Swin [21], used through timm.
- **Occlusion augmentation.** Cutout [22] is applied to the image only, while the mask target is unchanged.

## What seems new

We found no prior work that combines all of these:
1. a single-image network, with no per-image optimization;
2. a full analytic periocular model (eyeball, iris, eyelids, caruncle, camera, laterality);
3. trained only from 2D *semantic masks* of ordinary face photos through a differentiable renderer;
4. aimed at ophthalmic and periorbital measurement rather than gaze.

The nearest works are [10], which has no eyelids and uses near-eye images, and [5], which optimizes per frame with an edge-based eyelid fit.

## References

1. E. Wood, T. Baltrušaitis, L.-P. Morency, P. Robinson, A. Bulling. *A 3D Morphable Eye Region Model for Gaze Estimation.* ECCV 2016. [link](https://perceptualui.org/publications/wood16_eccv/)
2. P. Bérard, D. Bradley, M. Nitti, T. Beeler, M. Gross. *High-Quality Capture of Eyes.* ACM TOG (SIGGRAPH Asia) 2014.
3. P. Bérard, D. Bradley, M. Gross, T. Beeler. *Lightweight Eye Capture Using a Parametric Model.* ACM TOG (SIGGRAPH) 2016.
4. L. Świrski, N. Dodgson. *A Fully-Automatic, Temporal Approach to Single Camera, Glint-Free 3D Eye Model Fitting.* PETMEI 2013.
5. Q. Wen, F. Xu, M. Lu, J.-H. Yong. *Real-time 3D Eyelids Tracking from Semantic Edges.* ACM TOG 36(6), SIGGRAPH Asia 2017. [pdf](https://lu-m13.github.io/data/siga2017_eyelid_paper.pdf)
6. E. Wood, T. Baltrušaitis, L.-P. Morency, P. Robinson, A. Bulling. *Learning an Appearance-Based Gaze Estimator from One Million Synthesised Images.* ETRA 2016.
7. A. Tewari et al. *MoFA: Model-based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction.* ICCV 2017.
8. K. Genova et al. *Unsupervised Training for 3D Morphable Model Regression.* CVPR 2018.
9. G. Pavlakos, L. Zhu, X. Zhou, K. Daniilidis. *Learning to Estimate 3D Human Pose and Shape from a Single Color Image.* CVPR 2018. [arXiv](https://arxiv.org/abs/1805.04092)
10. N. Popovic, D. Christodoulou, D. P. Paudel, X. Wang, L. Van Gool. *Model-aware 3D Eye Gaze from Weak and Few-shot Supervisions.* ISMAR 2023. [arXiv](https://arxiv.org/abs/2311.12157)
11. M. Loper, M. Black. *OpenDR: An Approximate Differentiable Renderer.* ECCV 2014.
12. H. Kato, Y. Ushiku, T. Harada. *Neural 3D Mesh Renderer.* CVPR 2018.
13. S. Liu, T. Li, W. Chen, H. Li. *Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning.* ICCV 2019.
14. S. Park, A. Spurr, O. Hilliges. *Deep Pictorial Gaze Estimation.* ECCV 2018. [arXiv](https://arxiv.org/abs/1807.10002)
15. R. S. Kothari et al. *EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking.* IEEE TVCG 2021. [arXiv](https://arxiv.org/abs/2007.09600)
16. S. J. Garbin et al. *OpenEDS: Open Eye Dataset.* arXiv 2019.
17. A. K. Chaudhary et al. *RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking.* ICCV Workshops 2019.
18. G. R. Nahass et al. *Open-Source Periorbital Segmentation Dataset for Ophthalmic Applications.* arXiv 2409.20407 (2024); Ophthalmology Science 2025. [arXiv](https://arxiv.org/abs/2409.20407)
19. J. M. Fácil et al. *CAM-Convs: Camera-Aware Multi-Scale Convolutions for Single-View Depth.* CVPR 2019.
20. A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
21. Z. Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
22. T. DeVries, G. W. Taylor. *Improved Regularization of Convolutional Neural Networks with Cutout.* arXiv 2017.
