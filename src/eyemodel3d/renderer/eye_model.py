"""
Minimal differentiable eye model (PyTorch): eyeball + limbus + pupil + eyelids + caruncle,
rendered by per-pixel ray casting into soft class masks. Batched: every parameter can be a
0-d tensor (single render) or a (B,) tensor (batch of renders).

Conventions (OpenCV-like camera):
  - World origin = eyeball centre = head frame origin. Units: mm.
  - At rest the camera sits at (0, 0, -D) looking along +z; x = image right, y = image down.
  - At rest the eye looks straight at the camera (gaze = -z).
  - Eye:    +yaw   -> gaze toward image right,  +pitch -> gaze up.
  - Camera: +az    -> camera moves to the right, +el    -> camera moves up
            (the camera orbits the eyeball centre and keeps looking at it);
            roll   -> rotation about the optical axis (head tilt in the image);
            tx, ty -> principal-point shift, as a fraction of image width / height:
                      the eyeball centre is drawn at (W/2 + tx*W, H/2 + ty*H).

Geometry:
  - Limbus = circle of radius r_limbus lying ON the sphere (radius R_EYE), so its
    plane sits at depth sqrt(R_EYE^2 - r_limbus^2) along the gaze direction.
  - Pupil = circle of radius r_pupil in the same plane, offset by (dx, dy) in eye frame.
  - The spherical cap in front of the limbus plane (the "cornea") is transparent;
    the rest of the sphere is opaque, so the iris is correctly hidden at oblique views.
  - Eyelids = two margin curves on a HEAD-FIXED "lid sphere" (radius R_LID, concentric
    with the globe, slightly larger). They do not rotate with the eye, except for an
    optional linear coupling of lid height to vertical gaze (k_up, k_low).
    Lid margins are written in head-frame spherical coordinates on that sphere:
        phi   = horizontal angle (+ toward image right at rest)
        theta = elevation        (+ up)
    Between the canthi (phi_left, phi_right; heights h_left, h_right):
        s        = (phi - phi_left) / (phi_right - phi_left)            in [0, 1]
        base(s)  = h_left + (h_right - h_left) * s                      canthal line
        bump(s)  = 4 u (1 - u),  u = s ** exp(skew)                     0 at canthi, 1 at peak
        theta_up = base + (open_up  + k_up  * pitch) * bump_up
        theta_lo = base - (open_low - k_low * pitch) * bump_low
    skew > 0 moves the peak of that margin toward image right.
    Rays entering the lid sphere between the margins see the globe; all other rays
    see lid/skin. Rays inside the aperture that miss the globe are also lid/skin.
  - Caruncle = medial part of the aperture, cut off by a boundary curve on the lid sphere.
    It is opaque and head-fixed (it does not rotate with gaze). With
        delta   = angular distance from the medial canthus, along phi
        dtheta  = theta - (medial canthus height)
    the caruncle is where
        delta < car_width + car_tilt * dtheta - car_round * dtheta^2 / car_width
    car_round = 0 gives a straight cut; > 0 bulges the boundary toward the cornea at
    canthal height. `medial` = +1 if the medial canthus is on the image right
    (a right eye / OD seen from the front), -1 if on the image left (OS).

Ignored for now: corneal refraction, iris convexity, lid thickness, everted lids,
plica semilunaris as a separate structure, torsion.

Usage:
  python eye_model.py              # interactive sliders
  python eye_model.py --fit        # gradient-descent fitting demo -> fit_demo.png
"""

import argparse
import math

import torch

R_EYE = 12.0  # eyeball radius (mm)
R_LID = 12.7  # head-fixed lid sphere radius (mm); lids rest just outside the globe
CLASSES = ["lid/skin", "sclera", "iris", "pupil", "caruncle"]
COLORS = torch.tensor(
    [
        [0.80, 0.60, 0.50],  # lid / skin (or background when lids are disabled)
        [0.95, 0.93, 0.88],  # sclera
        [0.25, 0.50, 0.80],  # iris
        [0.05, 0.05, 0.05],  # pupil
        [0.85, 0.35, 0.45],  # caruncle
    ]
)

# name: (group, lo, hi, default). Natural units: angles in degrees, lengths in mm,
# tx / ty in fractions of the image size, f in pixels, k_* and skew/round/tilt unitless.
# lo / hi are both the slider ranges and the bounds of a network's predictions.
PARAM_SPECS = {
    # eye
    "yaw": ("eye", -60, 60, 0),
    "pitch": ("eye", -45, 45, 0),
    "r_limbus": ("eye", 4.5, 7.0, 5.9),
    "r_pupil": ("eye", 0.5, 4.5, 2.0),
    "dx": ("eye", -1.5, 1.5, 0.0),
    "dy": ("eye", -1.5, 1.5, 0.0),
    # camera
    "az": ("cam", -60, 60, 0),
    "el": ("cam", -45, 45, 0),
    "roll": ("cam", -45, 45, 0),
    "D": ("cam", 30, 150, 60),
    "f": ("cam", 200, 1500, 600),
    "tx": ("cam", -0.5, 0.5, 0.0),
    "ty": ("cam", -0.5, 0.5, 0.0),
    # lids (k_* are rough guesses)
    "open_up": ("lids", -5, 40, 24),
    "open_low": ("lids", -5, 40, 26),
    "phi_left": ("lids", -85, -30, -62),
    "phi_right": ("lids", 30, 85, 62),
    "h_left": ("lids", -15, 15, 0),
    "h_right": ("lids", -15, 15, 0),
    "skew_up": ("lids", -1.5, 1.5, 0.0),
    "skew_low": ("lids", -1.5, 1.5, 0.0),
    "k_up": ("lids", 0, 1.2, 0.6),
    "k_low": ("lids", 0, 1.2, 0.3),
    # caruncle
    "car_width": ("lids", 0, 40, 20),
    "car_round": ("lids", -1, 3, 0.5),
    "car_tilt": ("lids", -1.5, 1.5, 0.0),
}
ANGLE_PARAMS = {
    "yaw",
    "pitch",
    "az",
    "el",
    "roll",
    "open_up",
    "open_low",
    "phi_left",
    "phi_right",
    "h_left",
    "h_right",
    "car_width",
}
DEFAULTS = {k: v[3] for k, v in PARAM_SPECS.items()}
DEFAULTS["medial"] = 1  # +1: medial canthus on image right (OD seen from the front)


def _e(x):
    """(B,) -> (B, 1, 1) for broadcasting against (B, H, W) maps."""
    return x[:, None, None]


# ----------------------------------------------------------------------------- parameters
def make_leaves(requires_grad=False, **vals):
    """Dict of 0-d tensors in natural units (degrees, mm, ...), defaults for missing keys."""
    v = dict(DEFAULTS, **vals)
    leaves = {
        k: torch.tensor(float(v[k]), requires_grad=requires_grad) for k in PARAM_SPECS
    }
    leaves["medial"] = torch.tensor(
        1.0 if v["medial"] >= 0 else -1.0
    )  # discrete, never optimized
    return leaves


def split_params(leaves):
    """Natural-unit dict -> (eye, cam, lids) dicts with angles in radians, for render()."""
    groups = {"eye": {}, "cam": {}, "lids": {}}
    for k, v in leaves.items():
        g = "lids" if k == "medial" else PARAM_SPECS[k][0]
        groups[g][k] = torch.deg2rad(v) if k in ANGLE_PARAMS else v
    return groups["eye"], groups["cam"], groups["lids"]


def make_params(requires_grad=False, **vals):
    return split_params(make_leaves(requires_grad, **vals))


def _batchify(d, B):
    return {k: (v.reshape(-1).expand(B) if v.numel() == 1 else v) for k, v in d.items()}


# ----------------------------------------------------------------------------- rotations
def rot_x(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(*a.shape, 3, 3)


def rot_y(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(*a.shape, 3, 3)


def rot_z(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*a.shape, 3, 3)


# ----------------------------------------------------------------------------- camera
def make_rays(cam, H, W):
    """Ray origins o (B, 3) and unit directions d (B, H, W, 3) in world coordinates."""
    dev = cam["f"].device
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=dev) + 0.5,
        torch.arange(W, dtype=torch.float32, device=dev) + 0.5,
        indexing="ij",
    )
    f = _e(cam["f"])
    cx = W / 2 + _e(cam["tx"]) * W
    cy = H / 2 + _e(cam["ty"]) * H
    x, y = (u - cx) / f, (v - cy) / f
    d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True)
    R_c2w = rot_y(-cam["az"]) @ rot_x(-cam["el"]) @ rot_z(cam["roll"])
    d = torch.einsum("bij,bhwj->bhwi", R_c2w, d_cam)
    z = torch.zeros_like(cam["D"])
    o = torch.einsum("bij,bj->bi", R_c2w, torch.stack([z, z, -cam["D"]], dim=-1))
    return o, d


def sphere_entry(o, d, radius, tau):
    """Soft hit mask (B, H, W) and entry point (B, H, W, 3) of rays on a sphere centred at 0."""
    ob = o[:, None, None, :]
    b = (d * ob).sum(-1)
    oo = _e((o * o).sum(-1))
    dist = torch.sqrt((oo - b * b).clamp(min=1e-9))  # distance centre <-> ray
    hit = torch.sigmoid((radius - dist) / tau)
    t_in = -b - torch.sqrt((b * b - (oo - radius**2)).clamp(min=1e-9))
    return hit, ob + t_in[..., None] * d


# ----------------------------------------------------------------------------- eyelids
def lid_aperture(o, d, lids, pitch, tau):
    """Soft masks (B, H, W): rays through the palpebral fissure, and the caruncle region
    (to be intersected with the fissure)."""
    hit, P = sphere_entry(o, d, R_LID, tau)
    phi = torch.atan2(P[..., 0], -P[..., 2])
    theta = torch.asin((-P[..., 1] / R_LID).clamp(-0.999, 0.999))

    L = {k: _e(v) for k, v in lids.items()}
    phi_l, phi_r = L["phi_left"], L["phi_right"]
    s = ((phi - phi_l) / (phi_r - phi_l)).clamp(0, 1)
    base = L["h_left"] + (L["h_right"] - L["h_left"]) * s

    def bump(skew):
        u = s.clamp(1e-4, 1 - 1e-4) ** torch.exp(skew)
        return 4 * u * (1 - u)

    p = _e(pitch)
    th_up = base + (L["open_up"] + L["k_up"] * p) * bump(L["skew_up"])
    th_lo = base - (L["open_low"] - L["k_low"] * p) * bump(L["skew_low"])

    tau_a = tau / R_LID  # mm -> radians on the lid sphere
    inside = (
        torch.sigmoid((th_up - theta) / tau_a)
        * torch.sigmoid((theta - th_lo) / tau_a)
        * torch.sigmoid((phi - phi_l) / tau_a)
        * torch.sigmoid((phi_r - phi) / tau_a)
    )

    # Caruncle: medial cut of the fissure
    m = L["medial"]  # +1: medial canthus on image right
    phi_mc = 0.5 * (phi_l + phi_r) + 0.5 * m * (phi_r - phi_l)
    th_mc = 0.5 * (L["h_left"] + L["h_right"]) + 0.5 * m * (L["h_right"] - L["h_left"])
    delta = m * (phi_mc - phi)  # >= 0 inside the fissure
    dth = theta - th_mc
    w = L["car_width"].clamp(min=1e-3)
    boundary = w + L["car_tilt"] * dth - L["car_round"] * dth**2 / w
    caruncle = torch.sigmoid((boundary - delta) / tau_a)

    return hit * inside, caruncle


# ----------------------------------------------------------------------------- renderer
def render(eye, cam, lids=None, H=256, W=256, tau=0.05):
    """
    eye:  yaw, pitch (rad), r_limbus, r_pupil, dx, dy (mm)
    cam:  az, el, roll (rad), D (mm), f (px), tx, ty (fraction of image size)
    lids: see module docstring (angles in rad), or None for no lids
    Each value is a 0-d tensor (single render) or a (B,) tensor (batch).
    Use split_params(...) to build these dicts from natural units.
    tau:  edge softness in mm (larger = smoother gradients, blurrier masks)
    Returns per-pixel class probabilities in CLASSES order, summing to 1:
    (H, W, 5) if the eye parameters are 0-d, else (B, H, W, 5).
    """
    single = eye["yaw"].dim() == 0
    B = 1 if single else eye["yaw"].shape[0]
    eye, cam = _batchify(eye, B), _batchify(cam, B)
    lids = _batchify(lids, B) if lids is not None else None

    o, d = make_rays(cam, H, W)
    ob = o[:, None, None, :]

    # Eye rotation and limbus plane
    R_eye = rot_y(-eye["yaw"]) @ rot_x(-eye["pitch"])  # (B, 3, 3)
    n = -R_eye[..., 2]  # R_eye @ (0, 0, -1) = gaze (B, 3)
    r_l = eye["r_limbus"].clamp(1.0, R_EYE - 0.5)
    r_p = eye["r_pupil"].clamp(0.3, R_EYE)
    depth = torch.sqrt(R_EYE**2 - r_l**2)
    c_l = depth[:, None] * n  # limbus centre (B, 3)
    off = torch.stack([eye["dx"], eye["dy"], torch.zeros_like(eye["dx"])], dim=-1)
    c_p = c_l + (R_eye @ off[..., None])[..., 0]  # pupil centre (in-plane)

    # Globe silhouette, and whether the ray enters through the transparent corneal cap
    sphere, E = sphere_entry(o, d, R_EYE, tau)
    nb = n[:, None, None, :]
    cap = torch.sigmoid(((E * nb).sum(-1) - _e(depth)) / tau)

    # Ray / limbus-plane intersection
    denom = (d * nb).sum(-1)
    front = (denom < 0).float()  # plane faces the camera
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, -1e-6), denom)
    t = _e(((c_l - o) * n).sum(-1)) / denom
    X = ob + t[..., None] * d

    rho_l = (X - c_l[:, None, None, :]).norm(dim=-1)
    rho_p = (X - c_p[:, None, None, :]).norm(dim=-1)
    limbus = (
        front * cap * torch.sigmoid((_e(r_l) - rho_l) / tau)
    )  # inside limbus & visible
    pupil = torch.sigmoid((_e(r_p) - rho_p) / tau)  # forced inside limbus below

    # Eyelid aperture gates everything behind it; the caruncle is opaque inside it
    if lids is not None:
        A, car = lid_aperture(o, d, lids, eye["pitch"], tau)
    else:
        A, car = torch.ones_like(sphere), torch.zeros_like(sphere)
    caruncle = A * car
    globe = A * (1 - car) * sphere

    probs = torch.stack(
        [
            1 - globe - caruncle,
            globe * (1 - limbus),
            globe * limbus * (1 - pupil),
            globe * limbus * pupil,
            caruncle,
        ],
        dim=-1,
    )
    return probs[0] if single else probs


def to_rgb(probs):
    return (probs @ COLORS.to(probs.device)).clamp(0, 1).detach().cpu().numpy()


# ----------------------------------------------------------------------------- interactive
def interactive():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    left = [k for k, v in PARAM_SPECS.items() if v[0] in ("eye", "cam")]
    right = [k for k, v in PARAM_SPECS.items() if v[0] == "lids"] + ["medial"]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    plt.subplots_adjust(bottom=0.42)
    ax.set_axis_off()
    with torch.no_grad():
        im = ax.imshow(to_rgb(render(*make_params())))

    sliders = {}
    for x0, column in ((0.10, left), (0.60, right)):
        for i, name in enumerate(column):
            lo, hi = (-1, 1) if name == "medial" else PARAM_SPECS[name][1:3]
            sax = fig.add_axes([x0, 0.37 - i * 0.025, 0.30, 0.018])
            sliders[name] = Slider(
                sax,
                name,
                lo,
                hi,
                valinit=DEFAULTS[name],
                valstep=2 if name == "medial" else None,
            )

    def update(_):
        vals = {k: s.val for k, s in sliders.items()}
        with torch.no_grad():
            im.set_data(to_rgb(render(*make_params(**vals))))
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)
    plt.show()


# ----------------------------------------------------------------------------- fitting demo
def soft_dice_loss(probs, target_onehot, eps=1e-6):
    inter = (probs * target_onehot).sum((0, 1))
    union = probs.sum((0, 1)) + target_onehot.sum((0, 1))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def fit_demo(steps=400, out="fit_demo.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Target: known parameters; camera fixed and known
    true_leaves = make_leaves(
        yaw=20,
        pitch=-8,
        r_limbus=6.1,
        r_pupil=1.6,
        dx=0.4,
        dy=-0.3,
        open_up=16,
        open_low=22,
        h_right=4,
        skew_up=0.3,
        car_width=25,
        car_round=1.0,
        car_tilt=0.3,
    )
    with torch.no_grad():
        target = render(*split_params(true_leaves))
    target_onehot = torch.nn.functional.one_hot(target.argmax(-1), len(CLASSES)).float()

    # Initial guess: defaults. Camera, k_up / k_low (confounded with open_up / open_low in a
    # single image) and laterality are not optimized.
    leaves = make_leaves(requires_grad=True)
    frozen = {"az", "el", "roll", "D", "f", "tx", "ty", "k_up", "k_low", "medial"}
    for k in frozen:
        leaves[k].requires_grad_(False)
    angles = [v for k, v in leaves.items() if k not in frozen and k in ANGLE_PARAMS]
    others = [v for k, v in leaves.items() if k not in frozen and k not in ANGLE_PARAMS]
    opt = torch.optim.Adam(
        [
            {"params": angles, "lr": 0.5},  # degrees per step
            {"params": others, "lr": 0.02},
        ]
    )
    with torch.no_grad():
        init_rgb = to_rgb(render(*split_params(leaves)))

    for it in range(steps):
        tau = 0.5 * (0.05 / 0.5) ** (it / (steps - 1))  # coarse-to-fine edge softness
        loss = soft_dice_loss(render(*split_params(leaves), tau=tau), target_onehot)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == steps - 1:
            print(f"it {it:4d}  tau {tau:.3f}  loss {loss.item():.4f}")

    for k in PARAM_SPECS:
        if k not in frozen:
            print(
                f"{k:10s} true {true_leaves[k].item():8.3f}   fitted {leaves[k].item():8.3f}"
            )

    with torch.no_grad():
        fit_rgb = to_rgb(render(*split_params(leaves)))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    for a, img, title in zip(
        axs, [init_rgb, to_rgb(target), fit_rgb], ["initial guess", "target", "fitted"]
    ):
        a.imshow(img)
        a.set_title(title)
        a.set_axis_off()
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fit", action="store_true", help="run the fitting demo instead of sliders"
    )
    args = ap.parse_args()
    fit_demo() if args.fit else interactive()
