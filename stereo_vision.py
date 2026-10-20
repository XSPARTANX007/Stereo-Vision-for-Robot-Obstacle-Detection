#!/usr/bin/env python3
"""
Stereo Vision for Robot Obstacle Detection
==========================================
Pipeline (each step is one small function below):
  1. match_points      find the same points in the left and right image
  2. eight_point       compute the Fundamental matrix F  (own implementation + RANSAC)
  3. epipoles / epilines   epipolar geometry from F
  4. rectify           warp both images so matching points lie on the same row
  5. disparity         how far each pixel moved between left and right
  6. depth             Z = focal * baseline / disparity
  7. to_cloud          3D points (X, Y, Z) -> view / save as .ply
  8. obstacles         mark everything closer than a danger distance

Run:  python stereo_vision.py                       (built-in demo, no camera)
      python stereo_vision.py --left L.png --right R.png --focal 700 --baseline 0.10
      python stereo_vision.py --live --cams 0 1 --focal 700 --baseline 0.10
"""
import argparse
import os

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 0. Demo scene (so the project runs without any camera or dataset)
# --------------------------------------------------------------------------
def make_demo_pair(w=640, h=480, f=500.0, B=0.12, misalign=True, return_gt=False):
    """Synthetic stereo pair: slanted floor + 3 obstacles at 1.2 m, 2.0 m, 3.5 m.
    The right camera is rotated 1 degree and shifted 3 px so rectification has work to do."""
    rng = np.random.default_rng(1)
    fine = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.2)
    coarse = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 14)
    tex = cv2.normalize(0.6 * cv2.normalize(fine, None, 0, 1, cv2.NORM_MINMAX) +
                        0.4 * cv2.normalize(coarse, None, 0, 1, cv2.NORM_MINMAX),
                        None, 60, 255, cv2.NORM_MINMAX)
    tex3 = np.dstack([tex] * 3)
    ys, xs = np.mgrid[0:h, 0:w]
    everything = np.ones((h, w), bool)

    def box(x0, x1, y0, y1):
        m = np.zeros((h, w), bool)
        m[y0:y1, x0:x1] = True
        return m

    floor_Z = 8.0 - 4.8 * ys / h                                  # 8 m (top) .. 3.2 m (bottom)
    layers = [(everything, floor_Z, (1.0, 0.9, 0.8)),                # far -> near (painter's order)
              (box(300, 345, 110, 400), 3.5, (0.6, 1.0, 0.6)),
              (box(380, 530, 220, 370), 2.0, (0.6, 0.8, 1.0)),
              (box(110, 260, 170, 400), 1.2, (0.5, 0.5, 1.0))]
    left = np.zeros((h, w, 3), np.float32)
    right = np.zeros((h, w, 3), np.float32)
    gt = np.zeros((h, w), np.float32)                            # true depth (left image) for accuracy tests
    xsf, ysf = xs.astype(np.float32), ys.astype(np.float32)
    for mask, Z, tint in layers:
        Zm = np.broadcast_to(Z, (h, w)).astype(np.float32)
        gt[mask] = Zm[mask]
        layer = tex3 * np.array(tint, np.float32)
        left[mask] = layer[mask]
        mx = xsf + f * B / Zm                                    # right pixel x  ->  left pixel x + disparity
        samp = cv2.remap(layer, mx, ysf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = cv2.remap(mask.astype(np.float32), mx, ysf, cv2.INTER_LINEAR) > 0.5
        right[m] = samp[m]                                       # far -> near: nearer layers overwrite
    right = np.clip(right, 0, 255).astype(np.uint8)
    left = np.clip(left + rng.normal(0, 2, left.shape), 0, 255).astype(np.uint8)
    right = np.clip(right + rng.normal(0, 2, right.shape), 0, 255).astype(np.uint8)
    if misalign:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.0, 1.0)
        M[1, 2] += 3
        right = cv2.warpAffine(right, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return (left, right, gt) if return_gt else (left, right)


# --------------------------------------------------------------------------
# 1. Feature matching
# --------------------------------------------------------------------------
def match_points(gL, gR, max_pts=3000):
    sift = cv2.SIFT_create(nfeatures=max_pts, contrastThreshold=0.02)
    kL, dL = sift.detectAndCompute(gL, None)
    kR, dR = sift.detectAndCompute(gR, None)
    if dL is None or dR is None:
        raise RuntimeError("No features found - the scene needs texture.")
    pairs = cv2.BFMatcher().knnMatch(dL, dR, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    p1 = np.float32([kL[m.queryIdx].pt for m in good])
    p2 = np.float32([kR[m.trainIdx].pt for m in good])
    return p1, p2


# --------------------------------------------------------------------------
# 2. Fundamental matrix: normalized 8-point algorithm (+ RANSAC for bad matches)
#    Convention: x_right^T  F  x_left = 0
# --------------------------------------------------------------------------
def _normalize(p):
    c = p.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return np.c_[p, np.ones(len(p))] @ T.T, T


def eight_point(p1, p2):
    x1, T1 = _normalize(p1)
    x2, T2 = _normalize(p2)
    A = np.stack([x2[:, 0] * x1[:, 0], x2[:, 0] * x1[:, 1], x2[:, 0],
                  x2[:, 1] * x1[:, 0], x2[:, 1] * x1[:, 1], x2[:, 1],
                  x1[:, 0], x1[:, 1], np.ones(len(x1))], axis=1)
    F = np.linalg.svd(A)[2][-1].reshape(3, 3)            # solve A f = 0
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0                                              # enforce rank 2
    F = T2.T @ (U @ np.diag(S) @ Vt) @ T1                 # undo normalization
    return F / np.linalg.norm(F)


def epipolar_error(F, p1, p2):
    """Squared distance (pixels^2) of each match from its epipolar line (Sampson)."""
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    return np.sum(x2 * Fx1, 1) ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)


def ransac_F(p1, p2, iters=2000, thr=1.0):
    """Tunable: iters (max trials), thr (inlier distance in px). Stops early when enough trials were done."""
    rng = np.random.default_rng(0)
    best, n, i = None, iters, 0
    while i < n:
        i += 1
        idx = rng.choice(len(p1), 8, replace=False)
        inl = epipolar_error(eight_point(p1[idx], p2[idx]), p1, p2) < thr ** 2
        if best is None or inl.sum() > best.sum():
            best = inl
            r = best.mean()                                      # adaptive trial count for 99% confidence
            n = int(min(iters, max(100, np.log(0.01) / np.log(max(1 - r ** 8, 1e-9)))))
    return eight_point(p1[best], p2[best]), best


# --------------------------------------------------------------------------
# 3. Epipolar geometry
# --------------------------------------------------------------------------
def epipoles(F):
    """Left epipole: F e = 0.  Right epipole: F^T e' = 0.  (homogeneous)"""
    return np.linalg.svd(F)[2][-1], np.linalg.svd(F.T)[2][-1]


def describe_epipole(e):
    if abs(e[2]) < 1e-4 * np.linalg.norm(e[:2]):
        return "at infinity -> epipolar lines are (almost) horizontal, cameras are parallel"
    return "(%.1f, %.1f) px" % (e[0] / e[2], e[1] / e[2])


def draw_epilines(img, lines, pts, seed=3):
    out = img.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    for (a, b, c), p in zip(lines, pts):
        col = tuple(int(v) for v in rng.integers(40, 255, 3))
        if abs(b) > 1e-9:
            y0, y1 = np.clip([-c / b, -(c + a * w) / b], -1e5, 1e5)
            cv2.line(out, (0, int(y0)), (w, int(y1)), col, 1, cv2.LINE_AA)
        cv2.circle(out, (int(p[0]), int(p[1])), 4, col, -1)
    return out


# --------------------------------------------------------------------------
# 4. Calibrate (steps 1-4 once) -> reusable state, so live mode is fast
# --------------------------------------------------------------------------
def _rect_ok(H, w, h):
    """Reject homographies that blow the image up, shrink it, or throw it far away."""
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(c, H).reshape(-1, 2)
    if not np.all(np.isfinite(q)) or q.min() < -max(w, h) or q[:, 0].max() > 2 * w or q[:, 1].max() > 2 * h:
        return False
    return 0.4 < cv2.contourArea(q.astype(np.float32)) / (w * h) < 2.5


def _disp_range(d):
    """Disparity search range from the matches, with generous margin so the NEAREST obstacle is never cut off."""
    lo, hi = np.percentile(d, [0.5, 99.5])
    pad = 0.25 * (hi - lo) + 12
    min_d = int(np.floor(lo - pad))
    return min_d, int(np.ceil((hi + pad - min_d) / 16) * 16)


def _range_ok(min_d, num_d):
    return num_d <= 256 and abs(min_d) <= 128


def calibrate_pair(gL, gR, prerectified=False):
    """prerectified=True: the pair is already rectified (e.g. RealSense IR imagers) -> F is still
    estimated from the images, but no extra warping is applied."""
    h, w = gL.shape
    try:
        p1, p2 = match_points(gL, gR)
        if len(p1) < 30:
            raise RuntimeError("Only %d matches - point the cameras at a textured scene." % len(p1))
        F, inl = ransac_F(p1, p2)
        if prerectified and inl.sum() < 15:
            raise RuntimeError("Only %d consistent matches." % inl.sum())
    except RuntimeError as e:
        if not prerectified:
            raise
        # Factory-rectified camera: its F is known exactly, so calibration does not need a textured scene.
        ideal = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float) / np.sqrt(2)
        return dict(F=ideal, H1=np.eye(3), H2=np.eye(3), p1=np.zeros((0, 2), np.float32), p2=np.zeros((0, 2), np.float32),
                    n_inliers=0, n_matches=0, min_d=0, num_d=96, offset=0.0,
                    note="%s F could not be measured, so the camera's ideal rectified F is used. "
                         "Turn the IR projector on (key L) or aim at a textured scene to measure it." % e)
    p1, p2 = p1[inl], p2[inl]
    if prerectified:
        ok, H1, H2 = False, np.eye(3), np.eye(3)
    else:
        ok, H1, H2 = cv2.stereoRectifyUncalibrated(p1, p2, F, (w, h))
    warp = lambda p, H: cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)
    d_raw = p1[:, 0] - p2[:, 0]                                  # disparity of matches before rectification
    note = ""
    use_rect = bool(ok) and _rect_ok(H1, w, h) and _rect_ok(H2, w, h)
    if use_rect:
        d_rect = warp(p1, H1)[:, 0] - warp(p2, H2)[:, 0]
        min_d, num_d = _disp_range(d_rect)
        use_rect = _range_ok(min_d, num_d)
    if not use_rect:                                             # unstable -> keep original images
        H1 = H2 = np.eye(3)
        d_rect = d_raw
        min_d, num_d = _disp_range(d_raw)
        note = "" if prerectified else ("Rectification was unstable (few/inconsistent matches or cameras not parallel), "
                "so the original images are used without rectifying.")
        if not prerectified and not _range_ok(min_d, num_d):
            raise RuntimeError("Matches disagree too much (disparity range %d..%d). Mount the cameras parallel, "
                               "use the same resolution on both, and aim at a textured, 3D scene "
                               "(not a flat wall)." % (min_d, min_d + num_d))
    return dict(F=F, H1=H1, H2=H2, p1=p1, p2=p2, n_inliers=int(inl.sum()), n_matches=len(inl),
                min_d=min_d, num_d=num_d, offset=float(np.median(d_raw - d_rect)), note=note)


# --------------------------------------------------------------------------
# 5-6. Disparity and depth
# --------------------------------------------------------------------------
def disparity(rL, rR, min_d, num_d, block, scale=1.0):
    """StereoSGBM. Tunable: block (matching window, odd), num_d (search range, multiple of 16),
    scale (<1 = match on a smaller image: much faster, a little less detail)."""
    h, w = rL.shape[:2]
    if scale < 1.0:
        rL, rR = (cv2.resize(i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for i in (rL, rR))
        min_d, num_d = int(np.floor(min_d * scale)), int(np.ceil(num_d * scale))
    block = max(3, block | 1)                                    # must be odd
    sgbm = cv2.StereoSGBM_create(minDisparity=min_d, numDisparities=int(np.clip(np.ceil(num_d / 16) * 16, 16, 256)),
                                 blockSize=block, P1=8 * block ** 2, P2=32 * block ** 2,
                                 disp12MaxDiff=1, uniquenessRatio=10, preFilterCap=63,
                                 speckleWindowSize=150, speckleRange=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(rL, rR).astype(np.float32) / 16
    d[d < min_d] = np.nan                                        # invalid pixels
    if scale < 1.0:
        d = cv2.resize(d / scale, (w, h), interpolation=cv2.INTER_NEAREST)
    return d


def clean_disparity(d, min_d, max_grad=1.5, min_blob=150):
    """Cleans the raw disparity map so the 3D cloud has no noise:
       median filter -> remove 'flying pixels' at depth edges (disparity jump > max_grad px per pixel)
       -> remove tiny isolated blobs (< min_blob pixels)."""
    f = np.where(np.isfinite(d), d, min_d - 1).astype(np.float32)
    f = cv2.medianBlur(f, 5)
    g = (np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))) / 8
    edge = cv2.dilate((g > max_grad).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    valid = (f >= min_d) & ~edge
    _, lab, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), connectivity=4)
    valid &= ~np.isin(lab, np.where(stats[:, cv2.CC_STAT_AREA] < min_blob)[0])
    return np.where(valid, f, np.nan).astype(np.float32)


def depth_from_disparity(d, focal, baseline, max_depth):
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = focal * baseline / d                                 # Z = f * B / d
    Z[(d <= 0) | (Z > max_depth)] = np.nan
    return Z


# --------------------------------------------------------------------------
# 7. 3D reconstruction
# --------------------------------------------------------------------------
def to_cloud(Z, bgr, focal, max_depth, cx=None, cy=None):
    h, w = Z.shape
    cx, cy = (w / 2 if cx is None else cx), (h / 2 if cy is None else cy)
    ys, xs = np.mgrid[0:h, 0:w]
    X, Y = (xs - cx) * Z / focal, (ys - cy) * Z / focal          # pinhole model
    ok = np.isfinite(Z) & (Z < max_depth)
    return np.c_[X[ok], Y[ok], Z[ok]], bgr[ok][:, ::-1]


def render_cloud(pts, rgb, cam, yaw=0.0, pitch=0.0, zoom=1.0, scale=1.5, splat=None):
    """Fast z-buffer renderer (numpy only). View (0, 0) is the LEFT CAMERA'S OWN VIEW, so the default
    picture matches the calibration image. yaw/pitch orbit the cloud around its centre."""
    f, cx, cy, w, h = cam
    H, W = int(h * scale), int(w * scale)
    c = np.median(pts, axis=0)
    cyw, syw, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cyw, 0, syw], [0, 1, 0], [-syw, 0, cyw]])
    t = min(1.0, (abs(yaw) + abs(pitch)) / 0.4)                  # 0 = original camera view, 1 = orbit around the centre
    q = (pts - c) @ R.T + (c * (1 - t) + np.array([0.0, 0.0, c[2]]) * t)
    z = q[:, 2]
    keep = z > 0.05
    q, z, col = q[keep], z[keep], rgb[keep][:, ::-1]
    u = (f * q[:, 0] / z + cx) * scale
    v = (f * q[:, 1] / z + cy) * scale
    if zoom != 1.0:
        u, v = (u - W / 2) * zoom + W / 2, (v - H / 2) * zoom + H / 2
    u, v = np.rint(u).astype(np.int32), np.rint(v).astype(np.int32)
    splat = splat or int(np.clip(np.ceil(scale * zoom), 1, 5))
    ok = (u >= 0) & (u < W - splat) & (v >= 0) & (v < H - splat)
    u, v, z, col = u[ok], v[ok], z[ok], col[ok]
    order = np.argsort(-z)                                       # far -> near, so nearer points overwrite
    u, v, col = u[order], v[order], col[order]
    img = np.full((H, W, 3), 24, np.uint8)
    for dy in range(splat):
        for dx in range(splat):
            img[v + dy, u + dx] = col
    return img


def _window_open(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def _close_window(name):
    try:
        cv2.destroyWindow(name)
    except cv2.error:                                            # already closed with the X button
        pass


def view_cloud(pts, rgb, cam, max_points=200000, title="3D reconstruction"):
    """Interactive viewer. Drag = rotate | wheel or +/- = zoom | 1/2/3 = front/side/top | R = reset | Q = close."""
    step = max(1, len(pts) // max_points)
    pts, rgb = pts[::step], rgb[::step]
    S = dict(yaw=0.0, pitch=0.0, zoom=1.0, drag=None, dirty=True, fast=False)

    def mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            S.update(drag=(x, y), fast=True)
        elif ev == cv2.EVENT_LBUTTONUP:
            S.update(drag=None, fast=False, dirty=True)
        elif ev == cv2.EVENT_MOUSEMOVE and S["drag"]:
            S["yaw"] += (x - S["drag"][0]) * 0.008
            S["pitch"] -= (y - S["drag"][1]) * 0.008
            S.update(drag=(x, y), dirty=True)
        elif ev == cv2.EVENT_MOUSEWHEEL:
            try:
                flags = cv2.getMouseWheelDelta(flags)
            except Exception:
                pass
            S.update(zoom=float(np.clip(S["zoom"] * (1.1 if flags > 0 else 1 / 1.1), 0.3, 6)), dirty=True)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse)
    while True:
        if S["dirty"]:                                           # only redraw when something changed
            sub = slice(None, None, 4) if S["fast"] else slice(None)        # coarse while dragging = smooth
            img = render_cloud(pts[sub], rgb[sub], cam, S["yaw"], S["pitch"], S["zoom"], splat=3 if S["fast"] else None)
            cv2.putText(img, "drag: rotate | wheel/+/-: zoom | 1 2 3: front/side/top | R: reset | Q: close", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, img)
            S["dirty"] = False
        k = cv2.waitKey(15) & 0xFF
        if k in (ord("q"), 27) or not _window_open(title):
            break
        presets = {ord("r"): (0, 0, 1), ord("1"): (0, 0, 1), ord("2"): (1.0, 0, 1), ord("3"): (0, -1.1, 1)}
        if k in presets:
            S.update(yaw=presets[k][0], pitch=presets[k][1], zoom=presets[k][2], dirty=True)
        if k in (ord("+"), ord("=")):
            S.update(zoom=min(S["zoom"] * 1.15, 6), dirty=True)
        if k in (ord("-"), ord("_")):
            S.update(zoom=max(S["zoom"] / 1.15, 0.3), dirty=True)
    _close_window(title)


def save_ply(path, pts, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n" % len(pts))
        for p, c in zip(pts, rgb):
            f.write("%.4f %.4f %.4f %d %d %d\n" % (*p, *c))


# --------------------------------------------------------------------------
# 8. Obstacle detection
# --------------------------------------------------------------------------
def obstacles(Z, bgr, danger, warn, min_area=400):
    out = bgr.copy()
    dists = []
    for limit, color, label in [(warn, (0, 200, 255), "WARN"), (danger, (0, 0, 255), "STOP")]:
        mask = (np.nan_to_num(Z, nan=1e9) < limit).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dist = np.nanmedian(Z[y:y + h, x:x + w][mask[y:y + h, x:x + w] > 0])
            dists.append(dist)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, "%s %.2f m" % (label, dist), (x, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if dists:
        nearest = min(dists)
    else:                                                        # nothing inside the warn zone
        nearest = np.nanpercentile(Z, 2) if np.isfinite(Z).any() else np.inf
    return out, nearest


def process(st, L, R, a, scale=None):
    """Rectify + disparity + depth + obstacles for one frame pair, using calibration state."""
    h, w = L.shape[:2]
    rL = cv2.warpPerspective(L, st["H1"], (w, h))
    rR = cv2.warpPerspective(R, st["H2"], (w, h))
    gL, gR = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (rL, rR))
    scale = scale or getattr(a, "disp_scale", 1.0) or 1.0
    d = clean_disparity(disparity(gL, gR, st["min_d"], a.num_disp or st["num_d"], a.block, scale), st["min_d"])
    Z = depth_from_disparity(d + st["offset"], a.focal, a.baseline, a.max_depth)
    ann, nearest = obstacles(Z, rL, a.danger, a.warn)
    return dict(rL=rL, rR=rR, disp=d, Z=Z, ann=ann, nearest=nearest)


def colorize(x, lo, hi):
    v = (np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    c = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    c[~np.isfinite(x)] = 0
    return c


# --------------------------------------------------------------------------
# Static mode: left/right image files (or the demo) -> full showcase
# --------------------------------------------------------------------------
def run_static(a):
    if a.left and a.right:
        L, R = cv2.imread(a.left), cv2.imread(a.right)
        if L is None or R is None:
            raise SystemExit("Could not read the left/right image files.")
    else:
        print("No images given -> using built-in demo scene (focal=500 px, baseline=0.12 m)")
        L, R = make_demo_pair()
        a.focal = a.focal or 500.0
        a.baseline = a.baseline or 0.12
    L = cv2.resize(L, (a.width, int(L.shape[0] * a.width / L.shape[1])))
    R = cv2.resize(R, L.shape[1::-1])
    a.focal = a.focal or 0.85 * L.shape[1]
    a.baseline = a.baseline or 0.10
    gL, gR = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (L, R))

    print("[1-2] Matching features and computing F with the 8-point algorithm + RANSAC ...")
    st = calibrate_pair(gL, gR)
    showcase(L, R, a, st)


def showcase(L, R, a, st):
    """Full report for one image pair: F, epipoles, epipolar lines, rectification,
    disparity, depth, 3D reconstruction and obstacles. Used by file mode AND live mode (key C / R)."""
    import matplotlib
    matplotlib.use("Agg")                                        # figure is drawn off-screen and shown through OpenCV
    import matplotlib.pyplot as plt
    F = st["F"]
    if st["n_inliers"]:
        err = np.sqrt(epipolar_error(F, st["p1"], st["p2"]).mean())
        print("      %d / %d matches kept, mean epipolar error %.3f px" % (st["n_inliers"], st["n_matches"], err))
    else:
        print("      no feature matches -> using the camera's ideal factory-rectified F")
    print("      F =\n", np.array2string(F, precision=6, suppress_small=True))

    print("[3]   Epipolar geometry")
    eL, eR = epipoles(F)
    print("      left epipole :", describe_epipole(eL))
    print("      right epipole:", describe_epipole(eR))
    for e in (eL, eR):
        if abs(e[2]) > 1e-4 * np.linalg.norm(e[:2]) and abs(e[0] / e[2]) < 3 * L.shape[1] and abs(e[1] / e[2]) < 3 * L.shape[0]:
            print("      HINT: an epipole is inside/near the image -> cameras are not parallel (tilted or toed-in) "
                  "or the matches are poor. Re-mount them side by side, parallel, and recalibrate.")
            break
    if st.get("note"):
        print("      WARNING:", st["note"])
    if len(st["p1"]):
        sel = np.random.default_rng(0).choice(len(st["p1"]), min(12, len(st["p1"])), replace=False)
        p1, p2 = st["p1"][sel], st["p2"][sel]
        linesR = (np.c_[p1, np.ones(len(p1))] @ F.T)             # line in right image for each left point
        linesL = (np.c_[p2, np.ones(len(p2))] @ F)               # line in left image for each right point
        epiL, epiR = draw_epilines(L, linesL, p1), draw_epilines(R, linesR, p2)
    else:                                                        # no matches: nothing to draw lines through
        epiL, epiR = L.copy(), R.copy()
        for im in (epiL, epiR):
            cv2.putText(im, "no matches (plain scene / IR projector off)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    print("[4-8] Rectifying, computing disparity, depth, 3D cloud and obstacles ...")
    res = process(st, L, R, a, scale=1.0)                        # the report always uses full resolution
    h_, w_ = L.shape[:2]
    cx, cy = (getattr(a, "cx", None) or w_ / 2), (getattr(a, "cy", None) or h_ / 2)
    pts, rgb = to_cloud(res["Z"], res["rL"], a.focal, a.max_depth, cx, cy)
    cam = (a.focal, cx, cy, w_, h_)
    print("      3D points: %d" % len(pts))
    print("      nearest object: %.2f m   (danger < %.1f m, warn < %.1f m)" % (res["nearest"], a.danger, a.warn))

    os.makedirs(a.save_dir, exist_ok=True)
    save_ply(os.path.join(a.save_dir, "cloud.ply"), pts, rgb)
    rgb_ = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pair = np.hstack([res["rL"], res["rR"]])
    for y in range(20, pair.shape[0], 40):
        cv2.line(pair, (0, y), (pair.shape[1], y), (0, 255, 0), 1)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("1. Epipolar lines - LEFT", rgb_(epiL), None), ("1. Epipolar lines - RIGHT", rgb_(epiR), None),
              ("2. Rectified pair (green rows align)", rgb_(pair), None),
              ("3. Disparity (px)", res["disp"], "jet"), ("4. Depth (m)", res["Z"], "jet_r"),
              ("5. Obstacle detection", rgb_(res["ann"]), None)]
    for axi, (title, im, cmap) in zip(ax.ravel(), panels):
        h = axi.imshow(im, cmap=cmap)
        axi.set_title(title)
        axi.axis("off")
        if cmap:
            fig.colorbar(h, ax=axi, fraction=0.046)
    fig.suptitle("Stereo vision  |  F-matrix inliers: %d  |  nearest object: %.2f m" % (st["n_inliers"], res["nearest"]))
    info = "Fundamental matrix F:\n" + np.array2string(F, precision=5, suppress_small=True) + \
           "\nleft epipole: " + describe_epipole(eL) + "\nright epipole: " + describe_epipole(eR)
    fig.text(0.02, 0.01, info, family="monospace", fontsize=8, va="bottom")
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(os.path.join(a.save_dir, "overview.png"), dpi=110)

    fig.canvas.draw()
    overview = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close("all")
    if len(pts) < 100:
        print("Too few valid 3D points to show - check focal/baseline and that the scene has texture.")
        return
    views = []                                                   # front / side / top picture, saved next to the cloud
    for name, yaw, pitch in [("front (= left camera view)", 0, 0), ("side", 1.0, 0), ("top", 0, -1.1)]:
        v = render_cloud(pts, rgb, cam, yaw, pitch, scale=1.0, splat=1 if yaw == pitch == 0 else 2)
        cv2.putText(v, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        views.append(v)
    cv2.imwrite(os.path.join(a.save_dir, "reconstruction3d.png"), np.hstack(views))
    print("Saved overview.png, reconstruction3d.png, cloud.ply in '%s/'" % a.save_dir)
    if not a.no_show:                                            # windows stay responsive; Q closes the 3D viewer
        cv2.imshow("Overview", cv2.resize(overview, (1280, int(overview.shape[0] * 1280 / overview.shape[1]))))
        view_cloud(pts, rgb, cam, a.max_points)
        _close_window("Overview")


# --------------------------------------------------------------------------
# Live mode: two cameras -> real-time distance to obstacles
#   keys:  c = calibrate + full report   r = report again   e = epipolar lines   l = RealSense IR projector   s = save point cloud   q = quit
# --------------------------------------------------------------------------
class CameraPair:
    """Two ordinary cameras: USB index numbers or phone stream URLs."""
    def __init__(self, a):
        src = [int(c) if c.isdigit() else c for c in a.cams]     # number = USB camera, text = stream URL
        self.capL, self.capR = cv2.VideoCapture(src[0]), cv2.VideoCapture(src[1])
        for c in (self.capL, self.capR):
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)                    # less lag on network streams
        if not (self.capL.isOpened() and self.capR.isOpened()):
            raise SystemExit("Could not open cameras %s and %s (USB: try --cams 1 2, phones: check the URLs)." % tuple(a.cams))

    def read(self):
        okL, okR = self.capL.grab(), self.capR.grab()            # grab both first -> better sync
        if not (okL and okR):
            return False, None, None
        (okL, L), (okR, R) = self.capL.retrieve(), self.capR.retrieve()
        return okL and okR, L, R

    def release(self):
        self.capL.release(); self.capR.release()


class RealSensePair:
    """Intel RealSense D415: reads ONLY the left/right infrared imagers (factory-calibrated and rectified).
    The camera's own depth stream is never enabled - depth is computed by this project."""
    def __init__(self, a):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise SystemExit("pyrealsense2 is not installed. Run:  python -m pip install pyrealsense2")
        self.rs = rs
        w, h = (int(v) for v in a.rs_size.lower().split("x"))
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, a.rs_fps)   # 1 = left imager
        cfg.enable_stream(rs.stream.infrared, 2, w, h, rs.format.y8, a.rs_fps)   # 2 = right imager
        self.pipe = rs.pipeline()
        try:
            prof = self.pipe.start(cfg)
        except RuntimeError as e:
            raise SystemExit("Could not start the RealSense camera (%s).\nUse a USB 3 port, close RealSense Viewer, "
                             "and check --rs-size (D415: 640x480, 848x480, 1280x720)." % e)
        self.sensor = prof.get_device().first_depth_sensor()
        i1, i2 = prof.get_stream(rs.stream.infrared, 1), prof.get_stream(rs.stream.infrared, 2)
        self.width = w
        intr = i1.as_video_stream_profile().get_intrinsics()
        self.fx, self.ppx, self.ppy = intr.fx, intr.ppx, intr.ppy           # focal length + principal point (factory)
        self.baseline = abs(i1.get_extrinsics_to(i2).translation[0])        # metres (factory)
        self.emitter = not a.no_emitter
        self._apply_emitter()
        for _ in range(15):                                                 # let auto-exposure settle
            try:
                self.pipe.wait_for_frames(5000)
            except RuntimeError:
                break
        print("RealSense ready: %dx%d, factory focal %.1f px, baseline %.1f mm" % (w, h, self.fx, self.baseline * 1000))

    def _apply_emitter(self):
        if self.sensor.supports(self.rs.option.emitter_enabled):
            self.sensor.set_option(self.rs.option.emitter_enabled, 1.0 if self.emitter else 0.0)

    def toggle_emitter(self):
        self.emitter = not self.emitter
        self._apply_emitter()
        print("IR dot projector:", "ON" if self.emitter else "OFF")

    def read(self):
        try:
            fs = self.pipe.wait_for_frames(5000)
        except RuntimeError:
            return False, None, None
        L, R = (cv2.cvtColor(np.asanyarray(fs.get_infrared_frame(i).get_data()), cv2.COLOR_GRAY2BGR) for i in (1, 2))
        return True, L, R

    def release(self):
        self.pipe.stop()


def run_live(a):
    cams = RealSensePair(a) if a.realsense else CameraPair(a)
    win = "Stereo obstacle detection"
    cv2.namedWindow(win)
    cv2.createTrackbar("danger cm", win, int(a.danger * 100), 500, lambda v: None)
    cv2.createTrackbar("warn cm", win, int(a.warn * 100), 800, lambda v: None)
    cv2.createTrackbar("block", win, a.block, 21, lambda v: None)
    cv2.createTrackbar("max disp x16", win, a.num_disp // 16 if a.num_disp else 0, 16, lambda v: None)
    if a.realsense:                                              # use the factory calibration of the camera
        a.focal = a.focal or cams.fx * a.width / cams.width
        a.cx, a.cy = cams.ppx * a.width / cams.width, cams.ppy * a.width / cams.width
        a.baseline = a.baseline or cams.baseline
    a.focal = a.focal or 0.85 * a.width
    a.baseline = a.baseline or 0.10
    a.disp_scale = a.disp_scale or 0.5                           # live: half-resolution matching for speed
    st, res, show_epi = None, None, False
    while True:
        ok, L, R = cams.read()
        if not ok:
            break
        L = cv2.resize(L, (a.width, int(L.shape[0] * a.width / L.shape[1])))
        if not a.realsense and R.shape[0] * L.shape[1] != L.shape[0] * R.shape[1] and not getattr(a, "_warned", False):
            print("WARNING: the two phones stream different aspect ratios - set the SAME resolution in both apps.")
            a._warned = True
        R = cv2.resize(R, L.shape[1::-1])
        a.danger = max(cv2.getTrackbarPos("danger cm", win), 10) / 100
        a.warn = max(cv2.getTrackbarPos("warn cm", win) / 100, a.danger)
        a.block = cv2.getTrackbarPos("block", win)
        a.num_disp = cv2.getTrackbarPos("max disp x16", win) * 16
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            try:
                st = calibrate_pair(*(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (L, R)), prerectified=a.realsense)
                if a.realsense:                                  # search range from the closest distance we care about
                    st["min_d"] = 0
                    st["num_d"] = int(np.clip(np.ceil(a.focal * a.baseline / a.min_depth / 16) * 16, 32, 256))
                print("Calibrated: %d inliers, disparity range %d..%d" % (st["n_inliers"], st["min_d"], st["min_d"] + st["num_d"]))
                showcase(L, R, a, st)                            # F, epipoles, epilines, rectified, disparity, depth, 3D
            except RuntimeError as e:
                print("Calibration failed:", e)
        if key == ord("r") and st is not None:
            showcase(L, R, a, st)
        if key == ord("e"):
            show_epi = not show_epi
        if key == ord("l") and a.realsense:
            cams.toggle_emitter()
        if st is None:
            view = np.hstack([L, R])
            cv2.putText(view, "Aim at a textured scene and press C to calibrate" + ("  (L = IR projector)" if a.realsense else ""), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            res = process(st, L, R, a)
            left = res["ann"]
            if show_epi and len(st["p1"]) >= 10:
                p1 = st["p1"][:10]
                left = draw_epilines(L, np.c_[st["p2"][:10], np.ones(10)] @ st["F"], p1)
            d = res["disp"]
            view = np.hstack([left, colorize(d, st["min_d"], st["min_d"] + (a.num_disp or st["num_d"]))])
            txt = "nearest: %.2f m" % res["nearest"] if np.isfinite(res["nearest"]) else "nearest: --"
            cv2.putText(view, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if res["nearest"] < a.danger else (0, 255, 0), 2)
        cv2.imshow(win, view)
        if key == ord("s") and res is not None:
            os.makedirs(a.save_dir, exist_ok=True)
            pts, rgb = to_cloud(res["Z"], res["rL"], a.focal, a.max_depth, getattr(a, "cx", None), getattr(a, "cy", None))
            save_ply(os.path.join(a.save_dir, "live_cloud.ply"), pts, rgb)
            print("Saved", os.path.join(a.save_dir, "live_cloud.ply"))
    cams.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Stereo vision: F-matrix -> rectification -> disparity -> depth -> 3D -> obstacles")
    p.add_argument("--left", help="left image file");  p.add_argument("--right", help="right image file")
    p.add_argument("--live", action="store_true", help="use two live cameras")
    p.add_argument("--cams", nargs=2, default=["0", "1"], help="USB camera numbers or phone stream URLs (left right)")
    p.add_argument("--realsense", action="store_true", help="use an Intel RealSense (D415) left/right infrared pair")
    p.add_argument("--rs-size", default="640x480", help="RealSense stream size, e.g. 640x480, 848x480, 1280x720")
    p.add_argument("--rs-fps", type=int, default=30)
    p.add_argument("--emitter", action="store_true", help="RealSense: IR dot projector ON (this is the default now)")
    p.add_argument("--no-emitter", action="store_true", help="RealSense: start with the IR dot projector OFF (key L toggles)")
    p.add_argument("--min-depth", type=float, default=0.4, help="RealSense: closest distance to measure in metres (sets disparity range)")
    p.add_argument("--focal", type=float, default=0, help="focal length in pixels (default ~0.85*width)")
    p.add_argument("--baseline", type=float, default=0, help="distance between cameras in metres")
    p.add_argument("--danger", type=float, default=1.5, help="STOP distance in metres")
    p.add_argument("--warn", type=float, default=2.5, help="WARN distance in metres")
    p.add_argument("--max-depth", type=float, default=10.0, help="ignore points farther than this (m)")
    p.add_argument("--num-disp", type=int, default=0, help="disparity search range (0 = auto)")
    p.add_argument("--block", type=int, default=5, help="matching window size (odd)")
    p.add_argument("--width", type=int, default=640, help="resize images to this width")
    p.add_argument("--disp-scale", type=float, default=0, help="matching resolution: 1 = full (sharper), 0.5 = 8x faster (default for live)")
    p.add_argument("--max-points", type=int, default=200000, help="max points drawn in the 3D viewer")
    p.add_argument("--save-dir", default="output")
    p.add_argument("--no-show", action="store_true", help="save results without opening windows")
    a = p.parse_args()
    run_live(a) if (a.live or a.realsense) else run_static(a)


if __name__ == "__main__":
    main()
