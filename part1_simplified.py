"""
part1.py - Planar tracking and homography estimation (PIV Project, Part 1).

Only numpy / scipy are used (no cv2). DLT and RANSAC are written out by hand.

Outputs, one file per frame:  homography_NNNN.mat  ->  {"H": H, "Hc": Hc}
    H  : frame NNNN -> reference image (templateimg.jpg), pixels
    Hc : frame NNNN -> court model, metres      (Hc = H_ref_to_court @ H)
plus tracking_log.json (method used per frame, so cuts can be checked later).

Tracking strategy (hybrid with re-anchoring):
    * sequential : match frame i to the last good frame, compose with its H
                   (small baseline, many matches, but drifts)
    * direct     : match frame i to the reference (no drift, fails when the
                   camera is far from the reference)
    * if both work, the direct estimate re-anchors the track unless it
      disagrees wildly with the sequential one (repetitive-line mismatch)
    * if neither works (cut, close-up, crowd) the last valid H is propagated
"""

import os
import re
import glob
import json
from itertools import combinations

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
from scipy.optimize import least_squares


# ------------------------------------------------------------------------------
# Tunable parameters
# ------------------------------------------------------------------------------
RATIO_THRESH = 0.75         # Lowe ratio test
RANSAC_THRESH = 3.5         # inlier threshold (px, symmetric transfer error)
MIN_MATCHES = 12            # matches needed before even running RANSAC
MIN_INLIERS_DIRECT = 15     # inliers needed to accept a direct estimate
MIN_INLIERS_SEQ = 12        # inliers needed to accept a sequential estimate
MIN_SPAN = (0.30, 0.20)     # inliers must spread over this fraction of the frame
                            # (rejects scoreboard / logo clusters)
AREA_RANGE = (0.05, 20.0)   # allowed warped-frame area / frame area
GATE_PX = 60.0              # max disagreement direct vs sequential (px in reference)
STRONG_INLIERS = 80         # a direct estimate this strong is trusted regardless


# ------------------------------------------------------------------------------
# 1. DLT with Hartley normalisation
# ------------------------------------------------------------------------------
def normalize_points(pts):
    """
    Hartley normalisation: move the centroid to the origin and scale so the
    mean distance to the origin is sqrt(2).

    Returns (pts_norm, T) with  [x_norm, 1]^T = T [x, 1]^T.
    """
    pts = np.asarray(pts, dtype=np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    mean_dist = np.mean(np.sqrt(np.sum(centered ** 2, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-8 else 1.0

    T = np.array([[scale, 0.0,   -scale * mean[0]],
                  [0.0,   scale, -scale * mean[1]],
                  [0.0,   0.0,    1.0]])
    return centered * scale, T


def normalize_scale(H):
    """Fix the arbitrary scale of H (H[2,2] = 1, or unit norm if that is ~0)."""
    if abs(H[2, 2]) > 1e-10:
        return H / H[2, 2]
    n = np.linalg.norm(H)
    return H / n if n > 1e-10 else H


def dlt_homography(pts_src, pts_dst):
    """
    Direct Linear Transform:  pts_dst ~ H pts_src.

    For one correspondence x=(x,y,1) -> x'=(u,v,1), the relation x' ~ H x means
        u = (h1.x) / (h3.x)      v = (h2.x) / (h3.x)
    where h1, h2, h3 are the rows of H. Cross-multiplying gives two equations
    that are linear in the 9 entries of H:
        -x*h11 - y*h12 - h13                    + u*x*h31 + u*y*h32 + u*h33 = 0
                          -x*h21 - y*h22 - h23  + v*x*h31 + v*y*h32 + v*h33 = 0

    Stacking N correspondences gives a (2N x 9) matrix A with A h = 0.
    H has 8 degrees of freedom (the scale is free), so 4 correspondences give
    8 equations and A has rank 8: its 1-D null space is H.  With more than 4
    points the system is over-determined and the right singular vector of the
    smallest singular value is the least-squares solution (min ||A h||, ||h||=1).

    Returns the 3x3 matrix, or None if N < 4 or the SVD fails.
    """
    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_dst = np.asarray(pts_dst, dtype=np.float64)
    N = len(pts_src)
    if N < 4:
        return None

    # 1. normalise both point sets (conditioning)
    src_n, T_src = normalize_points(pts_src)
    dst_n, T_dst = normalize_points(pts_dst)

    # 2. build A (2N x 9)
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i]     = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    # 3. solve A h = 0.  full_matrices=True (default) matters when N = 4:
    #    A is 8x9, and the 9th right-singular vector is the null vector.
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    H_norm = Vt[-1].reshape(3, 3)

    # 4. undo the normalisation:  H = T_dst^-1  H_norm  T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return normalize_scale(H)


# ------------------------------------------------------------------------------
# 2. Projection, errors, validity checks
# ------------------------------------------------------------------------------
def apply_homography(H, pts):
    """Project (N,2) points with H. Points that map to infinity become NaN."""
    pts = np.asarray(pts, dtype=np.float64)
    ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    w = ph[:, 2]
    out = np.full((len(pts), 2), np.nan)
    ok = np.abs(w) > 1e-12
    out[ok] = ph[ok, :2] / w[ok, None]
    return out


def symmetric_transfer_error(H, pts_src, pts_dst):
    """
    Per-correspondence error (pixels):
        sqrt( 0.5 * ( |dst - H src|^2 + |src - H^-1 dst|^2 ) )
    Invalid points get +inf.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(pts_src), np.inf)

    fwd = np.sum((apply_homography(H, pts_src) - pts_dst) ** 2, axis=1)
    bwd = np.sum((apply_homography(H_inv, pts_dst) - pts_src) ** 2, axis=1)
    err = np.sqrt(0.5 * (fwd + bwd))
    err[~np.isfinite(err)] = np.inf
    return err


def is_valid_homography(H):
    """Finite, invertible and not absurdly ill-conditioned."""
    return (H is not None and np.all(np.isfinite(H))
            and abs(np.linalg.det(H)) > 1e-12
            and np.linalg.cond(H) < 1e10)


def is_sample_degenerate(src4, dst4, tol=1e-3):
    """
    True if any 3 of the 4 points are (nearly) collinear in the source or the
    destination. The triangle area is measured relative to the squared spread
    of the 4 points, so the test does not depend on units (pixels or metres).
    """
    for pts in (src4, dst4):
        scale = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        if scale < 1e-9:
            return True
        for i, j, k in combinations(range(4), 3):
            area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
                             - (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0]))
            if area / scale ** 2 < tol:
                return True
    return False


def is_plausible(H, w, h):
    """
    Sanity check for a frame->reference homography: the frame rectangle must
    land on a convex, non-mirrored quadrilateral of reasonable size.
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    if not is_valid_homography(H):
        return False
    ph = np.column_stack([corners, np.ones(4)]) @ H.T
    if not (np.all(ph[:, 2] > 1e-12) or np.all(ph[:, 2] < -1e-12)):
        return False                        # corners on both sides of the horizon
    q = ph[:, :2] / ph[:, 2:3]

    # convexity: all consecutive edge cross-products have the same sign
    edges = np.roll(q, -1, axis=0) - q
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] \
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not np.all(cross > 0):               # > 0 also rules out mirroring
        return False

    area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    ratio = area / (w * h)
    return AREA_RANGE[0] <= ratio <= AREA_RANGE[1]


# ------------------------------------------------------------------------------
# 3. Non-linear refinement
# ------------------------------------------------------------------------------
def refine_homography(H0, pts_src, pts_dst):
    """
    Levenberg-Marquardt on the symmetric transfer error. The residual vector
    holds the x and y components of both the forward and backward errors
    (4N values), so the cost is smooth. Used on RANSAC inliers only.
    The result is kept only if the mean error actually decreases.
    """
    if len(pts_src) < 4 or abs(H0[2, 2]) < 1e-10:
        return H0
    H0 = normalize_scale(H0)

    def residuals(p):
        H = np.append(p, 1.0).reshape(3, 3)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.full(4 * len(pts_src), 1e3)
        r = np.concatenate([(apply_homography(H, pts_src) - pts_dst).ravel(),
                            (apply_homography(H_inv, pts_dst) - pts_src).ravel()])
        r[~np.isfinite(r)] = 1e3
        return r

    try:
        res = least_squares(residuals, H0.ravel()[:8], method="lm",
                            x_scale="jac", max_nfev=200)
    except Exception:
        return H0

    H = np.append(res.x, 1.0).reshape(3, 3)
    if not is_valid_homography(H):
        return H0
    before = np.mean(symmetric_transfer_error(H0, pts_src, pts_dst))
    after = np.mean(symmetric_transfer_error(H, pts_src, pts_dst))
    return H if after < before else H0


# ------------------------------------------------------------------------------
# 4. RANSAC
# ------------------------------------------------------------------------------
def ransac_homography(pts_src, pts_dst, max_iters=2000, threshold=3.0,
                      min_inliers=10, confidence=0.995, rng=None):
    """
    RANSAC for a homography.

    Loop:
        1. draw a minimal sample of 4 correspondences
        2. reject it if 3 of the points are collinear
        3. fit H with the DLT
        4. score H: inliers are the points with symmetric transfer error < threshold
        5. keep the model with the most inliers (ties: smaller total error)
        6. shrink the number of iterations needed:
               N = log(1 - confidence) / log(1 - w^4),  w = inlier ratio so far
    Finally H is re-estimated with the DLT on all inliers, refined with LM, and
    the inliers are recomputed with the final H.

    Returns (H, inlier_mask). H is None if fewer than `min_inliers` inliers
    were found.
    """
    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_dst = np.asarray(pts_dst, dtype=np.float64)
    N = len(pts_src)
    no_inliers = np.zeros(N, dtype=bool)
    if N < max(4, min_inliers):
        return None, no_inliers

    if rng is None:
        rng = np.random.default_rng(42)          # fixed seed: reproducible runs

    best_mask, best_n, best_cost = no_inliers, 0, np.inf
    n_iters = max_iters
    it = 0
    while it < n_iters:
        it += 1

        idx = rng.choice(N, size=4, replace=False)          # 1. sample
        if is_sample_degenerate(pts_src[idx], pts_dst[idx]):  # 2. degeneracy
            continue
        H = dlt_homography(pts_src[idx], pts_dst[idx])        # 3. fit
        if not is_valid_homography(H):
            continue

        err = symmetric_transfer_error(H, pts_src, pts_dst)   # 4. score
        mask = err < threshold
        n_in = int(mask.sum())
        cost = float(err[mask].sum())

        if n_in > best_n or (n_in == best_n and cost < best_cost):   # 5. keep best
            best_mask, best_n, best_cost = mask, n_in, cost

            w = n_in / N                                            # 6. adapt
            if w ** 4 >= 1.0:
                n_iters = min(n_iters, it)
            elif w ** 4 > 1e-12:
                needed = np.log(1.0 - confidence) / np.log(1.0 - w ** 4)
                n_iters = min(n_iters, int(np.ceil(needed)))

    if best_n < min_inliers:
        return None, best_mask

    # final least-squares fit on all inliers, then non-linear refinement
    src_in, dst_in = pts_src[best_mask], pts_dst[best_mask]
    H = dlt_homography(src_in, dst_in)
    if not is_valid_homography(H):
        return None, best_mask
    H = refine_homography(H, src_in, dst_in)

    mask = symmetric_transfer_error(H, pts_src, pts_dst) < threshold
    if mask.sum() < min_inliers:
        return None, mask
    return H, mask


# ------------------------------------------------------------------------------
# 5. Features: loading and matching
# ------------------------------------------------------------------------------
def load_feature_mat(mat_path):
    """
    Load 'kp' of shape (130, N) from a feature .mat file:
    row 0 = x, row 1 = y, rows 2..129 = SIFT descriptor.
    Returns pts (N,2) and L2-normalised descriptors (N,128).
    """
    data = sio.loadmat(mat_path)
    if "kp" not in data:
        raise KeyError(f"{mat_path} has no 'kp' entry")
    kp = data["kp"]
    if kp.shape[0] != 130 and kp.shape[1] == 130:
        kp = kp.T
    pts = kp[:2, :].T.astype(np.float64)
    des = kp[2:, :].T.astype(np.float32)
    norms = np.linalg.norm(des, axis=1, keepdims=True)
    norms[norms < 1e-7] = 1.0
    return pts, des / norms


def match_sift_features(pts1, des1, pts2, des2, ratio_thresh=RATIO_THRESH,
                        mutual_check=True):
    """
    Nearest-neighbour matching with Lowe's ratio test and an optional mutual
    (cross) check, using scipy's cKDTree. Returns the matched points of both sets.
    """
    N1, N2 = len(des1), len(des2)
    if N1 < 4 or N2 < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    dist, idx = cKDTree(des2).query(des1, k=2, workers=-1)
    keep = dist[:, 0] < ratio_thresh * dist[:, 1]            # ratio test

    if mutual_check:                                         # 2->1 must agree
        _, back = cKDTree(des1).query(des2, k=1, workers=-1)
        keep &= back[idx[:, 0]] == np.arange(N1)

    i1 = np.where(keep)[0]
    return pts1[i1], pts2[idx[i1, 0]]


# ------------------------------------------------------------------------------
# 6. Court anchor
# ------------------------------------------------------------------------------
def _as_points(a):
    a = np.asarray(a, dtype=np.float64)
    return a.T if a.shape[0] == 2 and a.shape[1] != 2 else a


def load_court_anchor(refdir):
    """
    H_ref_to_court (reference-image pixels -> court metres), by DLT on landmarks.

    Expects in refdir:
        courtmodel.mat    {"pts": (N,2) metres, "names": ...}
        court_base*.mat   {"img_pts": (N,2) pixel positions of the same landmarks
                           in templateimg.jpg}; an optional "court_pts" key
                           overrides the model points (for a subset of landmarks)
    Raises if the anchor cannot be built: a silent identity would make Hc garbage.
    """
    cm_path = os.path.join(refdir, "courtmodel.mat")
    if not os.path.exists(cm_path):
        raise FileNotFoundError(f"{cm_path} not found")
    court_pts = _as_points(sio.loadmat(cm_path)["pts"])

    base_files = sorted(glob.glob(os.path.join(refdir, "court_base*.mat")))
    if not base_files:
        raise FileNotFoundError(
            f"No court_base*.mat (with 'img_pts') in {refdir}: the anchor needs "
            "the pixel positions of the court landmarks in the reference image.")
    base = sio.loadmat(base_files[0])
    img_pts = _as_points(base["img_pts"])
    if "court_pts" in base:
        court_pts = _as_points(base["court_pts"])

    if img_pts.shape != court_pts.shape or len(img_pts) < 4:
        raise ValueError(f"Need >= 4 matching landmark pairs, got image {img_pts.shape} "
                         f"vs court {court_pts.shape}")

    H = dlt_homography(img_pts, court_pts)
    if not is_valid_homography(H):
        raise ValueError("Court anchor DLT produced a degenerate homography")

    rms_cm = 100 * np.sqrt(np.mean(np.sum((apply_homography(H, img_pts) - court_pts) ** 2, axis=1)))
    print(f"[part1] Anchor from {os.path.basename(base_files[0])}: "
          f"{len(img_pts)} landmarks, RMS residual {rms_cm:.1f} cm")
    return H


# ------------------------------------------------------------------------------
# 7. Tracking helpers
# ------------------------------------------------------------------------------
def estimate_pair(cur, ref, extent, min_inliers):
    """
    Homography cur -> ref from SIFT matches + RANSAC.
    Returns (H, n_inliers), or (None, 0) if matches are too few, RANSAC fails,
    or the inliers are bunched in one place (scoreboard / logo).
    """
    (c_pts, c_des), (r_pts, r_des) = cur, ref
    if c_des is None or r_des is None:
        return None, 0
    m_c, m_r = match_sift_features(c_pts, c_des, r_pts, r_des)
    if len(m_c) < MIN_MATCHES:
        return None, 0
    H, inl = ransac_homography(m_c, m_r, threshold=RANSAC_THRESH, min_inliers=min_inliers)
    if H is None:
        return None, 0
    span = np.ptp(m_c[inl], axis=0) / extent
    if span[0] < MIN_SPAN[0] or span[1] < MIN_SPAN[1]:
        return None, 0
    return H, int(inl.sum())


def disagreement(Ha, Hb, w, h):
    """Mean distance (reference pixels) between two homographies on 5 frame points."""
    g = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], dtype=np.float64)
    return float(np.mean(np.linalg.norm(apply_homography(Ha, g) - apply_homography(Hb, g), axis=1)))


def frame_geometry(ref_pts):
    """Approximate frame size and keypoint extent from the reference keypoints."""
    w, h = float(ref_pts[:, 0].max()), float(ref_pts[:, 1].max())
    extent = np.maximum(np.ptp(ref_pts, axis=0), 100.0)
    return max(w, 1.0), max(h, 1.0), extent


def list_frames(images_dir):
    """[(number, number_string, name_without_ext)] sorted by frame number."""
    frames = []
    for f in glob.glob(os.path.join(images_dir, "*.jpg")):
        name = os.path.splitext(os.path.basename(f))[0]
        m = re.search(r"(\d+)$", name)
        if m:
            frames.append((int(m.group(1)), m.group(1), name))
    return sorted(frames)


def try_load(path):
    if not os.path.exists(path):
        return None, None
    try:
        return load_feature_mat(path)
    except Exception as e:
        print(f"[part1] could not read {path}: {e}")
        return None, None


# ------------------------------------------------------------------------------
# 8. Entry point
# ------------------------------------------------------------------------------
def part1(path1, path2, path3, path4):
    """part1(path_to_refdir, path_images_dir, path_feature_dir, path_output_dir)"""
    refdir, images_dir, feat_dir, out_dir = path1, path2, path3, path4
    os.makedirs(out_dir, exist_ok=True)

    H_ref_to_court = load_court_anchor(refdir)

    frames = list_frames(images_dir)
    if not frames:
        raise FileNotFoundError(f"No NAME_NNNN.jpg images in {images_dir}")
    print(f"[part1] {len(frames)} frames")

    # reference features (templateimg.mat); if absent, the first frame is the reference
    ref = (None, None)
    for d in (feat_dir, refdir):
        ref = try_load(os.path.join(d, "templateimg.mat"))
        if ref[0] is not None:
            break
    geometry = frame_geometry(ref[0]) if ref[0] is not None else None

    prev = None                       # features of the last frame with a valid H
    H_prev = np.eye(3)                # its H
    last_valid_H = np.eye(3)
    log = []

    for idx, (num, num_str, name) in enumerate(frames):
        cur = try_load(os.path.join(feat_dir, f"{name}.mat"))
        has_feat = cur[1] is not None

        if ref[0] is None and has_feat:                   # first frame becomes the reference
            ref = cur
            geometry = frame_geometry(ref[0])
            print(f"[part1] reference initialised from frame {num_str}")

        method, n_in, H = "fallback", 0, None

        if has_feat and ref[0] is not None:
            w, h, extent = geometry

            # --- sequential: last good frame -> current, composed with its H
            H_seq = None
            if prev is not None:
                H_rel, n_seq = estimate_pair(cur, prev, extent, MIN_INLIERS_SEQ)
                if H_rel is not None:
                    cand = normalize_scale(H_prev @ H_rel)
                    if is_plausible(cand, w, h):
                        H_seq = cand

            # --- direct: current -> reference
            H_dir, n_dir = estimate_pair(cur, ref, extent, MIN_INLIERS_DIRECT)
            if H_dir is not None and not is_plausible(H_dir, w, h):
                H_dir = None

            # --- choose (direct re-anchors, unless it contradicts the track)
            if H_dir is not None and H_seq is not None:
                if n_dir >= STRONG_INLIERS or disagreement(H_dir, H_seq, w, h) <= GATE_PX:
                    H, method, n_in = H_dir, "direct", n_dir
                else:
                    H, method, n_in = H_seq, "sequential", n_seq
            elif H_dir is not None:
                H, method, n_in = H_dir, "direct", n_dir
            elif H_seq is not None:
                H, method, n_in = H_seq, "sequential", n_seq

        if H is None:                                     # cut / close-up / crowd
            H = last_valid_H.copy()
        else:
            last_valid_H = H.copy()
            prev, H_prev = cur, H.copy()

        Hc = normalize_scale(H_ref_to_court @ H)
        sio.savemat(os.path.join(out_dir, f"homography_{num_str}.mat"),
                    {"H": H.astype(np.float64), "Hc": Hc.astype(np.float64)})
        log.append({"frame": num, "method": method, "inliers": n_in})

        if (idx + 1) % 50 == 0 or idx + 1 == len(frames):
            print(f"[part1] {idx + 1}/{len(frames)}  frame {num_str}: {method} ({n_in} inliers)")

    with open(os.path.join(out_dir, "tracking_log.json"), "w") as f:
        json.dump(log, f)

    counts = {m: sum(1 for e in log if e["method"] == m) for m in ("direct", "sequential", "fallback")}
    print(f"[part1] done: {counts}  -> {out_dir}")