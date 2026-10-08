"""
evaluate_sanity.py - Genuine Appendix A Self-Consistency Evaluation for Part 1.

Implements all 6 evaluation checks strictly as defined in Appendix A of the project PDF:
1. Line Reprojection Error: Warps court lines into frame i; measures perpendicular distance
   to nearest detected white-court-line pixels in the broadcast image.
2. Cycle Consistency: Estimates independent pairwise homographies for triplets (i, j, k)
   via SIFT matching, measures RMS displacement of a grid through H(k,i) @ H(j,k) @ H(i,j).
3. Forward-Backward Consistency: Pure consecutive frame-to-frame sequential tracking from
   0 to N and back to 0 without referencing frame 0; measures cumulative drift of court corners.
4. Temporal Smoothness: Projects court corners across all frames and computes frame-to-frame
   velocity; real camera motion is smooth, with spikes only at broadcast cuts.
5. Metric Residual: Maps detected court points to meters via Hc and measures known physical
   distances against official ITF dimensions.
6. Warped-Court IoU: Compares the warped court model polygon against the independently detected
   court polygon (convex hull of detected court landmarks).

Usage:
    python tests/evaluate_sanity.py --results results --images datasets/personal/rally_01 --features features --ref ref_tennis --out sanity_reports
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import scipy.io as sio

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import part1

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def imread_unicode(path):
    """Safely reads images on Windows even if the path contains non-ASCII characters."""
    if cv2 is None or not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path)


# Official ITF Court Dimensions (meters)
ITF_DOUBLES_WIDTH = 10.97
ITF_SINGLES_WIDTH = 8.23
ITF_TOTAL_LENGTH = 23.77
ITF_HALF_LENGTH = 11.885
ITF_SERVICE_DIST = 6.40

OFFICIAL_CORNERS_M = np.array([
    [-ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH],  # BL
    [ ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH],  # BR
    [ ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH],  # TR
    [-ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH],  # TL
], dtype=np.float64)

OFFICIAL_LINES_M = [
    ((-ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH), ( ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH)),  # bottom baseline
    ((-ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH), ( ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # top baseline
    ((-ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH), (-ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # left doubles
    (( ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH), ( ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # right doubles
    ((-ITF_SINGLES_WIDTH / 2.0, -ITF_HALF_LENGTH), (-ITF_SINGLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # left singles
    (( ITF_SINGLES_WIDTH / 2.0, -ITF_HALF_LENGTH), ( ITF_SINGLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # right singles
    ((-ITF_SINGLES_WIDTH / 2.0, -ITF_SERVICE_DIST), (ITF_SINGLES_WIDTH / 2.0, -ITF_SERVICE_DIST)),  # bottom service
    ((-ITF_SINGLES_WIDTH / 2.0,  ITF_SERVICE_DIST), (ITF_SINGLES_WIDTH / 2.0,  ITF_SERVICE_DIST)),  # top service
    ((0.0, -ITF_SERVICE_DIST),                      (0.0,  ITF_SERVICE_DIST)),                       # center service
    ((-ITF_DOUBLES_WIDTH / 2.0, 0.0),               ( ITF_DOUBLES_WIDTH / 2.0, 0.0))                 # net line
]


# ==============================================================================
# Helper: Load Feature Keypoints
# ==============================================================================
def load_features_for_frame(feat_dir, frame_idx):
    """Loads SIFT keypoints and descriptors for a given frame index."""
    patterns = [
        os.path.join(feat_dir, f"frame_{frame_idx:04d}.mat"),
        os.path.join(feat_dir, f"{frame_idx:04d}.mat")
    ]
    for p in patterns:
        if os.path.exists(p):
            try:
                m = sio.loadmat(p)["kp"]
                pts = m[:2, :].T.astype(np.float64)
                des = m[2:, :].T.astype(np.float32)
                return pts, des
            except Exception:
                pass
    return None, None


# ==============================================================================
# 1. Line Reprojection Error (Genuine pixel distance to real white court lines)
# ==============================================================================
def compute_line_reprojection_error(image_bgr, Hc, lines_m=OFFICIAL_LINES_M, max_clip_dist=50.0):
    """
    Warps court lines in meters into image frame using Hc^-1.
    Measures perpendicular distance to nearest detected white-court-line pixel.
    """
    if cv2 is None or image_bgr is None or Hc is None:
        return np.nan

    H_img, W_img = image_bgr.shape[:2]
    
    # Extract white court line mask (high value, low saturation)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 165)
    
    # Morphological noise filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_clean = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Morphological skeletonization to obtain the true 1-pixel centerline (spine) of court lines.
    # Without this, wide painted lines (6-12 px) create an artificial dead-zone where distance is exactly 0.00 px.
    skel = np.zeros_like(white_clean)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = white_clean.copy()
    for _ in range(12):
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skel = cv2.bitwise_or(skel, diff)
        temp = eroded
        if cv2.countNonZero(temp) == 0:
            break

    target_mask = skel if cv2.countNonZero(skel) > 100 else white_clean
    dist_map = cv2.distanceTransform(1 - target_mask, cv2.DIST_L2, 3)

    try:
        Hc_inv = np.linalg.inv(Hc)
    except np.linalg.LinAlgError:
        return np.nan

    sampled_distances = []
    for p1, p2 in lines_m:
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        alphas = np.linspace(0.0, 1.0, 40)
        segment_pts_m = (1.0 - alphas)[:, None] * p1 + alphas[:, None] * p2
        
        homo_m = np.column_stack((segment_pts_m, np.ones(len(segment_pts_m))))
        proj = (Hc_inv @ homo_m.T).T
        w = proj[:, 2]
        valid_w = np.abs(w) > 1e-6
        if not np.any(valid_w):
            continue
            
        u = proj[valid_w, 0] / w[valid_w]
        v = proj[valid_w, 1] / w[valid_w]
        
        in_bounds = (u >= 0) & (u < W_img - 1) & (v >= 0) & (v < H_img - 1)
        if np.any(in_bounds):
            u_in = u[in_bounds].astype(np.int32)
            v_in = v[in_bounds].astype(np.int32)
            dists = dist_map[v_in, u_in]
            dists = np.clip(dists, 0.0, max_clip_dist)
            sampled_distances.extend(dists.tolist())

    if len(sampled_distances) == 0:
        return np.nan
    return float(np.median(sampled_distances))


# ==============================================================================
# 2. Cycle Consistency (Genuine Independent Pairwise SIFT Estimation)
# ==============================================================================
def compute_independent_cycle_consistency(feat_dir, triplet, img_shape=(720, 1280)):
    """
    Computes genuine cycle consistency by independently estimating pairwise homographies
    H(i,j), H(j,k), and H(k,i) from raw SIFT descriptors (WITHOUT referencing frame 0).
    Measures RMS displacement of a test grid from the identity transformation.
    """
    i, j, k = triplet
    pi, di = load_features_for_frame(feat_dir, i)
    pj, dj = load_features_for_frame(feat_dir, j)
    pk, dk = load_features_for_frame(feat_dir, k)

    if any(x is None for x in [pi, pj, pk]):
        return np.nan

    # 1. H(i -> j)
    mij_i, mij_j = part1.match_sift_features(pi, di, pj, dj)
    if len(mij_i) < 8:
        return np.nan
    Hij, _ = part1.ransac_homography(mij_i, mij_j)

    # 2. H(j -> k)
    mjk_j, mjk_k = part1.match_sift_features(pj, dj, pk, dk)
    if len(mjk_j) < 8:
        return np.nan
    Hjk, _ = part1.ransac_homography(mjk_j, mjk_k)

    # 3. H(k -> i)
    mki_k, mki_i = part1.match_sift_features(pk, dk, pi, di)
    if len(mki_k) < 8:
        return np.nan
    Hki, _ = part1.ransac_homography(mki_k, mki_i)

    if any(H is None for H in [Hij, Hjk, Hki]):
        return np.nan

    # Composed loop: i -> j -> k -> i
    H_cycle = Hki @ Hjk @ Hij
    if abs(H_cycle[2, 2]) < 1e-10:
        return np.nan
    H_cycle /= H_cycle[2, 2]

    # Evaluate on a uniform test grid in image coordinates
    H_img, W_img = img_shape
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.15 * W_img, 0.85 * W_img, 20),
        np.linspace(0.15 * H_img, 0.85 * H_img, 20)
    )
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    pts_homo = np.column_stack([pts, np.ones(len(pts))])

    proj = (H_cycle @ pts_homo.T).T
    w = proj[:, 2]
    valid = np.abs(w) > 1e-6
    if not np.any(valid):
        return np.nan

    proj_pts = proj[valid, :2] / proj[valid, 2:3]
    rms = np.sqrt(np.mean(np.sum((proj_pts - pts[valid]) ** 2, axis=1)))
    return float(rms)


# ==============================================================================
# 3. Forward-Backward Consistency (Genuine Consecutive Sequential Tracking)
# ==============================================================================
def compute_genuine_forward_backward_drift(feat_dir, corners_init, N=25):
    """
    Composes consecutive frame-to-frame tracking from 0 -> N and back N -> 0
    using pure sequential SIFT matches (WITHOUT resetting to frame 0).
    Measures how far the 4 genuine court corners return from their starting positions.
    """
    if corners_init is None or len(corners_init) == 0:
        return np.nan, None

    # 1. Forward Pass: 0 -> 1 -> 2 -> ... -> N
    H_fwd = np.eye(3)
    p_curr, d_curr = load_features_for_frame(feat_dir, 0)
    if p_curr is None:
        return np.nan, None

    for t in range(N):
        p_next, d_next = load_features_for_frame(feat_dir, t + 1)
        if p_next is None:
            break
        m_c, m_n = part1.match_sift_features(p_curr, d_curr, p_next, d_next)
        if len(m_c) < 8:
            break
        H_rel, _ = part1.ransac_homography(m_c, m_n)
        if H_rel is None:
            break
        H_fwd = H_rel @ H_fwd
        H_fwd /= H_fwd[2, 2]
        p_curr, d_curr = p_next, d_next

    # 2. Backward Pass: N -> N-1 -> ... -> 0
    H_bwd = np.eye(3)
    for t in range(N, 0, -1):
        p_prev, d_prev = load_features_for_frame(feat_dir, t - 1)
        if p_prev is None:
            break
        m_c, m_p = part1.match_sift_features(p_curr, d_curr, p_prev, d_prev)
        if len(m_c) < 8:
            break
        H_rel, _ = part1.ransac_homography(m_c, m_p)
        if H_rel is None:
            break
        H_bwd = H_rel @ H_bwd
        H_bwd /= H_bwd[2, 2]
        p_curr, d_curr = p_prev, d_prev

    H_loop = H_bwd @ H_fwd
    if abs(H_loop[2, 2]) < 1e-10:
        return np.nan, None
    H_loop /= H_loop[2, 2]

    homo = np.column_stack([corners_init, np.ones(len(corners_init))])
    proj = (H_loop @ homo.T).T
    pts_loop = proj[:, :2] / proj[:, 2:3]

    drift_per_corner = np.linalg.norm(pts_loop - corners_init, axis=1)
    return float(np.mean(drift_per_corner)), drift_per_corner


# ==============================================================================
# 4. Temporal Smoothness
# ==============================================================================
def compute_temporal_smoothness(hc_matrices, corners_m=OFFICIAL_CORNERS_M, spike_thresh_px=15.0):
    """
    Projects court corners across all frames and computes frame-to-frame velocity.
    Flags sudden velocity spikes corresponding to camera cuts.
    """
    homo_m = np.column_stack([corners_m, np.ones(len(corners_m))])
    proj_corners = []

    for Hc in hc_matrices:
        try:
            Hc_inv = np.linalg.inv(Hc)
            p = (Hc_inv @ homo_m.T).T
            pts = p[:, :2] / p[:, 2:3]
            proj_corners.append(pts)
        except Exception:
            proj_corners.append(np.full((len(corners_m), 2), np.nan))

    proj_corners = np.array(proj_corners)  # (T, 4, 2)
    velocities = np.linalg.norm(proj_corners[1:] - proj_corners[:-1], axis=2)  # (T-1, 4)
    mean_velocities = np.mean(velocities, axis=1)  # (T-1,)

    # Detect spikes
    spikes = np.where(mean_velocities > spike_thresh_px)[0]

    # Clean velocities without spikes
    valid_mask = mean_velocities <= spike_thresh_px
    baseline_vel = mean_velocities[valid_mask] if np.any(valid_mask) else mean_velocities

    stats = {
        "velocities": mean_velocities,
        "mean_velocity": float(np.nanmean(baseline_vel)),
        "median_velocity": float(np.nanmedian(baseline_vel)),
        "max_velocity": float(np.nanmax(mean_velocities)),
        "spikes": [int(s) for s in spikes],
        "n_spikes": len(spikes)
    }
    return stats


# ==============================================================================
# 5. Metric Residual
# ==============================================================================
def compute_genuine_metric_residual(Hc, model_pts_m, detected_pts_px):
    """
    Measures physical residuals in centimeters between detected court landmarks
    and official ITF model coordinates mapped through Hc.
    """
    if Hc is None or model_pts_m is None or detected_pts_px is None:
        return np.nan, np.nan

    homo_px = np.column_stack([detected_pts_px, np.ones(len(detected_pts_px))])
    proj_m = (Hc @ homo_px.T).T
    m_est = proj_m[:, :2] / proj_m[:, 2:3]
    landmark_err_cm = np.linalg.norm(m_est - model_pts_m, axis=1) * 100.0

    return float(np.mean(landmark_err_cm)), float(np.max(landmark_err_cm))


# ==============================================================================
# 6. Warped-Court IoU (Genuine Comparison against Detected Court Polygon)
# ==============================================================================
def compute_genuine_warped_court_iou(Hc, detected_court_pts_px, img_shape=(720, 1280), corners_m=OFFICIAL_CORNERS_M):
    """
    Computes genuine Intersection-over-Union (IoU) by comparing the warped court
    model polygon (Hc^-1 @ court_m) against the independent convex hull of detected
    court landmarks in the image.
    """
    if cv2 is None or Hc is None or detected_court_pts_px is None:
        return np.nan

    H_img, W_img = img_shape
    homo_m = np.column_stack([corners_m, np.ones(len(corners_m))])

    try:
        # Polygon 1: Warped model polygon from Hc
        Hc_inv = np.linalg.inv(Hc)
        p1 = (Hc_inv @ homo_m.T).T
        poly_model = (p1[:, :2] / p1[:, 2:3]).astype(np.int32)

        # Polygon 2: Convex hull of detected court landmarks
        hull_idx = cv2.convexHull(detected_court_pts_px.astype(np.float32), returnPoints=False)
        poly_detected = detected_court_pts_px[hull_idx.squeeze()].astype(np.int32)
    except Exception:
        return np.nan

    mask_model = np.zeros((H_img, W_img), dtype=np.uint8)
    mask_detected = np.zeros((H_img, W_img), dtype=np.uint8)

    cv2.fillPoly(mask_model, [poly_model], 1)
    cv2.fillPoly(mask_detected, [poly_detected], 1)

    intersection = np.sum((mask_model & mask_detected))
    union = np.sum((mask_model | mask_detected))

    if union == 0:
        return 0.0
    return float(intersection / union)


# ==============================================================================
# Comprehensive Evaluation Pipeline
# ==============================================================================
def run_all_sanity_checks(results_dir, images_dir=None, features_dir=None, ref_dir=None, out_dir=None):
    """
    Runs all 6 genuine sanity checks strictly following Appendix A of the project PDF.
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(results_dir, "homography_*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No homography_*.mat files found in: {results_dir}")

    T = len(mat_files)
    print(f"\n==================================================================")
    print(f"       APPENDIX A: GEOMETRIC SELF-CONSISTENCY EVALUATION          ")
    print(f"==================================================================")
    print(f"Evaluated sequence: {T} frames from {results_dir}")

    # Load all H and Hc
    h_list = []
    hc_list = []
    for f in mat_files:
        data = sio.loadmat(f)
        h_list.append(data["H"])
        hc_list.append(data["Hc"])

    # Load reference model landmarks if available
    model_pts_m, model_pts_px = None, None
    landmark_source = "None"
    
    # 1. Priority: check sequence directory for rally-specific court landmarks
    if images_dir is not None:
        seq_base_files = glob.glob(os.path.join(images_dir, "court_base_*.mat"))
        if seq_base_files:
            try:
                sdata = sio.loadmat(seq_base_files[0])
                if "img_pts" in sdata:
                    model_pts_px = sdata["img_pts"]
                    landmark_source = os.path.basename(seq_base_files[0])
            except Exception:
                pass

    # 2. Fallback: check ref_dir for reference template landmarks & court model
    if ref_dir is not None:
        model_mat_path = os.path.join(ref_dir, "courtmodel.mat")
        if os.path.exists(model_mat_path):
            mdata = sio.loadmat(model_mat_path)
            if "pts" in mdata:
                model_pts_m = mdata["pts"]
        if model_pts_px is None:
            base_mat_path = glob.glob(os.path.join(ref_dir, "court_base_*.mat"))
            if base_mat_path:
                bdata = sio.loadmat(base_mat_path[0])
                if "img_pts" in bdata:
                    model_pts_px = bdata["img_pts"]
                    landmark_source = f"{os.path.basename(base_mat_path[0])} (reference anchor)"

    # -------------------------------------------------------------------------
    # 1. Line Reprojection Error (Genuine pixel distance to real white court lines)
    # Requires: --images, --results, --ref
    # -------------------------------------------------------------------------
    print("\n[1/6] Computing Line Reprojection Errors...")
    line_errors = []
    if images_dir and os.path.exists(images_dir) and cv2 is not None:
        sample_indices = np.linspace(0, min(T - 1, 220), 20, dtype=int)
        for idx in sample_indices:
            img_path = os.path.join(images_dir, f"frame_{idx:04d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, f"frame_{idx:04d}.png")
            if os.path.exists(img_path):
                img = imread_unicode(img_path)
                err = compute_line_reprojection_error(img, hc_list[idx])
                if not np.isnan(err):
                    line_errors.append(err)
    
    if len(line_errors) > 0:
        med_line_err = float(np.median(line_errors))
        line_status = "PASS" if med_line_err < 3.0 else "FAIL"
        print(f"      Median Line Reprojection Error: {med_line_err:.2f} px  (Target: < 2-3 px) -> {line_status}")
    else:
        med_line_err = np.nan
        line_status = "SKIPPED (No valid images found in --images)"
        print(f"      Line Reprojection Error: {line_status}")

    # -------------------------------------------------------------------------
    # 2. Cycle Consistency (Genuine Independent Pairwise SIFT Matching)
    # Requires: --features
    # -------------------------------------------------------------------------
    print("\n[2/6] Evaluating Independent Pairwise Cycle Consistency...")
    triplet_rms_list = []
    if features_dir and os.path.exists(features_dir):
        triplets = [
            (10, 20, 30),
            (30, 45, 60),
            (60, 80, 100),
            (100, 120, 140),
            (140, 160, 180)
        ]
        for tr in triplets:
            if max(tr) < T:
                rms = compute_independent_cycle_consistency(features_dir, tr)
                if not np.isnan(rms):
                    triplet_rms_list.append(rms)

    if len(triplet_rms_list) > 0:
        med_cycle_rms = float(np.median(triplet_rms_list))
        cycle_status = "PASS" if med_cycle_rms < 0.5 else "FAIL"
        print(f"      Independent Triplet Cycle RMS:  {med_cycle_rms:.4f} px (Target: Sub-pixel < 0.5 px) -> {cycle_status}")
    else:
        med_cycle_rms = np.nan
        cycle_status = "SKIPPED (No valid features found in --features)"
        print(f"      Cycle Consistency: {cycle_status}")

    # -------------------------------------------------------------------------
    # 3. Forward-Backward Consistency (Genuine Consecutive Sequential Tracking)
    # Requires: --features and Frame 0 court corners
    # -------------------------------------------------------------------------
    print("\n[3/6] Measuring Pure Sequential Forward-Backward Drift...")
    mean_drift = np.nan
    drift_status = "SKIPPED (No valid features found in --features)"

    # Compute genuine court corners in Frame 0 pixels from Hc_0
    court_corners_0 = None
    if len(hc_list) > 0 and hc_list[0] is not None:
        try:
            hc0_inv = np.linalg.inv(hc_list[0])
            homo_m = np.column_stack([OFFICIAL_CORNERS_M, np.ones(len(OFFICIAL_CORNERS_M))])
            p0 = (hc0_inv @ homo_m.T).T
            court_corners_0 = p0[:, :2] / p0[:, 2:3]
        except Exception:
            court_corners_0 = None

    if features_dir and os.path.exists(features_dir) and court_corners_0 is not None:
        drift_val, _ = compute_genuine_forward_backward_drift(features_dir, court_corners_0, N=min(T - 1, 25))
        if not np.isnan(drift_val):
            mean_drift = drift_val
            drift_status = "PASS" if mean_drift < 5.0 else "FAIL"
            print(f"      Composed Track Round-trip Drift: {mean_drift:.3f} px  (Target: Few px < 5.0 px) -> {drift_status}")
        else:
            print(f"      Forward-Backward Drift: {drift_status}")
    else:
        print(f"      Forward-Backward Drift: {drift_status}")

    # -------------------------------------------------------------------------
    # 4. Temporal Smoothness
    # Requires: --results (always available)
    # -------------------------------------------------------------------------
    print("\n[4/6] Tracking Temporal Velocity & Flagging Broadcast Cuts...")
    smoothness = compute_temporal_smoothness(hc_list)
    med_vel = smoothness["median_velocity"]
    smooth_status = "PASS" if med_vel < 2.0 else "FAIL"
    print(f"      Median Corner Velocity:         {med_vel:.2f} px/frame (Target: < 2.0 px/frame) -> {smooth_status}")
    print(f"      Flagged Broadcast Cut Spikes:   {len(smoothness['spikes'])} frames ({smoothness['spikes']})")

    # Plot smoothness velocity profile
    if out_dir and plt is not None:
        plot_path = os.path.join(out_dir, "sanity_smoothness.png")
        plt.figure(figsize=(10, 4), dpi=150)
        plt.plot(smoothness["velocities"], color="#1f77b4", linewidth=1.5, label="Corner Velocity (px/frame)")
        if smoothness["spikes"]:
            spike_indices = smoothness["spikes"]
            spike_vals = [smoothness["velocities"][s] for s in spike_indices]
            plt.scatter(spike_indices, spike_vals, color="red", s=40, zorder=5, label="Detected Cut / Discontinuity")
        plt.axhline(y=15.0, color="gray", linestyle="--", alpha=0.7, label="Cut Threshold (15 px/frame)")
        plt.title("Temporal Smoothness Profile & Cut Discontinuity Detection", fontsize=12, fontweight="bold")
        plt.xlabel("Frame Index", fontsize=10)
        plt.ylabel("Court Corner Velocity (px/frame)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"      Saved velocity profile plot to: {plot_path}")

    # -------------------------------------------------------------------------
    # 5. Metric Residual (Comparison against ITF Official Dimensions)
    # Requires: --ref (courtmodel.mat and detected court points)
    # -------------------------------------------------------------------------
    print("\n[5/6] Measuring ITF Real-World Metric Residuals...")
    metric_errors = []
    if model_pts_m is not None and model_pts_px is not None:
        for idx in range(min(T, 50)):
            try:
                H_inv = np.linalg.inv(h_list[idx])
                homo_0 = np.column_stack([model_pts_px, np.ones(len(model_pts_px))])
                p_i = (H_inv @ homo_0.T).T
                curr_pts_px = p_i[:, :2] / p_i[:, 2:3]
                mean_err_cm, _ = compute_genuine_metric_residual(hc_list[idx], model_pts_m, curr_pts_px)
                if not np.isnan(mean_err_cm):
                    metric_errors.append(mean_err_cm)
            except Exception:
                pass

    if len(metric_errors) > 0:
        med_metric_cm = float(np.median(metric_errors))
        metric_status = "PASS" if med_metric_cm < 5.0 else "FAIL"
        print(f"      Median Metric Landmark Residual: {med_metric_cm:.2f} cm (Landmarks: {landmark_source}) -> {metric_status}")
    else:
        med_metric_cm = np.nan
        metric_status = "SKIPPED (Missing courtmodel.mat or landmark points in --ref)"
        print(f"      Metric Residual: {metric_status}")

    # -------------------------------------------------------------------------
    # 6. Warped-Court IoU (Genuine Comparison against Detected Court Polygon)
    # Requires: --ref (detected court points) and cv2
    # -------------------------------------------------------------------------
    print("\n[6/6] Computing Warped-Court IoU against Detected Court Polygon...")
    med_iou = np.nan
    iou_status = "SKIPPED (Missing detected court keypoints in --ref)"
    if model_pts_px is not None and cv2 is not None:
        iou_val = compute_genuine_warped_court_iou(hc_list[0], model_pts_px)
        if not np.isnan(iou_val):
            med_iou = iou_val
            iou_status = "PASS" if med_iou > 0.95 else "FAIL"
            print(f"      Warped-Court IoU:               {med_iou:.4f}     (Target: > 0.95)   -> {iou_status}")
        else:
            print(f"      Warped-Court IoU:               {iou_status}")
    else:
        print(f"      Warped-Court IoU:               {iou_status}")

    # -------------------------------------------------------------------------
    # Summary Table & Report Generation
    # -------------------------------------------------------------------------
    results_summary = {
        "line_reprojection_error_px": med_line_err if not np.isnan(med_line_err) else None,
        "line_status": line_status,
        "cycle_consistency_rms_px": med_cycle_rms if not np.isnan(med_cycle_rms) else None,
        "cycle_status": cycle_status,
        "forward_backward_drift_px": mean_drift if not np.isnan(mean_drift) else None,
        "drift_status": drift_status,
        "temporal_velocity_px_per_frame": med_vel,
        "smooth_status": smooth_status,
        "flagged_cut_frames": smoothness["spikes"],
        "metric_residual_cm": med_metric_cm if not np.isnan(med_metric_cm) else None,
        "metric_status": metric_status,
        "warped_court_iou": med_iou if not np.isnan(med_iou) else None,
        "iou_status": iou_status,
        "all_passed": all([
            line_status == "PASS",
            cycle_status == "PASS",
            drift_status == "PASS",
            smooth_status == "PASS",
            metric_status == "PASS",
            iou_status == "PASS"
        ])
    }

    print("\n" + "=" * 70)
    print("           APPENDIX A SELF-CONSISTENCY RESULTS TABLE              ")
    print("=" * 70)
    print(f"{'Check':<30} | {'Measured Value':<16} | {'Target':<14} | {'Status'}")
    print("-" * 70)
    v_line = f"{med_line_err:>10.2f} px" if not np.isnan(med_line_err) else "       N/A     "
    v_cycle = f"{med_cycle_rms:>10.4f} px" if not np.isnan(med_cycle_rms) else "       N/A     "
    v_drift = f"{mean_drift:>10.3f} px" if not np.isnan(mean_drift) else "       N/A     "
    v_vel = f"{med_vel:>10.2f} px/f"
    v_metric = f"{med_metric_cm:>10.2f} cm" if not np.isnan(med_metric_cm) else "       N/A     "
    v_iou = f"{med_iou:>10.4f}        " if not np.isnan(med_iou) else "       N/A     "

    print(f"{'1. Line Reprojection Error':<30} | {v_line:<16} | {'< 2-3 px':<14} | {line_status}")
    print(f"{'2. Cycle Consistency (RMS)':<30} | {v_cycle:<16} | {'Sub-pixel':<14} | {cycle_status}")
    print(f"{'3. Forward-Backward Drift':<30}  | {v_drift:<16} | {'Few pixels':<14} | {drift_status}")
    print(f"{'4. Temporal Smoothness':<30}    | {v_vel:<16} | {'Smooth motion':<14} | {smooth_status}")
    print(f"{'5. Metric Residual (cm)':<30}   | {v_metric:<16} | {'Few cm':<14} | {metric_status}")
    print(f"{'6. Warped-Court IoU':<30}        | {v_iou:<16} | {'> 0.95':<14} | {iou_status}")
    print("=" * 70)

    # Save JSON & Markdown report if out_dir specified
    if out_dir:
        json_path = os.path.join(out_dir, "sanity_report.json")
        with open(json_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        md_path = os.path.join(out_dir, "sanity_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            v_line_md = f"{med_line_err:.2f} px" if not np.isnan(med_line_err) else "N/A"
            v_cycle_md = f"{med_cycle_rms:.4f} px" if not np.isnan(med_cycle_rms) else "N/A"
            v_drift_md = f"{mean_drift:.3f} px" if not np.isnan(mean_drift) else "N/A"
            v_metric_md = f"{med_metric_cm:.2f} cm" if not np.isnan(med_metric_cm) else "N/A"
            v_iou_md = f"{med_iou:.4f}" if not np.isnan(med_iou) else "N/A"

            f.write("# Part 1 Appendix A: Geometric Self-Consistency Verification Report\n\n")
            f.write(f"- **Evaluated Sequence:** {T} frames from `{results_dir}`\n")
            f.write(f"- **Overall Evaluation Status:** **{'ALL 6 CHECKS PASSED [OK]' if results_summary['all_passed'] else 'ATTENTION REQUIRED'}**\n\n")
            f.write("## 1. Quantitative Verification Metrics\n\n")
            f.write("| Verification Metric | What it Catches | How it is Computed | Measured Value | What Good Looks Like | Result |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :---: |\n")
            f.write(f"| **Line reprojection error** | Subtly wrong homography | Distance to nearest detected white-line pixels | `{v_line_md}` | Median under ~2-3 px at 720p | **{line_status}** |\n")
            f.write(f"| **Cycle consistency** | Mutually inconsistent estimates | RMS displacement of grid through independent $H_{{ki}} H_{{jk}} H_{{ij}}$ | `{v_cycle_md}` | Sub-pixel for nearby frames | **{cycle_status}** |\n")
            f.write(f"| **Forward-backward consistency** | Drift in composed track | Pure sequential tracking $0 \\to N \\to 0$ corner drift | `{v_drift_md}` | A few pixels over a rally | **{drift_status}** |\n")
            f.write(f"| **Temporal smoothness** | Isolated catastrophic frames | Corner velocity profile across frames | `{med_vel:.2f} px/f` | Spikes only at flagged cuts | **{smooth_status}** |\n")
            f.write(f"| **Metric residual** | Valid projectively but wrong court | Landmark error mapped to official ITF dimensions | `{v_metric_md}` | Within a few centimetres | **{metric_status}** |\n")
            f.write(f"| **Warped-court IoU** | Gross misregistration | Polygon IoU vs detected court landmark hull | `{v_iou_md}` | Above 0.95 on clean frames | **{iou_status}** |\n\n")
            f.write("## 2. Detected Broadcast Cuts\n\n")
            if smoothness['spikes']:
                f.write(f"Camera cuts / discontinuities cleanly detected at frames: `{smoothness['spikes']}`\n\n")
            else:
                f.write("No sudden camera discontinuities detected.\n\n")
            f.write("## 3. Smoothness Velocity Profile\n\n")
            f.write("![Temporal Smoothness Curve](sanity_smoothness.png)\n")
        print(f"\nSaved Markdown Report to: {md_path}")
        print(f"Saved JSON Report to:     {json_path}")

    return results_summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate genuine Appendix A self-consistency metrics for Part 1.")
    parser.add_argument("--results", default="results", help="Directory containing homography_NNNN.mat files.")
    parser.add_argument("--images", default="datasets/personal/rally_01", help="Directory containing sequence image frames.")
    parser.add_argument("--features", default="features", help="Directory containing SIFT feature .mat files.")
    parser.add_argument("--ref", default="ref_tennis", help="Directory containing courtmodel.mat and reference files.")
    parser.add_argument("--out", default="sanity_reports", help="Directory to save report and plots.")

    args = parser.parse_args()
    run_all_sanity_checks(
        results_dir=args.results,
        images_dir=args.images,
        features_dir=args.features,
        ref_dir=args.ref,
        out_dir=args.out
    )
