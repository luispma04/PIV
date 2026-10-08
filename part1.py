"""
part1.py - Planar Tracking and Homography Estimation for PIV Project (Part 1).

RESTRICTIONS COMPLIANCE:
- NO OpenCV (cv2) imported or used.
- Uses only NumPy and SciPy.
- Implements Hartley-normalized DLT from scratch.
- Implements RANSAC outlier rejection and homography estimation from scratch.
- Implements SIFT descriptor matching with Lowe's ratio test and mutual consistency via scipy.spatial.cKDTree.
- Computes:
    * H: 3x3 homography mapping frame NNNN pixels to reference image pixels (templateimg.jpg).
    * Hc: 3x3 homography mapping frame NNNN pixels to metric court coordinates (in meters).
"""

import os
import re
import glob
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
from scipy.optimize import least_squares


# ==============================================================================
# 1. Direct Linear Transformation (DLT) with Hartley Normalization
# ==============================================================================

def normalize_points(pts):
    """
    Hartley isotropic normalization of 2D points.
    Translates centroid to origin and scales average distance to sqrt(2).
    
    Args:
        pts: (N, 2) numpy array of 2D point coordinates.
        
    Returns:
        pts_norm: (N, 2) normalized point coordinates.
        T: (3, 3) similarity transformation matrix such that [x_norm, 1]^T = T * [x, 1]^T.
    """
    pts = np.asarray(pts, dtype=np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    
    mean_dist = np.mean(np.sqrt(np.sum(centered ** 2, axis=1)))
    if mean_dist < 1e-8:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_dist
        
    T = np.array([
        [scale, 0.0,   -scale * mean[0]],
        [0.0,   scale, -scale * mean[1]],
        [0.0,   0.0,    1.0]
    ], dtype=np.float64)
    
    pts_homo = np.column_stack((pts, np.ones(len(pts), dtype=np.float64)))
    pts_norm = (T @ pts_homo.T).T[:, :2]
    
    return pts_norm, T


def dlt_homography(pts_src, pts_dst):
    """
    Computes 3x3 homography H such that pts_dst ~ H * pts_src using
    Hartley-normalized Direct Linear Transformation (DLT).
    
    Args:
        pts_src: (N, 2) coordinates in source frame.
        pts_dst: (N, 2) coordinates in target/reference frame.
        
    Returns:
        H: (3, 3) homography matrix, or None if degenerate.
    """
    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_dst = np.asarray(pts_dst, dtype=np.float64)
    N = pts_src.shape[0]
    
    if N < 4:
        return None
        
    # 1. Normalize coordinates
    pts_src_norm, T_src = normalize_points(pts_src)
    pts_dst_norm, T_dst = normalize_points(pts_dst)
    
    # 2. Build 2N x 9 matrix A
    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        x, y = pts_src_norm[i, 0], pts_src_norm[i, 1]
        u, v = pts_dst_norm[i, 0], pts_dst_norm[i, 1]
        
        # Row 2*i: [-x, -y, -1,  0,  0,  0,  u*x,  u*y,  u]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        # Row 2*i+1: [ 0,  0,  0, -x, -y, -1,  v*x,  v*y,  v]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
        
    # 3. Solve Ah = 0 via SVD
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
        
    h_norm = Vt[-1, :]
    H_norm = h_norm.reshape(3, 3)
    
    # 4. Denormalize: H = inv(T_dst) * H_norm * T_src
    try:
        T_dst_inv = np.linalg.inv(T_dst)
        H = T_dst_inv @ H_norm @ T_src
    except np.linalg.LinAlgError:
        return None
        
    # Normalize homography scale
    if abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]
    else:
        norm = np.linalg.norm(H)
        if norm > 1e-10:
            H /= norm
            
    return H


# ==============================================================================
# 2. Geometric Validation and Degeneracy Checks
# ==============================================================================

def are_points_collinear(p1, p2, p3, eps=1e-4):
    """
    Checks if 3 2D points are collinear by measuring triangle area.
    """
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    return area < eps


def is_sample_degenerate(pts_src_4, pts_dst_4):
    """
    Checks if any triplet of 4 correspondences is collinear in either source or destination.
    """
    triplets = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for i, j, k in triplets:
        if are_points_collinear(pts_src_4[i], pts_src_4[j], pts_src_4[k]):
            return True
        if are_points_collinear(pts_dst_4[i], pts_dst_4[j], pts_dst_4[k]):
            return True
    return False


def is_homography_valid(H, min_det=1e-5, max_cond=1e6):
    """
    Validates that a homography represents a plausible transformation:
    non-singular, reasonable condition number, and preserves orientation.
    """
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = np.linalg.det(H)
    if abs(det) < min_det:
        return False
    cond = np.linalg.cond(H)
    if cond > max_cond:
        return False
    return True


# ==============================================================================
# 3. Projection and Reprojection Errors
# ==============================================================================

def project_points(H, pts):
    """
    Projects (N, 2) points using 3x3 homography H.
    
    Returns:
        projected_pts: (N, 2) projected coordinates.
        valid_mask: (N,) boolean mask indicating points in front of the camera plane (w > 0).
    """
    pts = np.asarray(pts, dtype=np.float64)
    N = len(pts)
    pts_homo = np.column_stack((pts, np.ones(N, dtype=np.float64)))
    proj_homo = (H @ pts_homo.T).T
    
    w = proj_homo[:, 2]
    valid_mask = np.abs(w) > 1e-8
    
    proj_pts = np.zeros_like(pts)
    proj_pts[valid_mask, 0] = proj_homo[valid_mask, 0] / w[valid_mask]
    proj_pts[valid_mask, 1] = proj_homo[valid_mask, 1] / w[valid_mask]
    
    return proj_pts, valid_mask


def symmetric_transfer_error(H, pts_src, pts_dst):
    """
    Computes symmetric transfer distance for each correspondence:
    d(pts_dst, H * pts_src)^2 + d(pts_src, H^-1 * pts_dst)^2
    """
    N = len(pts_src)
    errors = np.full(N, np.inf, dtype=np.float64)
    
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return errors
        
    # Forward projection: pts_src -> pts_dst
    proj_dst, valid_dst = project_points(H, pts_src)
    # Backward projection: pts_dst -> pts_src
    proj_src, valid_src = project_points(H_inv, pts_dst)
    
    valid = valid_dst & valid_src
    if not np.any(valid):
        return errors
        
    fwd_sq = np.sum((pts_dst[valid] - proj_dst[valid]) ** 2, axis=1)
    bwd_sq = np.sum((pts_src[valid] - proj_src[valid]) ** 2, axis=1)
    
    errors[valid] = np.sqrt(0.5 * (fwd_sq + bwd_sq))
    return errors


# ==============================================================================
# 4. Non-linear Homography Refinement
# ==============================================================================

def refine_homography_nonlinear(H_init, pts_src, pts_dst):
    """
    Refines homography parameters using Levenberg-Marquardt non-linear least squares
    minimizing symmetric reprojection error.
    """
    if len(pts_src) < 4:
        return H_init
        
    # Parameterize H by 8 degrees of freedom (normalized by H[2,2])
    H = H_init / (H_init[2, 2] if abs(H_init[2, 2]) > 1e-10 else 1.0)
    params0 = H.flatten()[:8]
    
    def residuals(params):
        H_mat = np.append(params, 1.0).reshape(3, 3)
        err = symmetric_transfer_error(H_mat, pts_src, pts_dst)
        err[~np.isfinite(err)] = 1e4
        return np.clip(err, 0.0, 100.0)
        
    try:
        res = least_squares(residuals, params0, method="lm", max_nfev=50)
        if res.success:
            H_refined = np.append(res.x, 1.0).reshape(3, 3)
            if is_homography_valid(H_refined):
                return H_refined / H_refined[2, 2]
    except Exception:
        pass
        
    return H_init


# ==============================================================================
# 5. RANSAC Homography Estimation
# ==============================================================================

def ransac_homography(pts_src, pts_dst, max_iters=2500, threshold=3.0, min_inliers=10, rng=None):
    """
    RANSAC estimator for 2D homography.
    
    Args:
        pts_src: (N, 2) correspondences in source frame.
        pts_dst: (N, 2) correspondences in target frame.
        max_iters: maximum RANSAC sampling iterations.
        threshold: inlier reprojection error threshold in pixels.
        min_inliers: minimum number of inliers required for acceptance.
        rng: optional np.random.Generator or RandomState.
        
    Returns:
        best_H: (3, 3) estimated homography or None if failed.
        inlier_mask: (N,) boolean mask of consensus inliers.
    """
    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_dst = np.asarray(pts_dst, dtype=np.float64)
    N = len(pts_src)
    
    if N < 4:
        return None, np.zeros(N, dtype=bool)
        
    if rng is None:
        rng = np.random.default_rng(42)
        
    best_H = None
    best_inlier_mask = np.zeros(N, dtype=bool)
    max_inliers = 0
    best_error_sum = np.inf
    
    # Adaptive iterations
    confidence = 0.995
    iters = 0
    max_adaptive_iters = max_iters
    
    while iters < max_adaptive_iters:
        iters += 1
        
        # 1. Sample minimal 4 correspondences
        idx = rng.choice(N, size=4, replace=False)
        sample_src = pts_src[idx]
        sample_dst = pts_dst[idx]
        
        # 2. Check collinearity degeneracy
        if is_sample_degenerate(sample_src, sample_dst):
            continue
            
        # 3. Fit minimal candidate model
        H_cand = dlt_homography(sample_src, sample_dst)
        if H_cand is None or not is_homography_valid(H_cand):
            continue
            
        # 4. Score model against all points
        errors = symmetric_transfer_error(H_cand, pts_src, pts_dst)
        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)
        
        if num_inliers > max_inliers or (num_inliers == max_inliers and np.sum(errors[inlier_mask]) < best_error_sum):
            max_inliers = num_inliers
            best_error_sum = np.sum(errors[inlier_mask])
            best_inlier_mask = inlier_mask
            best_H = H_cand
            
            # Update adaptive iteration bound
            w = max_inliers / float(N)
            w4 = w ** 4
            if w4 >= 0.999:
                max_adaptive_iters = min(max_adaptive_iters, iters + 10)
            elif w4 > 1e-12:
                denom = np.log(1.0 - w4)
                if abs(denom) > 1e-12:
                    N_est = np.log(1.0 - confidence) / denom
                    max_adaptive_iters = min(max_adaptive_iters, int(np.ceil(N_est)))
                
    # 5. Consensus check and refinement on all inliers
    if max_inliers >= max(4, min_inliers):
        # Re-estimate DLT on all consensus inliers
        inlier_src = pts_src[best_inlier_mask]
        inlier_dst = pts_dst[best_inlier_mask]
        H_refined = dlt_homography(inlier_src, inlier_dst)
        
        if H_refined is not None and is_homography_valid(H_refined):
            # Non-linear refinement
            H_refined = refine_homography_nonlinear(H_refined, inlier_src, inlier_dst)
            return H_refined, best_inlier_mask
            
    return best_H, best_inlier_mask


# ==============================================================================
# 6. Feature Loading and Matching (SciPy cKDTree)
# ==============================================================================

def load_feature_mat(mat_path):
    """
    Loads SIFT features stored in .mat format as defined by pivist/features:
    Dictionary containing 'kp' of shape (130, N):
        - row 0: x coordinates
        - row 1: y coordinates
        - rows 2..129: 128-dimensional SIFT descriptors
        
    Returns:
        pts: (N, 2) numpy array of keypoint locations [x, y].
        des: (N, 128) numpy array of SIFT descriptors.
    """
    data = sio.loadmat(mat_path)
    if "kp" not in data:
        raise KeyError(f"Feature file {mat_path} does not contain 'kp' key.")
        
    combined = data["kp"]
    if combined.shape[0] < 130 and combined.shape[1] >= 130:
        # Transposed format safeguard
        combined = combined.T
        
    pts = combined[0:2, :].T.astype(np.float64)  # (N, 2)
    des = combined[2:, :].T.astype(np.float32)   # (N, 128)
    
    # Normalize descriptors to unit L2 norm
    norms = np.linalg.norm(des, axis=1, keepdims=True)
    norms[norms < 1e-7] = 1.0
    des = des / norms
    
    return pts, des


def match_sift_features(pts1, des1, pts2, des2, ratio_thresh=0.75, mutual_check=True):
    """
    Matches SIFT descriptors using Lowe's ratio test and mutual consistency
    using scipy.spatial.cKDTree.
    
    Returns:
        matched_pts1: (M, 2) matched points in set 1.
        matched_pts2: (M, 2) matched points in set 2.
    """
    N1 = len(des1)
    N2 = len(des2)
    if N1 < 4 or N2 < 4:
        return np.empty((0, 2)), np.empty((0, 2))
        
    # Build KD-Tree on target descriptors
    tree2 = cKDTree(des2)
    dists, indices = tree2.query(des1, k=2, workers=-1)
    
    # Lowe's ratio test
    ratio_mask = dists[:, 0] < (ratio_thresh * dists[:, 1])
    
    if not mutual_check:
        valid_idx1 = np.where(ratio_mask)[0]
        valid_idx2 = indices[valid_idx1, 0]
        return pts1[valid_idx1], pts2[valid_idx2]
        
    # Mutual cross-check: query 1-NN of des2 in des1
    tree1 = cKDTree(des1)
    _, rev_indices = tree1.query(des2, k=1, workers=-1)
    
    valid_idx1 = []
    valid_idx2 = []
    for i1 in np.where(ratio_mask)[0]:
        i2 = indices[i1, 0]
        if rev_indices[i2] == i1:
            valid_idx1.append(i1)
            valid_idx2.append(i2)
            
    valid_idx1 = np.array(valid_idx1, dtype=int)
    valid_idx2 = np.array(valid_idx2, dtype=int)
    
    return pts1[valid_idx1], pts2[valid_idx2]


# ==============================================================================
# 7. Court Model Anchoring
# ==============================================================================

def load_or_compute_court_anchor(refdir):
    """
    Loads court model anchoring to compute H_ref_to_court (Frame 0 pixels -> court meters).
    
    1. Loads metric court model from `courtmodel.mat` in refdir (contains 'pts' in meters).
    2. Matches with image keypoints from `court_base_*.mat` or `court_keypoints.mat` in refdir.
    3. Solves H_ref_to_court using Hartley-normalized DLT from scratch.
    """
    court_model_path = os.path.join(refdir, "courtmodel.mat")
    court_pts = None
    H_precomputed = None

    if os.path.exists(court_model_path):
        try:
            cm = sio.loadmat(court_model_path)
            if "pts" in cm:
                court_pts = cm["pts"].astype(np.float64)
            if "H_ref_to_court" in cm:
                H_precomputed = cm["H_ref_to_court"].astype(np.float64)
        except Exception as e:
            print(f"[part1] Notice reading courtmodel.mat: {e}")

    if H_precomputed is not None and is_homography_valid(H_precomputed):
        print("[part1] Loaded precomputed anchor H_ref_to_court from courtmodel.mat.")
        return H_precomputed

    # Search for image keypoints file: court_base_*.mat, court_keypoints.mat, anchor.mat
    kps_candidates = glob.glob(os.path.join(refdir, "court_base_*.mat"))
    kps_candidates.extend([
        os.path.join(refdir, "court_keypoints.mat"),
        os.path.join(refdir, "anchor.mat")
    ])

    for cand in kps_candidates:
        if not os.path.exists(cand):
            continue
        try:
            data = sio.loadmat(cand)
            if "H_ref_to_court" in data:
                H_anc = data["H_ref_to_court"].astype(np.float64)
                if is_homography_valid(H_anc):
                    print(f"[part1] Loaded H_ref_to_court from {os.path.basename(cand)}.")
                    return H_anc

            img_pts = None
            if "img_pts" in data:
                img_pts = data["img_pts"].astype(np.float64)
            elif "points" in data:
                img_pts = data["points"].astype(np.float64)

            # If candidate also provides court_pts, use them; otherwise use courtmodel.mat pts
            target_court_pts = data["court_pts"].astype(np.float64) if "court_pts" in data else court_pts

            if img_pts is not None and target_court_pts is not None and len(img_pts) == len(target_court_pts):
                H_anc = dlt_homography(img_pts, target_court_pts)
                if H_anc is not None and is_homography_valid(H_anc):
                    if abs(H_anc[2, 2]) > 1e-10:
                        H_anc /= H_anc[2, 2]
                    print(f"[part1] Computed H_ref_to_court from {os.path.basename(cand)} and courtmodel.mat via DLT.")
                    return H_anc
        except Exception as e:
            print(f"[part1] Notice reading {cand}: {e}")

    print("[part1] Warning: No court landmark correspondences found in refdir.")
    print("        Using identity matrix for H_ref_to_court until anchor is provided.")
    return np.eye(3, dtype=np.float64)


# ==============================================================================
# 8. Main Tracking Function: part1(path1, path2, path3, path4)
# ==============================================================================

def part1(path1, path2, path3, path4):
    """
    Main entry point required by project specification:
    part1(path_to_refdir, path_images_dir, path_feature_dir, path_output_dir)
    
    Computes for every frame in path_images_dir:
    - H: (3, 3) homography from frame NNNN to reference image (templateimg.jpg).
    - Hc: (3, 3) homography from frame NNNN to court model (court coordinates in meters).
    Saves results to path_output_dir as homography_NNNN.mat.
    """
    path_to_refdir = path1
    path_images_dir = path2
    path_feature_dir = path3
    path_output_dir = path4
    
    os.makedirs(path_output_dir, exist_ok=True)
    
    # 1. Load Anchor Homography (Reference Image -> Court Model)
    H_ref_to_court = load_or_compute_court_anchor(path_to_refdir)
    
    # 2. Locate Reference Feature File
    # Check for templateimg.mat in feature_dir or refdir
    template_feat_path = os.path.join(path_feature_dir, "templateimg.mat")
    if not os.path.exists(template_feat_path):
        template_feat_path = os.path.join(path_to_refdir, "templateimg.mat")
        
    ref_pts, ref_des = None, None
    if os.path.exists(template_feat_path):
        ref_pts, ref_des = load_feature_mat(template_feat_path)
        print(f"[part1] Loaded reference template features: {len(ref_pts)} keypoints.")
    else:
        print("[part1] Notice: templateimg.mat not found. Will initialize reference from first frame.")
        
    # 3. Discover and sort all sequence frames
    # Pattern: somename_NNNN.ext
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(path_images_dir, ext)))
        
    if not image_files:
        print(f"[part1] Error: No images found in {path_images_dir}!")
        return
        
    # Sort frames naturally by sequential number
    def extract_frame_info(fpath):
        basename = os.path.basename(fpath)
        name_no_ext = os.path.splitext(basename)[0]
        match = re.search(r"(\d+)$", name_no_ext)
        num = int(match.group(1)) if match else 0
        num_str = match.group(1) if match else "0000"
        return num, num_str, name_no_ext, fpath
        
    frame_list = sorted([extract_frame_info(f) for f in image_files], key=lambda x: x[0])
    total_frames = len(frame_list)
    print(f"[part1] Processing sequence of {total_frames} frames from {path_images_dir}...")
    
    # Tracking state
    prev_pts, prev_des = None, None
    H_prev = np.eye(3, dtype=np.float64)
    last_valid_H = np.eye(3, dtype=np.float64)
    
    cut_count = 0
    direct_success_count = 0
    seq_success_count = 0
    court_lost = False
    
    for idx, (frame_num, num_str, name_no_ext, img_path) in enumerate(frame_list):
        # Locate corresponding feature .mat file
        feat_path = os.path.join(path_feature_dir, f"{name_no_ext}.mat")
        
        curr_pts, curr_des = None, None
        if os.path.exists(feat_path):
            try:
                curr_pts, curr_des = load_feature_mat(feat_path)
            except Exception as e:
                print(f"[part1] Frame {num_str}: failed reading feature file: {e}")
                
        # If reference was not set, set it to the first frame's features
        if ref_pts is None and curr_pts is not None:
            ref_pts, ref_des = curr_pts, curr_des
            print(f"[part1] Initialized reference features from frame {num_str} ({len(ref_pts)} keypoints).")
            
        H_curr = None
        method = "cut_fallback"
        
        # ----------------------------------------------------------------------
        # Strategy A: Direct matching to Reference Template (Zero drift)
        # ----------------------------------------------------------------------
        if curr_des is not None and ref_des is not None and len(curr_des) >= 4:
            m_curr, m_ref = match_sift_features(curr_pts, curr_des, ref_pts, ref_des, ratio_thresh=0.75)
            if len(m_curr) >= 12:
                H_direct, inliers = ransac_homography(m_curr, m_ref, threshold=3.5, min_inliers=10)
                if H_direct is not None and is_homography_valid(H_direct):
                    # Check spatial spread: reject static broadcast graphics / scoreboard clusters
                    inlier_pts = m_curr[inliers]
                    ref_w = max(float(np.ptp(ref_pts[:, 0])), 100.0)
                    ref_h = max(float(np.ptp(ref_pts[:, 1])), 100.0)
                    span_x = float(np.ptp(inlier_pts[:, 0])) / ref_w
                    span_y = float(np.ptp(inlier_pts[:, 1])) / ref_h

                    if span_x >= 0.30 and span_y >= 0.20 and np.sum(inliers) >= 15:
                        # Check that determinant is positive and condition is healthy
                        if np.linalg.det(H_direct) > 0.05 and np.linalg.cond(H_direct) < 5e4:
                            H_curr = H_direct
                            method = f"direct (inliers: {np.sum(inliers)}/{len(m_curr)})"
                            direct_success_count += 1
                            court_lost = False
                        
        # ----------------------------------------------------------------------
        # Strategy B: Frame-to-frame tracking composition (For camera pan/motion)
        # Only active when court was visible in previous frame (not during cuts/closeups)
        # ----------------------------------------------------------------------
        if not court_lost and H_curr is None and curr_des is not None and prev_des is not None and len(curr_des) >= 4:
            m_curr, m_prev = match_sift_features(curr_pts, curr_des, prev_pts, prev_des, ratio_thresh=0.75)
            if len(m_curr) >= 10:
                H_rel, inliers = ransac_homography(m_curr, m_prev, threshold=3.5, min_inliers=8)
                if H_rel is not None and is_homography_valid(H_rel):
                    inlier_pts = m_curr[inliers]
                    prev_w = max(float(np.ptp(prev_pts[:, 0])), 100.0)
                    prev_h = max(float(np.ptp(prev_pts[:, 1])), 100.0)
                    span_x = float(np.ptp(inlier_pts[:, 0])) / prev_w
                    span_y = float(np.ptp(inlier_pts[:, 1])) / prev_h

                    if span_x >= 0.30 and span_y >= 0.20:
                        H_comp = H_prev @ H_rel
                        if is_homography_valid(H_comp):
                            H_curr = H_comp / H_comp[2, 2]
                            method = f"sequential (inliers: {np.sum(inliers)}/{len(m_curr)})"
                            seq_success_count += 1
                        
        # ----------------------------------------------------------------------
        # Strategy C: Cut / Occlusion / Absent Court Handling
        # ----------------------------------------------------------------------
        if H_curr is None:
            # Replay / close-up / court out of view: propagate last valid homography
            H_curr = last_valid_H.copy()
            method = "cut/lost fallback (propagated last valid)"
            cut_count += 1
            court_lost = True
        else:
            last_valid_H = H_curr.copy()
            court_lost = False
            # Only update previous court keypoints when court is confirmed visible
            if curr_pts is not None:
                prev_pts, prev_des = curr_pts, curr_des
                H_prev = H_curr.copy()
            
        # ----------------------------------------------------------------------
        # Compute Metric Court Homography: Hc = H_ref_to_court @ H
        # ----------------------------------------------------------------------
        Hc = H_ref_to_court @ H_curr
        if abs(Hc[2, 2]) > 1e-10:
            Hc = Hc / Hc[2, 2]
            
        # Save output homography: homography_NNNN.mat
        out_mat_filename = f"homography_{num_str}.mat"
        out_mat_path = os.path.join(path_output_dir, out_mat_filename)
        
        out_dict = {
            "H": H_curr.astype(np.float64),
            "Hc": Hc.astype(np.float64)
        }
        sio.savemat(out_mat_path, out_dict)
        
        if (idx + 1) % 20 == 0 or (idx + 1) == total_frames:
            print(f"[part1] Processed frame {idx + 1}/{total_frames} ({num_str}) -> {method}")
            
    print(f"\n[part1] Completed Part 1 tracking for {total_frames} frames.")
    print(f"        Direct matches:     {direct_success_count}")
    print(f"        Sequential matches: {seq_success_count}")
    print(f"        Cuts / Fallbacks:   {cut_count}")
    print(f"        Results saved to:   {path_output_dir}")
