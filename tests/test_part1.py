"""
test_part1.py - Automated Tests for Part 1 Implementation.

Tests:
1. Compliance check: ensure part1.py does NOT import cv2.
2. Hartley normalization test.
3. DLT homography estimation accuracy on synthetic clean points.
4. RANSAC robustness with 40% random outliers.
5. SIFT descriptor matching with Lowe's ratio test and mutual consistency.
6. End-to-end pipeline test on synthetic image sequence.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import scipy.io as sio

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import part1


def test_no_cv2_in_part1():
    """Verify that part1.py does NOT import cv2 anywhere."""
    part1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "part1.py"))
    with open(part1_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Check for cv2 import
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if line_clean.startswith("import cv2") or line_clean.startswith("from cv2"):
            raise AssertionError(f"cv2 is imported in part1.py at line {idx + 1}: {line}")
            
    print("[PASS] test_no_cv2_in_part1: part1.py does not import cv2.")


def test_hartley_normalization():
    """Verify Hartley point normalization: mean is (0,0) and mean distance is sqrt(2)."""
    pts = np.array([
        [100.0, 200.0],
        [300.0, 200.0],
        [300.0, 500.0],
        [100.0, 500.0],
        [200.0, 350.0]
    ])
    pts_norm, T = part1.normalize_points(pts)
    
    mean = np.mean(pts_norm, axis=0)
    assert np.allclose(mean, [0.0, 0.0], atol=1e-10), f"Mean not zero: {mean}"
    
    mean_dist = np.mean(np.sqrt(np.sum(pts_norm ** 2, axis=1)))
    assert np.isclose(mean_dist, np.sqrt(2.0), atol=1e-10), f"Mean dist not sqrt(2): {mean_dist}"
    print("[PASS] test_hartley_normalization passed.")


def test_dlt_synthetic():
    """Verify DLT accurately reconstructs a known homography."""
    # Ground truth homography
    theta = np.radians(15.0)
    c, s = np.cos(theta), np.sin(theta)
    H_gt = np.array([
        [c * 1.2, -s * 0.9, 45.0],
        [s * 1.1,  c * 1.3, -30.0],
        [0.0003,  -0.0002, 1.0]
    ])
    H_gt /= H_gt[2, 2]
    
    pts_src = np.array([
        [50.0, 60.0],
        [400.0, 70.0],
        [420.0, 350.0],
        [60.0, 380.0],
        [200.0, 200.0],
        [150.0, 280.0]
    ])
    
    # Project with H_gt
    pts_homo = np.column_stack((pts_src, np.ones(len(pts_src))))
    proj = (H_gt @ pts_homo.T).T
    pts_dst = proj[:, :2] / proj[:, 2:3]
    
    H_est = part1.dlt_homography(pts_src, pts_dst)
    assert H_est is not None, "DLT failed"
    
    # Verify projected points match
    proj_est = (H_est @ pts_homo.T).T
    pts_dst_est = proj_est[:, :2] / proj_est[:, 2:3]
    
    max_err = np.max(np.linalg.norm(pts_dst - pts_dst_est, axis=1))
    assert max_err < 1e-4, f"DLT reconstruction error too high: {max_err}"
    print(f"[PASS] test_dlt_synthetic passed (max error: {max_err:.2e} px).")


def test_ransac_with_outliers():
    """Verify RANSAC recovers homography when 40% of points are random outliers."""
    rng = np.random.default_rng(123)
    
    H_gt = np.array([
        [1.05, -0.05, 20.0],
        [0.04,  0.98, 15.0],
        [0.0001, 0.0002, 1.0]
    ])
    H_gt /= H_gt[2, 2]
    
    N_inliers = 60
    N_outliers = 40
    N_total = N_inliers + N_outliers
    
    # Inliers
    src_inliers = rng.uniform(50.0, 800.0, size=(N_inliers, 2))
    pts_homo = np.column_stack((src_inliers, np.ones(N_inliers)))
    proj = (H_gt @ pts_homo.T).T
    dst_inliers = proj[:, :2] / proj[:, 2:3]
    # Add small Gaussian noise to inliers
    dst_inliers += rng.normal(0.0, 0.5, size=dst_inliers.shape)
    
    # Outliers
    src_outliers = rng.uniform(50.0, 800.0, size=(N_outliers, 2))
    dst_outliers = rng.uniform(50.0, 800.0, size=(N_outliers, 2))
    
    pts_src = np.vstack([src_inliers, src_outliers])
    pts_dst = np.vstack([dst_inliers, dst_outliers])
    
    H_est, inlier_mask = part1.ransac_homography(pts_src, pts_dst, threshold=3.5, min_inliers=20, rng=rng)
    assert H_est is not None, "RANSAC failed"
    
    detected_inliers = np.sum(inlier_mask[:N_inliers])
    false_positives = np.sum(inlier_mask[N_inliers:])
    
    assert detected_inliers > 0.85 * N_inliers, f"Too few inliers detected: {detected_inliers}/{N_inliers}"
    assert false_positives < 5, f"Too many false positives: {false_positives}"
    print(f"[PASS] test_ransac_with_outliers passed (detected {detected_inliers}/{N_inliers} inliers, {false_positives} FP).")


def test_kd_tree_sift_matching():
    """Verify SIFT matching with Lowe's ratio test and mutual consistency."""
    rng = np.random.default_rng(456)
    N = 50
    # Generate random distinct descriptors
    des1 = rng.normal(0, 1, size=(N, 128)).astype(np.float32)
    des1 /= np.linalg.norm(des1, axis=1, keepdims=True)
    pts1 = rng.uniform(10, 500, size=(N, 2))
    
    # Create des2 as permuted des1 with small noise
    perm = rng.permutation(N)
    des2 = des1[perm] + rng.normal(0, 0.05, size=des1.shape).astype(np.float32)
    des2 /= np.linalg.norm(des2, axis=1, keepdims=True)
    pts2 = pts1[perm] + 10.0
    
    m_pts1, m_pts2 = part1.match_sift_features(pts1, des1, pts2, des2, ratio_thresh=0.8, mutual_check=True)
    assert len(m_pts1) >= 40, f"Expected high match rate, got {len(m_pts1)}/{N}"
    print(f"[PASS] test_kd_tree_sift_matching passed (matched {len(m_pts1)}/{N}).")


def test_full_pipeline_mock():
    """Verify complete end-to-end flow of part1() creating homography_NNNN.mat."""
    tmpdir = tempfile.mkdtemp(prefix="piv_test_")
    try:
        refdir = os.path.join(tmpdir, "ref")
        images_dir = os.path.join(tmpdir, "images")
        features_dir = os.path.join(tmpdir, "features")
        output_dir = os.path.join(tmpdir, "output")
        
        os.makedirs(refdir)
        os.makedirs(images_dir)
        os.makedirs(features_dir)
        
        # 1. Mock courtmodel.mat
        court_pts = np.array([
            [-4.115, -11.885],
            [ 4.115, -11.885],
            [ 4.115,  11.885],
            [-4.115,  11.885]
        ])
        sio.savemat(os.path.join(refdir, "courtmodel.mat"), {
            "pts": court_pts,
            "names": np.array(["BL", "BR", "TR", "TL"], dtype=object)
        })
        
        # 2. Mock anchor.mat
        H_ref_to_court = np.array([
            [0.01, 0.0, -5.0],
            [0.0, 0.01, -10.0],
            [0.0, 0.0, 1.0]
        ])
        sio.savemat(os.path.join(refdir, "anchor.mat"), {
            "H_ref_to_court": H_ref_to_court
        })
        
        # 3. Create mock features for 3 frames
        rng = np.random.default_rng(789)
        base_des = rng.normal(0, 1, size=(50, 128)).astype(np.float32)
        base_pts = rng.uniform(50, 700, size=(50, 2)).astype(np.float32)
        
        for i in range(3):
            fname = f"frame_{i:04d}.jpg"
            # Touch dummy image file
            with open(os.path.join(images_dir, fname), "wb") as f:
                f.write(b"")
                
            # Shift points slightly per frame
            pts_i = base_pts + float(i * 2.0)
            des_i = base_des.copy()
            combined = np.concatenate((pts_i.T, des_i.T), axis=0)  # (130, 50)
            
            sio.savemat(os.path.join(features_dir, f"frame_{i:04d}.mat"), {"kp": combined})
            
        # Also save template features
        template_combined = np.concatenate((base_pts.T, base_des.T), axis=0)
        sio.savemat(os.path.join(features_dir, "templateimg.mat"), {"kp": template_combined})
        
        # Run part1
        part1.part1(refdir, images_dir, features_dir, output_dir)
        
        # Verify output files
        for i in range(3):
            out_file = os.path.join(output_dir, f"homography_{i:04d}.mat")
            assert os.path.exists(out_file), f"Output file missing: {out_file}"
            
            data = sio.loadmat(out_file)
            assert "H" in data, f"'H' missing in {out_file}"
            assert "Hc" in data, f"'Hc' missing in {out_file}"
            
            H = data["H"]
            Hc = data["Hc"]
            assert H.shape == (3, 3), f"H shape wrong: {H.shape}"
            assert Hc.shape == (3, 3), f"Hc shape wrong: {Hc.shape}"
            assert not np.any(np.isnan(H)), f"NaN in H: {H}"
            assert not np.any(np.isnan(Hc)), f"NaN in Hc: {Hc}"
            
        print("[PASS] test_full_pipeline_mock passed successfully.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_no_cv2_in_part1()
    test_hartley_normalization()
    test_dlt_synthetic()
    test_ransac_with_outliers()
    test_kd_tree_sift_matching()
    test_full_pipeline_mock()
    print("\nALL TESTS PASSED SUCCESSFULLY! :)")
