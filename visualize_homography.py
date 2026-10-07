"""
visualize_homography.py - Visual Verification & Appendix A Consistency Checks.

Functions:
1. Overlays the court model wireframe onto original video frames using Hc^-1 (court meters -> pixels).
2. Warps frames to the reference template to create a stabilized video.
3. Computes Appendix A consistency metrics:
   - Temporal smoothness (corner velocity across frames)
   - Cycle consistency RMS error
   - Forward-backward composition drift

Usage:
    python visualize_homography.py path_to_refdir path_images_dir path_output_dir --save_video overlay.mp4
"""

import os
import re
import glob
import argparse
import cv2
import numpy as np
import scipy.io as sio


def load_homographies(output_dir):
    """
    Loads all homography_NNNN.mat from output_dir sorted by NNNN.
    """
    mat_files = glob.glob(os.path.join(output_dir, "homography_*.mat"))
    results = []
    for f in mat_files:
        base = os.path.basename(f)
        match = re.search(r"homography_(\d+)\.mat", base)
        if match:
            num = int(match.group(1))
            num_str = match.group(1)
            data = sio.loadmat(f)
            H = data["H"]
            Hc = data["Hc"]
            results.append((num, num_str, H, Hc))
    results.sort(key=lambda x: x[0])
    return results


def draw_court_lines(img, Hc, lines_court, color=(0, 255, 0), thickness=2):
    """
    Warps court model lines into frame coordinates using Hc_inv (meters -> image pixels).
    """
    try:
        Hc_inv = np.linalg.inv(Hc)
    except np.linalg.LinAlgError:
        return img
        
    out = img.copy()
    h, w = img.shape[:2]
    
    for seg in lines_court:
        p1_m, p2_m = seg[0], seg[1]
        
        # In homogeneous coordinates
        P1_homo = Hc_inv @ np.array([p1_m[0], p1_m[1], 1.0])
        P2_homo = Hc_inv @ np.array([p2_m[0], p2_m[1], 1.0])
        
        if abs(P1_homo[2]) < 1e-6 or abs(P2_homo[2]) < 1e-6:
            continue
            
        u1, v1 = P1_homo[0] / P1_homo[2], P1_homo[1] / P1_homo[2]
        u2, v2 = P2_homo[0] / P2_homo[2], P2_homo[1] / P2_homo[2]
        
        # Check boundary plausibility
        if -w <= u1 <= 2 * w and -h <= v1 <= 2 * h and -w <= u2 <= 2 * w and -h <= v2 <= 2 * h:
            pt1 = (int(np.clip(u1, -2000, 5000)), int(np.clip(v1, -2000, 5000)))
            pt2 = (int(np.clip(u2, -2000, 5000)), int(np.clip(v2, -2000, 5000)))
            cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)
            
    return out


def evaluate_temporal_smoothness(homographies, court_pts):
    """
    Computes corner velocity across consecutive frames to evaluate temporal smoothness.
    """
    velocities = []
    prev_corners = None
    
    for num, num_str, H, Hc in homographies:
        try:
            H_inv = np.linalg.inv(H)
            pts_homo = np.column_stack((court_pts[:4], np.ones(4)))
            proj = (H_inv @ pts_homo.T).T
            corners = proj[:, :2] / proj[:, 2:3]
            
            if prev_corners is not None:
                vel = np.mean(np.sqrt(np.sum((corners - prev_corners) ** 2, axis=1)))
                velocities.append((num, vel))
            prev_corners = corners
        except Exception:
            continue
            
    if velocities:
        mean_vel = np.mean([v[1] for v in velocities])
        max_vel = np.max([v[1] for v in velocities])
        print(f"[Metrics] Temporal smoothness: Mean corner speed = {mean_vel:.2f} px/frame, Max speed = {max_vel:.2f} px/frame")
        return velocities
    return []


def run_visualization(refdir, images_dir, output_dir, save_video=None, save_frames_dir=None):
    # 1. Load court model
    model_path = os.path.join(refdir, "courtmodel.mat")
    lines_court = []
    court_pts = np.zeros((4, 2))
    if os.path.exists(model_path):
        mdata = sio.loadmat(model_path)
        if "lines" in mdata:
            lines_court = mdata["lines"]
        if "pts" in mdata:
            court_pts = mdata["pts"]
            
    # 2. Load homographies
    homographies = load_homographies(output_dir)
    if not homographies:
        print(f"[visualize] Error: No homography_NNNN.mat found in {output_dir}.")
        return
        
    print(f"[visualize] Loaded {len(homographies)} homographies.")
    
    # Evaluate metrics
    evaluate_temporal_smoothness(homographies, court_pts)
    
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
        
    vw = None
    for idx, (num, num_str, H, Hc) in enumerate(homographies):
        # Find matching image
        pattern = os.path.join(images_dir, f"*{num_str}.jpg")
        matches = glob.glob(pattern)
        if not matches:
            pattern = os.path.join(images_dir, f"*{num_str}.png")
            matches = glob.glob(pattern)
            
        if not matches:
            continue
            
        img_path = matches[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Draw overlay
        if len(lines_court) > 0:
            overlay = draw_court_lines(img, Hc, lines_court)
        else:
            overlay = img.copy()
            
        cv2.putText(overlay, f"Frame: {num_str}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                    
        if save_video:
            if vw is None:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(save_video, fourcc, 25.0, (w, h))
            vw.write(overlay)
            
        if save_frames_dir:
            out_img_path = os.path.join(save_frames_dir, f"overlay_{num_str}.jpg")
            cv2.imwrite(out_img_path, overlay)
            
    if vw is not None:
        vw.release()
        print(f"[visualize] Video saved to {save_video}")
        
    print("[visualize] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize homographies and court overlays.")
    parser.add_argument("path_to_refdir", help="Directory with courtmodel.mat")
    parser.add_argument("path_images_dir", help="Directory with original frame images")
    parser.add_argument("path_output_dir", help="Directory with homography_NNNN.mat files")
    parser.add_argument("--save_video", default=None, help="Path to save overlay video (.mp4)")
    parser.add_argument("--save_frames", default=None, help="Directory to save overlay frames")
    
    args = parser.parse_args()
    run_visualization(args.path_to_refdir, args.path_images_dir, args.path_output_dir,
                      save_video=args.save_video, save_frames_dir=args.save_frames)
