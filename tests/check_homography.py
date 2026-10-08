"""
check_homography.py - Visual verification of homography tracking (H, Hc, and Top-View Minimap).

Usage:
    python check_homography.py 50 --images datasets/tennis/game1/Clip1 --ref ref_tennis --results results_tennis
"""

import sys
import os
import glob
import argparse
import cv2
import numpy as np
import scipy.io as sio


def render_topdown_court_view(Hc, frame_img, canvas_w=400, canvas_h=800):
    """
    Renders synthetic top-down bird's-eye view of the court in metric meters,
    and warps the camera view into top-down court coordinates.
    """
    # Real-world metric boundaries with margin:
    # X in [-6.5, 6.5] meters (width = 13.0 m)
    # Y in [-13.5, 13.5] meters (height = 27.0 m)
    margin_x = 6.5
    margin_y = 13.5

    # Mapping: (x_m, y_m) -> (u_canvas, v_canvas)
    # u = (x_m + margin_x) / (2 * margin_x) * canvas_w
    # v = (margin_y - y_m) / (2 * margin_y) * canvas_h  (y positive is top)
    Sx = canvas_w / (2.0 * margin_x)
    Sy = canvas_h / (2.0 * margin_y)

    M_court_to_canvas = np.array([
        [Sx,   0.0, Sx * margin_x],
        [0.0, -Sy,  Sy * margin_y],
        [0.0,  0.0, 1.0]
    ], dtype=np.float64)

    # Composite homography: Frame Pixels -> Top-Down Canvas
    # H_frame_to_canvas = M_court_to_canvas @ Hc
    H_frame_to_canvas = M_court_to_canvas @ Hc

    # Warp camera frame into top-down metric perspective
    warped_court = cv2.warpPerspective(frame_img, H_frame_to_canvas, (canvas_w, canvas_h))

    # Draw synthetic official court lines on top
    half_wd = 5.485
    half_ws = 4.115
    half_l = 11.885
    serv = 6.40

    court_lines_m = [
        ((-half_wd, -half_l), (half_wd, -half_l)),  # bottom baseline
        ((-half_wd, half_l), (half_wd, half_l)),    # top baseline
        ((-half_wd, -half_l), (-half_wd, half_l)),  # left doubles sideline
        ((half_wd, -half_l), (half_wd, half_l)),    # right doubles sideline
        ((-half_ws, -half_l), (-half_ws, half_l)),  # left singles sideline
        ((half_ws, -half_l), (half_ws, half_l)),    # right singles sideline
        ((-half_ws, -serv), (half_ws, -serv)),      # bottom service line
        ((-half_ws, serv), (half_ws, serv)),        # top service line
        ((0.0, -serv), (0.0, serv)),                # center service line
        ((-half_wd, 0.0), (half_wd, 0.0))           # net line
    ]

    for p1_m, p2_m in court_lines_m:
        u1 = int((p1_m[0] + margin_x) * Sx)
        v1 = int((margin_y - p1_m[1]) * Sy)
        u2 = int((p2_m[0] + margin_x) * Sx)
        v2 = int((margin_y - p2_m[1]) * Sy)
        cv2.line(warped_court, (u1, v1), (u2, v2), (0, 255, 255), 2, cv2.LINE_AA)

    # Net line in red
    u_net1 = int((-half_wd + margin_x) * Sx)
    v_net = int((margin_y - 0.0) * Sy)
    u_net2 = int((half_wd + margin_x) * Sx)
    cv2.line(warped_court, (u_net1, v_net), (u_net2, v_net), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.putText(warped_court, "Top-Down (Metric)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return warped_court


def draw_court_lines_on_frame(img, Hc, model_path):
    """
    Projects court lines in meters from courtmodel.mat onto frame pixels using Hc^-1.
    """
    if not os.path.exists(model_path):
        return img, False

    try:
        mdata = sio.loadmat(model_path)
        if "lines" not in mdata:
            return img, False
        lines_court = mdata["lines"]
        pts_court = mdata["pts"] if "pts" in mdata else []

        Hc_inv = np.linalg.inv(Hc)
    except Exception:
        return img, False

    out = img.copy()
    h, w = img.shape[:2]
    lines_drawn = 0

    for seg in lines_court:
        p1_m, p2_m = seg[0], seg[1]
        P1 = Hc_inv @ np.array([p1_m[0], p1_m[1], 1.0])
        P2 = Hc_inv @ np.array([p2_m[0], p2_m[1], 1.0])

        if abs(P1[2]) < 1e-6 or abs(P2[2]) < 1e-6:
            continue

        u1, v1 = P1[0] / P1[2], P1[1] / P1[2]
        u2, v2 = P2[0] / P2[2], P2[1] / P2[2]

        if -w <= u1 <= 2 * w and -h <= v1 <= 2 * h and -w <= u2 <= 2 * w and -h <= v2 <= 2 * h:
            pt1 = (int(np.clip(u1, -2000, 5000)), int(np.clip(v1, -2000, 5000)))
            pt2 = (int(np.clip(u2, -2000, 5000)), int(np.clip(v2, -2000, 5000)))
            cv2.line(out, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            lines_drawn += 1

    for pt_m in pts_court:
        P = Hc_inv @ np.array([pt_m[0], pt_m[1], 1.0])
        if abs(P[2]) < 1e-6:
            continue
        u, v = int(P[0] / P[2]), int(P[1] / P[2])
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), 4, (0, 0, 255), -1)

    return out, (lines_drawn > 0)


def find_file(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Check homography alignment visually.")
    parser.add_argument("frame", type=int, default=50, help="Frame number to check (e.g. 50)")
    parser.add_argument("--images", default=None, help="Directory containing sequence frames")
    parser.add_argument("--ref", default=None, help="Directory containing templateimg.jpg and courtmodel.mat")
    parser.add_argument("--results", default=None, help="Directory containing homography_NNNN.mat files")
    args = parser.parse_args()

    num_str = f"{args.frame:04d}"

    # Auto-detection
    images_dir = args.images or ("datasets/tennis/game1/Clip1" if os.path.exists("datasets/tennis/game1/Clip1") else "datasets/imageslisbon")
    results_dir = args.results or ("results_tennis" if os.path.exists("results_tennis") else "results")
    ref_dir = args.ref or ("ref_tennis" if os.path.exists("ref_tennis") else "ref")

    print(f"[check_homography] Frame:   {num_str}")
    print(f"[check_homography] Images:  {images_dir}")
    print(f"[check_homography] Ref:     {ref_dir}")
    print(f"[check_homography] Results: {results_dir}\n")

    ref_path = os.path.join(ref_dir, "templateimg.jpg")
    model_path = os.path.join(ref_dir, "courtmodel.mat")
    mat_path = os.path.join(results_dir, f"homography_{num_str}.mat")

    frame_path = find_file(images_dir, f"*{num_str}.jpg") or find_file(images_dir, f"*{num_str}.png")

    if not os.path.exists(ref_path):
        print(f"Error: {ref_path} not found.")
        return
    if not frame_path or not os.path.exists(frame_path):
        print(f"Error: Frame image for {num_str} not found in {images_dir}.")
        return
    if not os.path.exists(mat_path):
        print(f"Error: {mat_path} not found.")
        return

    # 1. Load images
    ref = cv2.imread(ref_path)
    frame = cv2.imread(frame_path)
    h, w = ref.shape[:2]

    # 2. Load homographies
    data = sio.loadmat(mat_path)
    H = data["H"]
    Hc = data["Hc"] if "Hc" in data else None

    # 3. Check H: Warp frame onto reference image
    warped = cv2.warpPerspective(frame, H, (w, h))
    blend = cv2.addWeighted(ref, 0.5, warped, 0.5, 0)

    # 4. Construct visualization panels
    panels = []
    panels.append(ref.copy())
    panels.append(warped.copy())
    panels.append(blend.copy())

    # 5. Check Hc: Project court lines onto camera frame
    if Hc is not None and os.path.exists(model_path):
        court_overlay, drawn = draw_court_lines_on_frame(frame, Hc, model_path)
        if drawn:
            panels.append(court_overlay)

    # 6. Check Top-Down Minimap
    topdown_img = None
    if Hc is not None:
        topdown_img = render_topdown_court_view(Hc, frame)

    labels = ["Reference (template)", f"Warped Frame {num_str} (H)", "50/50 Overlay Blend", "Court Reprojection (Hc)"]
    target_height = 420
    resized_panels = []

    for idx, p in enumerate(panels):
        lbl = labels[idx] if idx < len(labels) else f"Panel {idx+1}"
        cv2.putText(p, lbl, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if idx == 0 else (0, 255, 0), 2)
        ph, pw = p.shape[:2]
        new_w = int(pw * (target_height / ph))
        resized_panels.append(cv2.resize(p, (new_w, target_height)))

    if topdown_img is not None:
        t_h, t_w = topdown_img.shape[:2]
        new_w = int(t_w * (target_height / t_h))
        resized_panels.append(cv2.resize(topdown_img, (new_w, target_height)))

    combined = np.hstack(resized_panels)
    out_file = f"comparison_{num_str}.jpg"
    cv2.imwrite(out_file, combined)
    print(f"[OK] Saved visual comparison to: {out_file}")


if __name__ == "__main__":
    main()
