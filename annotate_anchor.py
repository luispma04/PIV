"""
annotate_anchor.py - Interactive Court Anchoring Utility for PIV Project.

Allows user to associate landmark points on templateimg.jpg with official
landmarks in courtmodel.mat to compute the anchor homography H_ref_to_court.

Saves:
    anchor.mat in path_to_refdir containing:
        - H_ref_to_court: (3, 3) numpy matrix mapping reference image pixels to court meters.
        - img_pts: (K, 2) clicked pixel coordinates.
        - court_pts: (K, 2) corresponding court coordinates in meters.
        - landmark_names: names of selected landmarks.
"""

import os
import argparse
import cv2
import numpy as np
import scipy.io as sio

import part1


def run_interactive_anchoring(refdir, max_display_dim=1200):
    img_path = os.path.join(refdir, "templateimg.jpg")
    model_path = os.path.join(refdir, "courtmodel.mat")
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run court_model.py first.")
        return
        
    img = cv2.imread(img_path)
    model = sio.loadmat(model_path)
    court_pts = model["pts"]
    court_names = [str(n).strip() for n in model["names"]]
    
    print("\n--- Court Anchoring Instructions ---")
    print("Click at least 4 key court landmarks on the reference image in sequence.")
    print("Recommended points:")
    print(" 1. BL_singles_corner_bottom (Bottom-Left singles baseline corner)")
    print(" 2. BR_singles_corner_bottom (Bottom-Right singles baseline corner)")
    print(" 3. TR_singles_corner_top    (Top-Right singles baseline corner)")
    print(" 4. TL_singles_corner_top    (Top-Left singles baseline corner)")
    print(" 5. T_service_bottom         (Bottom service T)")
    print(" 6. T_service_top            (Top service T)")
    print("\nPress 'u' to undo last click, 'c' when done, 'q' to cancel.\n")
    
    # Scale for display if image is very large (e.g. 4K)
    h, w = img.shape[:2]
    scale = min(1.0, max_display_dim / max(h, w))
    disp_w, disp_h = int(w * scale), int(h * scale)
    
    clicked_disp = []
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_disp.append((x, y))
            
    win_name = "Anchor Court Landmarks (Click 4+ points, then press 'c')"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)
    
    while True:
        disp_img = cv2.resize(img, (disp_w, disp_h))
        for idx, pt in enumerate(clicked_disp):
            cv2.circle(disp_img, pt, 5, (0, 0, 255), -1)
            cv2.putText(disp_img, f"P{idx+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
        cv2.imshow(win_name, disp_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
        elif key == ord('u') and clicked_disp:
            clicked_disp.pop()
        elif key == ord('c'):
            if len(clicked_disp) < 4:
                print("Please click at least 4 points before proceeding.")
            else:
                break
                
    cv2.destroyAllWindows()
    
    # Convert clicked points back to full image resolution
    orig_img_pts = np.array([[pt[0] / scale, pt[1] / scale] for pt in clicked_disp], dtype=np.float64)
    print(f"\nCollected {len(orig_img_pts)} landmark points.")
    
    # Let user select which landmarks they clicked, or default to the 4 main court corners
    # Default 4: BL_singles, BR_singles, TR_singles, TL_singles
    default_names = [
        "BL_singles_corner_bottom",
        "BR_singles_corner_bottom",
        "TR_singles_corner_top",
        "TL_singles_corner_top",
        "T_service_bottom",
        "T_service_top"
    ]
    selected_names = default_names[:len(orig_img_pts)]
    
    name_to_pt = {name: pt for name, pt in zip(court_names, court_pts)}
    selected_court_pts = np.array([name_to_pt[name] for name in selected_names], dtype=np.float64)
    
    # Compute homography H_ref_to_court using Hartley DLT + RANSAC
    H_ref_to_court, _ = part1.ransac_homography(orig_img_pts, selected_court_pts, threshold=0.2)
    
    if H_ref_to_court is None:
        print("Error: Could not compute valid homography from points.")
        return
        
    out_anchor_path = os.path.join(refdir, "anchor.mat")
    sio.savemat(out_anchor_path, {
        "H_ref_to_court": H_ref_to_court,
        "img_pts": orig_img_pts,
        "court_pts": selected_court_pts,
        "landmark_names": np.array(selected_names, dtype=object)
    })
    print(f"Successfully saved anchor homography to {out_anchor_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive court landmark anchoring.")
    parser.add_argument("path_to_refdir", help="Directory with templateimg.jpg and courtmodel.mat")
    args = parser.parse_args()
    run_interactive_anchoring(args.path_to_refdir)
