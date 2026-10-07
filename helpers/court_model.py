"""
court_model.py - ITF Official Tennis Court Model Generator.

Official ITF Dimensions:
- Total length: 23.77 m
- Singles width: 8.23 m
- Doubles width: 10.97 m
- Service line: 6.40 m from net
- Net line: at center (0.0 m)

This script generates `courtmodel.mat` containing:
- "pts": (N, 2) numpy array of key court landmark points in meters.
- "names": list/array of string labels for each landmark.
- "lines": court line segment definitions for rendering and reprojection error checks.
"""

import os
import argparse
import numpy as np
import scipy.io as sio


def create_tennis_court_model(origin="net_center"):
    """
    Creates official ITF tennis court landmarks and boundary lines.
    
    Coordinates in meters.
    If origin == 'net_center':
        x: across width [-5.485, +5.485]
        y: along length [-11.885, +11.885]
        net is at y = 0
    If origin == 'corner':
        x: [0.0, 10.97]
        y: [0.0, 23.77]
    """
    total_length = 23.77
    doubles_width = 10.97
    singles_width = 8.23
    service_dist_from_net = 6.40
    
    half_l = total_length / 2.0        # 11.885 m
    half_wd = doubles_width / 2.0      # 5.485 m
    half_ws = singles_width / 2.0      # 4.115 m
    
    # Landmark definitions in centered coordinates (x, y)
    landmarks = {
        # Outer doubles corners
        "BL_doubles_corner_bottom": (-half_wd, -half_l),
        "BR_doubles_corner_bottom": (half_wd, -half_l),
        "TL_doubles_corner_top": (-half_wd, half_l),
        "TR_doubles_corner_top": (half_wd, half_l),
        
        # Singles corners on baselines
        "BL_singles_corner_bottom": (-half_ws, -half_l),
        "BR_singles_corner_bottom": (half_ws, -half_l),
        "TL_singles_corner_top": (-half_ws, half_l),
        "TR_singles_corner_top": (half_ws, half_l),
        
        # Service line intersections with singles sidelines
        "L_service_bottom": (-half_ws, -service_dist_from_net),
        "R_service_bottom": (half_ws, -service_dist_from_net),
        "L_service_top": (-half_ws, service_dist_from_net),
        "R_service_top": (half_ws, service_dist_from_net),
        
        # Center service Ts
        "T_service_bottom": (0.0, -service_dist_from_net),
        "T_service_top": (0.0, service_dist_from_net),
        
        # Net intersections
        "Net_center": (0.0, 0.0),
        "Net_left_singles": (-half_ws, 0.0),
        "Net_right_singles": (half_ws, 0.0),
        "Net_left_doubles": (-half_wd, 0.0),
        "Net_right_doubles": (half_wd, 0.0),
        
        # Center baseline marks
        "Center_mark_bottom": (0.0, -half_l),
        "Center_mark_top": (0.0, half_l),
    }
    
    names = list(landmarks.keys())
    pts = np.array([landmarks[k] for k in names], dtype=np.float64)
    
    if origin == "corner":
        # Shift so (0,0) is bottom-left doubles corner
        pts[:, 0] += half_wd
        pts[:, 1] += half_l
        
    # Standard lines for court rendering: pairs of coordinates ((x1, y1), (x2, y2))
    lines = [
        # Doubles sidelines
        ((-half_wd, -half_l), (-half_wd, half_l)),
        ((half_wd, -half_l), (half_wd, half_l)),
        # Baselines
        ((-half_wd, -half_l), (half_wd, -half_l)),
        ((-half_wd, half_l), (half_wd, half_l)),
        # Singles sidelines
        ((-half_ws, -half_l), (-half_ws, half_l)),
        ((half_ws, -half_l), (half_ws, half_l)),
        # Service lines
        ((-half_ws, -service_dist_from_net), (half_ws, -service_dist_from_net)),
        ((-half_ws, service_dist_from_net), (half_ws, service_dist_from_net)),
        # Center service line
        ((0.0, -service_dist_from_net), (0.0, service_dist_from_net)),
        # Net line
        ((-half_wd, 0.0), (half_wd, 0.0)),
    ]
    lines = np.array(lines, dtype=np.float64)
    if origin == "corner":
        lines[:, :, 0] += half_wd
        lines[:, :, 1] += half_l
        
    return pts, names, lines


def save_court_model(output_path, origin="net_center"):
    pts, names, lines = create_tennis_court_model(origin=origin)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save formatted court model
    data = {
        "pts": pts,
        "names": np.array(names, dtype=object),
        "lines": lines
    }
    sio.savemat(output_path, data)
    print(f"[court_model] Saved court model to {output_path} with {len(pts)} landmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ITF tennis court model.")
    parser.add_argument("--output", default="courtmodel.mat", help="Path to output .mat file")
    parser.add_argument("--origin", default="net_center", choices=["net_center", "corner"],
                        help="Coordinate frame origin (default: net_center)")
    args = parser.parse_args()
    save_court_model(args.output, origin=args.origin)
