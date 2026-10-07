"""
main1.py - Driver script for Part 1 of PIV Project.

Usage:
    python main1.py path_to_refdir path_images_dir path_feature_dir path_output_dir

Responsibilities:
1. May import cv2.
2. Must import part1.
3. Extracts SIFT features from all frames in path_images_dir and saves them as .mat files
   in path_feature_dir adhering to the format defined by pivist/features:
   combined = np.concatenate((np.array([k.pt for k in kp]).T, des.T), axis=0) -> shape (130, N).
4. Extracts SIFT features for templateimg.jpg in path_to_refdir if not already extracted.
5. Calls part1.part1(path_to_refdir, path_images_dir, path_feature_dir, path_output_dir).
"""

import os
import sys
import argparse
import cv2
import numpy as np
import scipy.io as sio

import part1


def extract_and_save_sift_features(image_path, out_mat_path, sift_detector, save_debug_image=False, debug_img_path=None):
    """
    Extracts SIFT features from an image and saves them in .mat format matching pivist/features.
    
    The saved dictionary contains:
        'kp': array of shape (130, N)
              rows 0-1: (x, y) keypoint coordinates
              rows 2-129: 128-dimensional SIFT descriptors
    """
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    kp, des = sift_detector.detectAndCompute(imrgb, None)
    
    if kp is None or len(kp) == 0 or des is None:
        combined = np.empty((130, 0), dtype=np.float32)
    else:
        pts = np.array([k.pt for k in kp], dtype=np.float32).T  # (2, N)
        combined = np.concatenate((pts, des.T.astype(np.float32)), axis=0)  # (130, N)
        
    sio.savemat(out_mat_path, {"kp": combined})
    
    if save_debug_image and debug_img_path and kp is not None:
        out_vis = im.copy()
        for k in kp:
            cx, cy = int(k.pt[0]), int(k.pt[1])
            cv2.circle(out_vis, (cx, cy), 2, (0, 0, 255), -1)
        cv2.imwrite(debug_img_path, out_vis)
        
    return len(kp) if kp else 0


def process_features(path_to_refdir, path_images_dir, path_feature_dir):
    """
    Generates .mat feature files for all images in path_images_dir
    and the reference image templateimg.jpg in path_to_refdir.
    """
    os.makedirs(path_feature_dir, exist_ok=True)
    sift = cv2.SIFT_create()
    
    # 1. Process reference image templateimg.jpg if it exists
    ref_image_path = os.path.join(path_to_refdir, "templateimg.jpg")
    ref_mat_path = os.path.join(path_feature_dir, "templateimg.mat")
    
    if os.path.exists(ref_image_path):
        if not os.path.exists(ref_mat_path):
            print(f"[main1] Extracting SIFT features for reference: {ref_image_path}...")
            n_kp = extract_and_save_sift_features(ref_image_path, ref_mat_path, sift)
            print(f"[main1] Reference features saved: {n_kp} keypoints.")
        else:
            print(f"[main1] Reference feature file already exists: {ref_mat_path}")
    else:
        print(f"[main1] Note: templateimg.jpg not found in {path_to_refdir}. Will use first sequence frame.")
        
    # 2. Process all images in path_images_dir
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    all_files = sorted([f for f in os.listdir(path_images_dir) if f.lower().endswith(exts)])
    
    if not all_files:
        print(f"[main1] Warning: No image files found in {path_images_dir}!")
        return
        
    print(f"[main1] Checking/extracting SIFT features for {len(all_files)} images...")
    extracted_count = 0
    skipped_count = 0
    
    for idx, fname in enumerate(all_files):
        base_name = os.path.splitext(fname)[0]
        out_mat_path = os.path.join(path_feature_dir, f"{base_name}.mat")
        img_path = os.path.join(path_images_dir, fname)
        
        if os.path.exists(out_mat_path):
            skipped_count += 1
            continue
            
        try:
            extract_and_save_sift_features(img_path, out_mat_path, sift)
            extracted_count += 1
        except Exception as e:
            print(f"[main1] Failed to extract features for {fname}: {e}")
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_files):
            print(f"[main1] Progress: {idx + 1}/{len(all_files)} processed.")
            
    print(f"[main1] SIFT extraction complete: {extracted_count} newly extracted, {skipped_count} cached.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Part 1 of PIV Project: SIFT extraction and planar homography tracking."
    )
    parser.add_argument("path_to_refdir", help="Directory with templateimg.jpg and courtmodel.mat")
    parser.add_argument("path_images_dir", help="Directory with image sequence somename_NNNN.jpg")
    parser.add_argument("path_feature_dir", help="Directory where .mat feature files are stored")
    parser.add_argument("path_output_dir", help="Directory for output homography_NNNN.mat results")
    
    args = parser.parse_args()
    
    # 1. Feature processing
    process_features(args.path_to_refdir, args.path_images_dir, args.path_feature_dir)
    
    # 2. Run core tracking pipeline (part1.py)
    print("\n[main1] Invoking part1.part1()...")
    part1.part1(args.path_to_refdir, args.path_images_dir, args.path_feature_dir, args.path_output_dir)
    print("[main1] Part 1 completed successfully.")


if __name__ == "__main__":
    main()
