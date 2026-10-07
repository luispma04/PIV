"""
main1.py - Driver script for Part 1 of PIV Project.

The running command:
    python main1.py path_to_refdir path_images_dir path_feature_dir path_output_dir
"""

import os
import sys
import cv2
import numpy as np
import scipy.io as sio

import part1


# ==============================================================================
# Exact code from pivist/features distributed by professors
# ==============================================================================

def get_keypoints(input_folder, fname, output_folder):
    # Prepare filenames
    impath = os.path.join(input_folder, fname)
    base_name = os.path.splitext(fname)[0]
    outim_path = os.path.join(output_folder, fname)
    outkp_path = os.path.join(output_folder, f"{base_name}.mat")

    # Load image
    im = cv2.imread(impath)
    imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(imrgb, None)
    for k in kp:
        cx, cy = int(k.pt[0]), int(k.pt[1])
        cv2.circle(im, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(outim_path, im)

    combined = np.concatenate((np.array([k.pt for k in kp], dtype=np.float32).T, des.T), axis=0)
    sio.savemat(outkp_path, {'kp': combined})


def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Iterate over all images in folder
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(exts):
            try:
                combined = get_keypoints(input_folder, fname, output_folder)
            except Exception as e:
                print(f"Failed to process {fname}: {e}")


# ==============================================================================
# Main Runner calling part1
# ==============================================================================

def main():
    if len(sys.argv) != 5:
        print("Usage: python main1.py path_to_refdir path_images_dir path_feature_dir path_output_dir")
        sys.exit(1)

    path_to_refdir = sys.argv[1]
    path_images_dir = sys.argv[2]
    path_feature_dir = sys.argv[3]
    path_output_dir = sys.argv[4]

    # Process reference image templateimg.jpg if it exists
    template_img_path = os.path.join(path_to_refdir, "templateimg.jpg")
    if os.path.exists(template_img_path):
        get_keypoints(path_to_refdir, "templateimg.jpg", path_feature_dir)

    # Process all image files in path_images_dir
    process_folder(path_images_dir, path_feature_dir)

    # Call part1 as required by assignment specification
    part1.part1(path_to_refdir, path_images_dir, path_feature_dir, path_output_dir)


if __name__ == "__main__":
    main()
