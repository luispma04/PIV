"""
extract_frames.py - Extract image sequence from video for PIV project.

Usage:
    python extract_frames.py video_path --images_dir data/images --ref_dir data/ref --max_frames 100 --step 1
"""

import os
import argparse
import cv2


def extract_frames(video_path, images_dir, ref_dir=None, max_frames=None, step=1, prefix="frame", resize_dim=None):
    os.makedirs(images_dir, exist_ok=True)
    if ref_dir:
        os.makedirs(ref_dir, exist_ok=True)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[extract_frames] Video opened: {video_path}")
    print(f"                 Total frames: {total_frames}, FPS: {fps}, Resolution: {w}x{h}")
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            if resize_dim:
                frame = cv2.resize(frame, resize_dim)
                
            num_str = f"{saved_count:04d}"
            fname = f"{prefix}_{num_str}.jpg"
            out_path = os.path.join(images_dir, fname)
            cv2.imwrite(out_path, frame)
            
            # Save first frame as templateimg.jpg if ref_dir provided
            if saved_count == 0 and ref_dir:
                template_path = os.path.join(ref_dir, "templateimg.jpg")
                cv2.imwrite(template_path, frame)
                print(f"[extract_frames] Saved reference image: {template_path}")
                
            saved_count += 1
            if max_frames and saved_count >= max_frames:
                break
                
        frame_idx += 1
        
    cap.release()
    print(f"[extract_frames] Finished: Extracted {saved_count} frames into {images_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames to sequence.")
    parser.add_argument("video_path", help="Path to input video file (e.g. .mp4, .mov)")
    parser.add_argument("--images_dir", default="data/images", help="Target images directory")
    parser.add_argument("--ref_dir", default="data/ref", help="Target reference directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract")
    parser.add_argument("--step", type=int, default=1, help="Frame step interval (e.g. 1 = every frame, 2 = every 2nd)")
    parser.add_argument("--prefix", default="rally", help="Prefix for image filenames")
    parser.add_argument("--resize", nargs=2, type=int, default=None, help="Optional resize width height (e.g. 1280 720)")
    
    args = parser.parse_args()
    resize_tuple = tuple(args.resize) if args.resize else None
    extract_frames(args.video_path, args.images_dir, args.ref_dir,
                   max_frames=args.max_frames, step=args.step, prefix=args.prefix,
                   resize_dim=resize_tuple)
