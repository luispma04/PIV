"""
extract_rally.py - Extract tennis rallies/clips from a long video for PIV Part 1.

Supports extracting a single rally or multiple rallies in one command.
Timestamps can be in seconds (e.g. 750), MM:SS (e.g. 12:30), or HH:MM:SS (e.g. 01:12:30).

Usage Examples:
    # 1. Single rally:
    python extract_rally.py match.mp4 --start 12:30 --end 12:45 --out datasets/my_match/rally1

    # 2. Single rally with duration:
    python extract_rally.py match.mp4 --start 12:30 --duration 15 --out datasets/my_match/rally1

    # 3. Multiple rallies at once:
    python extract_rally.py match.mp4 --rallies "12:30-12:45" "18:10-18:25" "25:00-25:20" --out datasets/my_match
    # (Creates datasets/my_match/rally_01/, rally_02/, etc.)
"""

import os
import sys
import argparse
import cv2


def parse_timestamp(t_str):
    """
    Parses timestamp string to seconds.
    Supports:
        - raw seconds: "750" or "750.5"
        - MM:SS:       "12:30"
        - HH:MM:SS:    "01:12:30"
    """
    t_str = t_str.strip()
    parts = t_str.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid timestamp format: '{t_str}'. Expected 'SS', 'MM:SS', or 'HH:MM:SS'.")


def extract_segment(cap, fps, start_sec, end_sec, out_dir, prefix="frame", target_size=(1280, 720)):
    """Extracts a segment of frames between start_sec and end_sec."""
    os.makedirs(out_dir, exist_ok=True)
    
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = int(round(end_sec * fps))
    total_frames = end_frame - start_frame
    
    if total_frames <= 0:
        print(f"[Error] End time ({end_sec}s) must be greater than start time ({start_sec}s).")
        return 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    saved_count = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[Notice] Video ended early at frame {start_frame + i}.")
            break

        # Resize to standard 720p if requested
        if target_size is not None:
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        out_name = os.path.join(out_dir, f"{prefix}_{saved_count:04d}.jpg")
        cv2.imwrite(out_name, frame)
        saved_count += 1

    print(f"[Done] Saved {saved_count} frames to: {out_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract tennis rallies/clips from video.")
    parser.add_argument("video", help="Path to input video file (e.g. match.mp4)")
    
    # Mode A: Single rally
    parser.add_argument("--start", help="Start time (e.g. '12:30' or '750')")
    parser.add_argument("--end", help="End time (e.g. '12:45' or '765')")
    parser.add_argument("--duration", type=float, help="Duration in seconds (e.g. 15)")
    
    # Mode B: Multiple rallies
    parser.add_argument("--rallies", nargs="+", help="Multiple intervals, e.g. --rallies '12:30-12:45' '18:00-18:15'")
    
    # Output settings
    parser.add_argument("--out", required=True, help="Output directory (e.g. datasets/my_match/rally1 or datasets/my_match)")
    parser.add_argument("--prefix", default="frame", help="Filename prefix (default: 'frame' -> frame_0000.jpg)")
    parser.add_argument("--no-resize", action="store_true", help="Keep native resolution instead of 1280x720")
    
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur_sec = total_video_frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n[Video Info] {args.video}")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS:        {fps:.2f}")
    print(f"  Duration:   {video_dur_sec / 60:.1f} minutes ({total_video_frames} frames)\n")

    target_size = None if args.no_resize else (1280, 720)

    # Process Multiple Rallies
    if args.rallies:
        for idx, interval in enumerate(args.rallies):
            if "-" not in interval:
                print(f"Skipping invalid interval format '{interval}'. Expected 'START-END' (e.g. 12:30-12:45).")
                continue
            parts = interval.split("-")
            t_start = parse_timestamp(parts[0])
            t_end = parse_timestamp(parts[1])
            rally_out_dir = os.path.join(args.out, f"rally_{idx+1:02d}")
            print(f"--- Extracting Rally {idx+1}/{len(args.rallies)}: {parts[0]} -> {parts[1]} ({t_end - t_start:.1f}s) ---")
            extract_segment(cap, fps, t_start, t_end, rally_out_dir, prefix=args.prefix, target_size=target_size)

    # Process Single Rally
    elif args.start is not None:
        t_start = parse_timestamp(args.start)
        if args.end is not None:
            t_end = parse_timestamp(args.end)
        elif args.duration is not None:
            t_end = t_start + args.duration
        else:
            print("Error: For a single rally, specify either --end (e.g. 12:45) or --duration (e.g. 15).")
            sys.exit(1)

        print(f"--- Extracting Rally: {t_start:.1f}s -> {t_end:.1f}s ({t_end - t_start:.1f}s) ---")
        extract_segment(cap, fps, t_start, t_end, args.out, prefix=args.prefix, target_size=target_size)

    else:
        print("Error: Please provide either --start and --end (for 1 rally) or --rallies (for multiple rallies).")
        parser.print_help()

    cap.release()


if __name__ == "__main__":
    main()
