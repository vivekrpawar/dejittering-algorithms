import os
import sys
import cv2
from pathlib import Path

def generate_videos_from_frames(src_dir, dest_dir, fps=60, frame_extension=".png", width=512, height=512):
    # Ensure destination directory exists
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all subdirectories in the source directory
    for subdir in Path(src_dir).iterdir():
        if subdir.is_dir():
            # Get all image frames sorted by filename
            frames = sorted(subdir.glob(f"*{frame_extension}"))

            if not frames:
                print(f"No frames found in {subdir}. Skipping...")
                continue

            # Read the first frame to get the frame size
            sample_frame = cv2.imread(str(frames[0]))
            h, w, _ = sample_frame.shape

            # Define the video output path
            video_name = f"{subdir.name}.mp4"
            video_path = dest_dir / video_name

            # Initialize the VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # Write frames to the video
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))  
                video_writer.write(frame)

            # Release the VideoWriter
            video_writer.release()
            print(f"Video saved: {video_path}")

# Example usage
src_dir = sys.argv[1]
dest_dir = sys.argv[2]
generate_videos_from_frames(src_dir, dest_dir, fps=60, frame_extension=".png", width=512, height=512)
