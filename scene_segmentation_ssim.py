from skimage.metrics import structural_similarity as ssim
import cv2
import os
import sys

def detect_and_save_scenes(video_path, output_dir, threshold=0.5):
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist!")
        return []

    if not cap.isOpened():
        print("Error: Could not open video file")
        return []

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize variables
    scene_changes = []
    scene_idx = 0
    frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(os.path.join(output_dir, f'scene_{scene_idx}.mp4'), fourcc, fps, (frame_width, frame_height))
    frame_per_scene = 0
    while True:
        output_msg = f'Processing frame {frame_idx+1}/{frame_count}'
        frame_per_scene += 1
        print(output_msg, end="", flush=True)

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(prev_gray, gray, full=True)

        if score < threshold:  # Scene change detected
            scene_changes.append(frame_idx)
            print(f"\nScene change detected at frame {frame_idx}, total frames {frame_per_scene}, saving scene {scene_idx}...")
            frame_per_scene = 0
            # Close the current writer
            out.release()
            scene_idx += 1
            out = cv2.VideoWriter(os.path.join(output_dir, f'scene_{scene_idx}.mp4'), fourcc, fps, (frame_width, frame_height))

        # Write frame to the current scene
        out.write(frame)
        prev_gray = gray
        frame_idx += 1

        print("\r" + " " * len(output_msg), end="", flush=True)
        print("\r", end="", flush=True)

    # Release resources
    cap.release()
    out.release()
    print("\nScene segmentation completed.")
    return scene_changes

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scene_segmentation.py <video_path> <output_directory>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_directory = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    scenes = detect_and_save_scenes(video_path, output_directory, threshold)
    print("Scene change detected at frames:", scenes)