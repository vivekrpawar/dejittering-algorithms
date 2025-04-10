from skimage.metrics import structural_similarity as ssim
import cv2
import os
import sys

def detect_scene_changes_ssim(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path) 
    if not os.path.exists(video_path): 
        print(f"Error: Video file {video_path} does not exist!")
        return []

    if not cap.isOpened():
        print("Error: Could not open video file")
        return []

    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    scene_changes = []
    frame_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        output_msg = f'Processing frame {frame_idx+1}/{frame_count}'
        print(output_msg, end="", flush=True)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(prev_gray, gray, full=True)

        if score < threshold:
            scene_changes.append(frame_idx)

        prev_gray = gray
        frame_idx += 1
        print("\r" + " " * len(output_msg), end="", flush=True)
        print("\r", end="", flush=True)

    cap.release()
    return scene_changes
video_path = sys.argv[1]
scenes = detect_scene_changes_ssim(video_path)
print("Scene change detected at frames:", scenes)