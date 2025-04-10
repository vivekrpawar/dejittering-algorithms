import os
import sys
import cv2
from datetime import datetime

def extract_center(frame, width=440, height=440):
    h, w, _ = frame.shape
    start_x = (w - width) // 2
    start_y = (h - height) // 2
    return frame[start_y:start_y + height, start_x:start_x + width]

def main(folder_path, output_filename):
    # Get the list of files in the folder
    files = sorted(os.listdir(folder_path))

    # Define the codec and output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'MJPG' for .avi format
    dimensions = cv2.imread(os.path.join(folder_path, files[0])).shape
    height, width, _ = dimensions
    # output_video = cv2.VideoWriter(output_filename+'.avi', fourcc, 30, (dimensions[0], dimensions[1]))
    output_video = cv2.VideoWriter(output_filename, fourcc, 30, (width, height))  # Adjust resolution and FPS as needed
    print(f"Dimensions {height} {width}")
    count = 0
    for file_name in files:
        count += 1
        if file_name.endswith('.jpg') or file_name.endswith('.png'):  # Adjust based on your file format
            file_path = os.path.join(folder_path, file_name)
            # Modified
            # frame = extract_center(cv2.imread(file_path), width, height)
            frame = cv2.imread(file_path)
            output_video.write(frame)

    output_video.release()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_video_from_frames.py <folder_path> <output_filename>")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_filename = sys.argv[2]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)
    start_time = datetime.now() 
    main(folder_path, output_filename)
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds()
    print(time_difference)
    print("Video generation completed. Check output.mp4 in the current directory.")