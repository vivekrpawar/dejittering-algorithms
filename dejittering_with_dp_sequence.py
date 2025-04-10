import numpy as np
import sys
import cv2
import os
import glob 

def intensity_diff(x, y, s, s_prev, max_shift, alpha):
    s_max = max(s, s_prev) # Maximum shift
    s_min = min(s, s_prev) 
    row_length = len(x) - 2 * max_shift
    valid_length = row_length - 2*s_max
    x_shifted = np.roll(x, s)
    y_shifted = np.roll(y, s_prev)
    x_shifted = x_shifted[max_shift+s_max:max_shift+row_length+s_min] 
    y_shifted = y_shifted[max_shift+s_max:max_shift+row_length+s_min]
    return (np.sum(100*np.abs(x_shifted - y_shifted)**alpha))/valid_length

def remove_line_jitter(image, max_shift, intensity_diff, penalty, alpha=2, _lambda=50):
    input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = input_frame.shape
    C = 3
    frame = np.zeros((H, W + 2 * max_shift))
    frame[:, max_shift:W + max_shift] = input_frame
    cost = [[float('inf')] * (2 * max_shift + 1) for _ in range(H)]
    shifts = [[0] * (2 * max_shift + 1) for _ in range(H)]
    
    
    for s in range(-max_shift, max_shift + 1):
        cost[0][s + max_shift] = intensity_diff(frame[0], frame[0], s, s, max_shift, alpha)
    
    
    for i in range(1, H):
        # output_msg = f'Processing row {i}/{H-1}'
        # print(output_msg, end="", flush=True)
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                shift_cost = cost[i-1][s_prev + max_shift] #+ _lambda * penalty(s - s_prev)
                current_cost = shift_cost + intensity_diff(frame[i], frame[i-1], s, s_prev, max_shift, alpha) #+ intensity_diff(frame[i], frame[i-1], s, s_prev, max_shift, 0.5)
                if current_cost < cost[i][s + max_shift]:
                    cost[i][s + max_shift] = current_cost
                    shifts[i][s + max_shift] = s_prev
        # print("\r" + " " * len(output_msg), end="", flush=True)
        # print("\r", end="", flush=True)

     
    # np.savetxt('cost.csv', cost, delimiter=',') 
    corrected_shifts = [0] * H
    s_opt = min(range(-max_shift, max_shift + 1), key=lambda s: cost[H-1][s + max_shift])
    corrected_shifts[H-1] = s_opt
    for i in range(H-2, -1, -1):
        corrected_shifts[i] = shifts[i+1][corrected_shifts[i+1] + max_shift]
    

   # Apply shifts to the frame
    corrected_frame = np.zeros(((H, W + 2 * max_shift, C)))
    for i in range(H):
        for c in range(3):  
            corrected_frame[i, max_shift+corrected_shifts[i]:W + max_shift + corrected_shifts[i], c] =  image[i,:, c]
    return corrected_frame[:, max_shift:W + max_shift, :].astype(np.uint8)

penalty = lambda x: x*x

def process_frames(input_video, output_video, max_shift, penalty, alpha, _lambda=50):
    # Create output directory if it doesn't exist
    output_dir = f"{output_video}_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(f"{output_video}.mp4", fourcc, fps, (frame_width, frame_height))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop when no more frames are available

        out_msg = f'Processing frame {count+1}/{total_frames}'
        print(out_msg, end="\n", flush=True)

        # Apply jitter removal
        corrected_image = remove_line_jitter(frame, max_shift, intensity_diff, penalty, alpha, _lambda)

        # Save corrected frame as an image
        cv2.imwrite(os.path.join(output_dir, f'corrected_{count}.png'), corrected_image)

        # Write corrected frame to output video
        out.write(corrected_image)

        count += 1

    # Release resources
    cap.release()
    out.release()
    
    print(f'Corrected video saved as {output_video}.mp4')
    print(f'Corrected frames saved in {output_dir}')

# Get input directory and output video file from command line arguments
input_dir = sys.argv[1]
output_video = sys.argv[2]
max_shift = int(sys.argv[3])
alpha = float(sys.argv[4]) 
# print(f"input directory: {input_dir} output video: {output_video} max_shift: {max_shift} alpha: {alpha} lambda: {_lambda}") 
process_frames(input_dir, output_video, max_shift, penalty, alpha)
