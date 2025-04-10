import numpy as np
import sys
import cv2
import os
import glob 

def compute_intensity_diff(x, y, s_x, s_y, max_shift, alpha): 
    s_max = max(s_x, s_y)
    s_min = min(s_x, s_y)
    row_length = len(x) - 2 * max_shift
    valid_length = row_length - 2 * s_max
    if valid_length <= 0:
        return float('inf')  # Invalid region
    
    x_shifted = np.roll(x, s_x)
    y_shifted = np.roll(y, s_y)
    
    # Extract valid overlapping region
    x_valid = x_shifted[max_shift + s_max : max_shift + row_length + s_min]
    y_valid = y_shifted[max_shift + s_max : max_shift + row_length + s_min]
    
    return (np.sum(100 * np.abs(x_valid - y_valid)**alpha)) / valid_length

def remove_line_jitter(image, max_shift, alpha=2, penalty_lambda=0):
    input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = input_frame.shape
    C = 3  # Assuming BGR input
    
    # Pad frame to handle shifts
    padded_frame = np.zeros((H, W + 2 * max_shift))
    padded_frame[:, max_shift:W + max_shift] = input_frame
    
    # DP tables: cost[i][s] = minimal cost to align row i with shift s
    cost = [[float('inf')] * (2 * max_shift + 1) for _ in range(H)]
    shifts = [[0] * (2 * max_shift + 1) for _ in range(H)]  # Track optimal previous shifts
    
    # Initialize first row
    for s in range(-max_shift, max_shift + 1):
        cost[0][s + max_shift] = 0  # No cost for first row
    
    # Dynamic Programming: forward pass
    for i in range(1, H):
        print(f"Processing row {i}/{H-1}", end="\r")
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                if cost[i-1][s_prev + max_shift] == float('inf'):
                    continue  # Skip invalid predecessors
                
                # Calculate cost from i-1
                prev_cost = cost[i-1][s_prev + max_shift]
                diff_i_minus_1 = compute_intensity_diff(padded_frame[i], padded_frame[i-1], s, s_prev, max_shift, alpha)
                
                # Include cost from i-2 if available
                if i >= 2:
                    s_prev_prev = shifts[i-1][s_prev + max_shift]
                    diff_i_minus_2 = compute_intensity_diff(padded_frame[i], padded_frame[i-2], s, s_prev_prev, max_shift, alpha)
                    diff_total = 0.7 * diff_i_minus_1 + 0.3 * diff_i_minus_2   
                else:
                    diff_total = diff_i_minus_1
                 
                shift_change_penalty = penalty_lambda * abs(s - s_prev)
                current_cost = prev_cost + diff_total + shift_change_penalty
                
                if current_cost < cost[i][s + max_shift]:
                    cost[i][s + max_shift] = current_cost
                    shifts[i][s + max_shift] = s_prev   
     
    corrected_shifts = [0] * H
    s_opt = np.argmin(cost[H-1]) - max_shift
    corrected_shifts[H-1] = s_opt
    
    for i in range(H-2, -1, -1):
        corrected_shifts[i] = shifts[i+1][corrected_shifts[i+1] + max_shift]
     
    corrected_frame = np.zeros((H, W + 2 * max_shift, C))
    for i in range(H):
        shift = corrected_shifts[i]
        start = max_shift + shift
        end = start + W 
        start = max(start, 0)
        end = min(end, W + 2 * max_shift)
        corrected_frame[i, start:end, :] = image[i, :end - start, :]
    
    return corrected_frame[:, max_shift:W + max_shift, :].astype(np.uint8)

penalty = lambda x: x*x

def process_frames(input_dir, output_video, max_shift, alpha, _lambda):
    if(not os.path.exists(output_video)):
        os.makedirs(output_video)
    # Get list of all image files in the directory
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))  # Assuming the frames are in PNG format
    if not image_files:
        print(f"No images found in directory {input_dir}")
        return
    print(f'output directory {output_video}')
    # Read the first image to get the dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    size = (width, height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(f'{output_video}.mp4', fourcc, 30.0, size)  # Assuming 30 FPS
    count = 0
    total_frames = len(image_files)
    for image_file in image_files:
        out_msg = f'Processing frame {count}/{total_frames}'
        print(out_msg, end="\n", flush=True)
        input_image = cv2.imread(image_file)
        corrected_image = remove_line_jitter(input_image, max_shift, alpha, _lambda)
        # Savd the corrected image
        cv2.imwrite(f'{output_video}/corrected_{count}.png', corrected_image)
        out.write(corrected_image) 
        count += 1

    out.release()
    print(f'Video saved as {output_video}')

# Get input directory and output video file from command line arguments
input_dir = sys.argv[1] # Input directory
output_video = sys.argv[2] # Output filename
alpha = float(sys.argv[3])  # alpha value
_lambda = float(sys.argv[4])  # weight to displacement
process_frames(input_dir, output_video, 15, alpha, _lambda)