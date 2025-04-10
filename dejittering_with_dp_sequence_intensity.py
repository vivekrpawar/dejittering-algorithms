import cv2
import os
import numpy as np
import sys

# corrected_image = remove_line_jitter(corrected_image, 30, lambda x, s: np.sum(np.abs(x - np.roll(x, s))), lambda x: np.abs(x))
def intensity_diff(x, y, neighbors_rows, s, s_prev, max_shift, alpha, lambada): 
    s_max = max(s, s_prev) # Maximum shift
    row_length = len(x) - 2 * max_shift # Length of the row after removing padding
    valid_length = row_length - 2*s_max
    x_shifted = np.roll(x, s)
    y_shifted = np.roll(y, s_prev)
    x_shifted = x_shifted[max_shift+s_max:max_shift+row_length-s_max]
    y_shifted = y_shifted[max_shift+s_max:max_shift+row_length-s_max]  
    L_neighbors = 0
    for neighbor_row in neighbors_rows: 
        neighbor_shifted = neighbor_row[max_shift+s_max:max_shift+row_length-s_max]
        # print(f'x_shifted: {x_shifted.shape} neighbours: {neighbor_shifted .shape}')
        L_neighbors += 1/ valid_length * np.sum(np.abs(x_shifted - neighbor_shifted )**alpha)
    
    return 1/ valid_length * np.sum(np.abs(x_shifted - y_shifted)**alpha) + lambada*L_neighbors
    # print(f's: {s} s_prev: {s_prev} {row_length-2*s_max}')
    # return 1/(row_length-2*s_max) * (np.sum(np.abs(np.roll(x, s) - np.roll(y, s_prev))**alpha))
    # return 1/(row_length-2*s_max) * (np.sum(np.abs(np.roll(x, s)[s_max:row_length-s_max] - np.roll(y, s_prev)[s_max:row_length-s_max])**alpha))

def remove_line_jitter(image, max_shift, intensity_diff, neighbors, penalty, alpha=0.5, _lambda=0, lambada=0.5):
    # Convert the image to grayscale
    input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f'image size: {input_frame.shape}')
    H, W = input_frame.shape

    # Padding the input image with zeros
    frame = np.zeros((H, W + 2 * max_shift))
    frame[:, max_shift:W + max_shift] = input_frame
    neighbors_frame = []
    for neighbor in neighbors:
        input_neighbor_frame = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
        neighbor_frame = np.zeros((H, W + 2 * max_shift))
        neighbor_frame[:, max_shift:W + max_shift] = input_neighbor_frame
        neighbors_frame.append(neighbor_frame)
    
    cost = [[float('inf')] * (2 * max_shift + 1) for _ in range(H)]
    shifts = [[0] * (2 * max_shift + 1) for _ in range(H)]
    
    neighbors_rows = []
    for neighbor_frame in neighbors_frame: 
        neighbors_rows.append(neighbor_frame[0, :])
    # Initialize cost for the first line
    for s in range(-max_shift, max_shift + 1): 
        cost[0][s + max_shift] = intensity_diff(frame[0], frame[0], neighbors_rows, s, s, max_shift, alpha, lambada)
    
    # Fill the DP table
    for i in range(1, H):
        output_msg = f'Processing row {i}/{H-1}'
        print(output_msg, end="", flush=True)
        neighbors_rows = []
        for neighbor_frame in neighbors_frame:
            neighbors_rows.append(neighbor_frame[i, :])
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                shift_cost = cost[i-1][s_prev + max_shift] + _lambda * penalty(s - s_prev)
                current_cost = shift_cost + intensity_diff(frame[i], frame[i-1], neighbors_rows, s, s_prev, max_shift, alpha, lambada)
                if current_cost < cost[i][s + max_shift]:
                    cost[i][s + max_shift] = current_cost
                    shifts[i][s + max_shift] = s_prev
        print("\r" + " " * len(output_msg), end="", flush=True)
        print("\r", end="", flush=True)

    # Backtrack to find optimal shifts
    corrected_shifts = [0] * H
    s_opt = min(range(-max_shift, max_shift + 1), key=lambda s: cost[H-1][s + max_shift])
    corrected_shifts[H-1] = s_opt
    for i in range(H-2, -1, -1):
        corrected_shifts[i] = shifts[i+1][corrected_shifts[i+1] + max_shift]

    # Apply shifts to the frame
    corrected_frame = np.zeros_like(image)
    for i in range(H):
        for c in range(3):  # Assuming the image has 3 color channels
            corrected_frame[i, :, c] = np.roll(image[i, :, c], corrected_shifts[i]) # Shift the row by the optimal shift
    return corrected_frame

# Get the input and output directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the input directory
filelist = os.listdir(input_dir)
filelist = sorted(filelist)
listlen = len(filelist)

w = 5

penalty = lambda x: x*x

for i in range(w, listlen-w+1):
    print(f'Processing image {i} of {listlen}')
    file_path = os.path.join(input_dir, filelist[i])
    image1 = cv2.imread(file_path)
    neighbors = []
    for j in range(i-w, i+w):
        if j != i:
            neighbors.append(cv2.imread(os.path.join(input_dir, filelist[j])))
    processed_image = remove_line_jitter(image1, 15, intensity_diff, neighbors, penalty)
    output_path = os.path.join(output_dir, filelist[i]) 
    cv2.imwrite(output_path, processed_image)

print('Processing complete')