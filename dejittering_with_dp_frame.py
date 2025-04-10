import numpy as np
import sys
import cv2
import time
def remove_line_jitter(image, max_shift, intensity_diff, penalty, alpha=0.5, _lambda=0):
    # Convert the image to grayscale
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f'image size: {frame.shape}')
    H, W = frame.shape
    cost = [[float('inf')] * (2 * max_shift + 1) for _ in range(H)]
    shifts = [[0] * (2 * max_shift + 1) for _ in range(H)]
    
    # Initialize cost for the first line
    for s in range(-max_shift, max_shift + 1):
        cost[0][s + max_shift] = intensity_diff(frame[0], frame[0], s, s, alpha)
    
    # Fill the DP table
    for i in range(1, H):
        output_msg = f'Processing row {i}/{H-1}'
        print(output_msg, end="", flush=True)
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                shift_cost = cost[i-1][s_prev + max_shift] + _lambda #* penalty(s - s_prev)
                current_cost = shift_cost + intensity_diff(frame[i], frame[i-1], s, s_prev, alpha)
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

input_file = sys.argv[1]
output_file = sys.argv[2]
print(f'input file {input_file}')
print(f'output file {output_file}')
print(input_file)
print(output_file)
input_image = cv2.imread(input_file)
# corrected_image = remove_line_jitter(corrected_image, 30, lambda x, s: np.sum(np.abs(x - np.roll(x, s))), lambda x: np.abs(x))
def intensity_diff(x, y, s, s_prev, alpha):
    s_max = max(s, s_prev)
    row_length = len(x)
    # print(f's: {s} s_prev: {s_prev} {row_length-2*s_max}')
    return 1/(row_length-2*s_max) * (np.sum(np.abs(np.roll(x, s) - np.roll(y, s_prev))**alpha))
    # return 1/(row_length-2*s_max) * (np.sum(np.abs(np.roll(x, s)[s_max:row_length-s_max] - np.roll(y, s_prev)[s_max:row_length-s_max])**alpha))

penalty = lambda x: x*x
start_time = time.time()
corrected_image = remove_line_jitter(input_image, 15, intensity_diff, penalty)
# Save the corrected image
end_time = time.time()
print(f'Time taken: {(end_time - start_time)*1000} ms')
cv2.imwrite(output_file, corrected_image)
