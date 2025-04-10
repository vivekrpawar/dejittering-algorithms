import numpy as np
import sys
import cv2
import time
import matplotlib.pyplot as plt


def intensity_diff(x, y, s, s_prev, max_shift, alpha):
    s_max = max(s, s_prev) # Maximum shift
    s_min = min(s, s_prev)
    # s_max = abs(s_prev - s)
    row_length = len(x) - 2 * max_shift # Length of the row after removing padding
    # valid_length = row_length - abs(s_prev - s)
    valid_length = row_length - 2*s_max
    x_shifted = np.roll(x, s)
    y_shifted = np.roll(y, s_prev)
    x_shifted = x_shifted[max_shift+s_max:max_shift+row_length+s_min] 
    y_shifted = y_shifted[max_shift+s_max:max_shift+row_length+s_min]
    return (np.sum(100*np.abs(x_shifted - y_shifted)**alpha))/valid_length

def remove_line_jitter(image, max_shift, intensity_diff, penalty, alpha=2, _lambda=50):
    print(f'alpha: {alpha} lambda: {_lambda}')
    input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f'image size: {input_frame.shape}')
    H, W = input_frame.shape
    C = 3
    frame = np.zeros((H, W + 2 * max_shift))
    frame[:, max_shift:W + max_shift] = input_frame
    print(f'frame size: {frame.shape}')
    cost = [[float('inf')] * (2 * max_shift + 1) for _ in range(H)]
    shifts = [[0] * (2 * max_shift + 1) for _ in range(H)]
    
    
    for s in range(-max_shift, max_shift + 1):
        cost[0][s + max_shift] = intensity_diff(frame[0], frame[0], s, s, max_shift, alpha)
        
    for i in range(1, H):
        output_msg = f'Processing row {i}/{H-1}'
        print(output_msg, end="", flush=True)
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                shift_cost = cost[i-1][s_prev + max_shift] + _lambda * penalty(s - s_prev)
                current_cost = shift_cost + intensity_diff(frame[i], frame[i-1], s, s_prev, max_shift, alpha) #+ intensity_diff(frame[i], frame[i-1], s, s_prev, max_shift, 0.5)
                if current_cost < cost[i][s + max_shift]:
                    cost[i][s + max_shift] = current_cost
                    shifts[i][s + max_shift] = s_prev
        print("\r" + " " * len(output_msg), end="", flush=True)
        print("\r", end="", flush=True)

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
    return corrected_frame[:, max_shift:W + max_shift, :]


input_file = sys.argv[1]
output_file = sys.argv[2] 
alpha = float(sys.argv[3])
_lambda = float(sys.argv[4])
print(f'input file {input_file}')
print(f'output file {output_file}')
print(input_file)
print(output_file)
input_image = cv2.imread(input_file)
input_image = cv2.rotate(input_image, cv2.ROTATE_180)
penalty = lambda x: x*x
start_time = time.time()
corrected_image = remove_line_jitter(input_image, 20, intensity_diff, penalty, alpha, _lambda)
# Save the corrected image
end_time = time.time()
print(f'Time taken: {(end_time - start_time)*1000} ms')
corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_180)
cv2.imwrite(output_file, corrected_image)
