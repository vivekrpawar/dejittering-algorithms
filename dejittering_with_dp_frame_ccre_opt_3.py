import numpy as np
import sys
import cv2
import time 
from collections import defaultdict
import numpy as np
def ccre(I1, I2):
    """
    CCRE: Cross Cumulative Residual Entropy between two images (I1, I2).
    Optimized using numpy vectorization.

    Reference:
    - Wang, F., & Vemuri, B. C. (2007). Non-rigid multi-modal image registration using cross-cumulative residual entropy.
      International Journal of Computer Vision.
    - Hasan, M., Pickering, M. R., & Jia, X. (2012). Robust Automatic Registration of Multimodal Satellite Images Using
      CCRE With Partial Volume Interpolation.
    """
    # Ensure images are same size
    if I1.shape != I2.shape:
        raise ValueError("The two images must have the same dimensions.")

    # Flatten images for histogram computation
    I1 = I1.ravel()
    I2 = I2.ravel()

    # Get min/max intensity values
    min1, max1 = I1.min(), I1.max()
    min2, max2 = I2.min(), I2.max()

    # Number of intensity levels
    N1 = int(max1 - min1 + 1)
    N2 = int(max2 - min2 + 1)

    # Compute joint histogram directly using numpy
    indices1 = (I1 - min1).astype(int)
    indices2 = (I2 - min2).astype(int)
    joint_hist = np.zeros((N1, N2), dtype=np.float64)

    # Vectorized histogram calculation
    np.add.at(joint_hist, (indices1, indices2), 1)

    # Normalize histograms to probabilities
    joint_prob = joint_hist / joint_hist.sum()
    marginal_prob1 = joint_prob.sum(axis=1)
    marginal_prob2 = joint_prob.sum(axis=0)

    # Compute marginal cumulative distribution (c1)
    cumulative_prob1 = np.cumsum(marginal_prob1)

    # Compute CCRE directly (vectorized form)
    denominator = cumulative_prob1[:, None] * marginal_prob2[None, :]
    epsilon = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        valid = (joint_prob > 0) & (denominator > 0)
        log_term = np.zeros_like(joint_prob)
        log_term[valid] = np.log2(joint_prob[valid] / (denominator[valid] + epsilon))

    # Final entropy calculation
    ccre_value = -np.sum(joint_prob * log_term)
    return ccre_value

def compute_intensity_diff(x, y, s, s_prev, max_shift):
    s_max = max(s, s_prev) # Maximum shift
    s_min = min(s, s_prev)
    # s_max = abs(s_prev - s)
    row_length = len(x) - 2 * max_shift # Length of the row after removing padding 
    x_shifted = np.roll(x, s)
    y_shifted = np.roll(y, s_prev)
    x_shifted = x_shifted[max_shift+s_max:max_shift+row_length+s_min] 
    y_shifted = y_shifted[max_shift+s_max:max_shift+row_length+s_min]
    return ccre(x_shifted, y_shifted)

def remove_line_jitter(image, max_shift):
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
                diff_i_minus_1 = compute_intensity_diff(padded_frame[i], padded_frame[i-1], s, s_prev, max_shift)
                
                # Include cost from i-2 if available
                if i >= 2:
                    s_prev_prev = shifts[i-1][s_prev + max_shift]
                    # Modified
                    #diff_i_minus_2 = compute_intensity_diff(padded_frame[i], padded_frame[i-2], s, s_prev_prev, max_shift, alpha)
                    # diff_total = 0.7 * diff_i_minus_1 + 0.3 * diff_i_minus_2   # Modified
                    diff_total = diff_i_minus_1
                else:
                    diff_total = diff_i_minus_1
                 
                #shift_change_penalty = penalty_lambda * abs(s - s_prev) # Modified
                current_cost = prev_cost + diff_total #+ shift_change_penalty # Modified
                
                if current_cost < cost[i][s + max_shift]:
                    cost[i][s + max_shift] = current_cost
                    shifts[i][s + max_shift] = s_prev   
    np.save('ccre_cost.npy', cost)
    np.save('ccre_shifts.npy', shifts)
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


input_file = sys.argv[1]
output_file = sys.argv[2]  
print(f'input file {input_file}')
print(f'output file {output_file}')
print(input_file)
print(output_file)
input_image = cv2.imread(input_file) 
start_time = time.time()
corrected_image = remove_line_jitter(input_image, 15)
# Save the corrected image
end_time = time.time()
print(f'Time taken: {(end_time - start_time)*1000} ms') 
cv2.imwrite(output_file, corrected_image)