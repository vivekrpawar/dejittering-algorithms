import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import time


def ccre(I1, I2):
    """
    CCRE: Cross Cumulative Residual Entropy between two images (I1, I2).
    Optimized using PyTorch on GPU.
    """
    # Ensure images are same size
    if I1.shape != I2.shape:
        raise ValueError("The two images must have the same dimensions.")
    
    # Flatten images for histogram computation
    I1 = I1.view(-1)
    I2 = I2.view(-1)
    
    # Get min/max intensity values
    min1, max1 = I1.min(), I1.max()
    min2, max2 = I2.min(), I2.max()
    
    # Number of intensity levels
    N1 = int(max1 - min1 + 1)
    N2 = int(max2 - min2 + 1)
    
    # Convert intensities to indices
    indices1 = (I1 - min1).long()
    indices2 = (I2 - min2).long()
    
    # Create a flat joint histogram
    flat_joint_hist = torch.zeros(N1 * N2, dtype=torch.float32, device=I1.device)
    
    # Compute flattened indices
    flat_indices = indices1 * N2 + indices2
    
    # Accumulate counts into the flat histogram
    flat_joint_hist.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=flat_joint_hist.dtype))
    
    # Reshape back to 2D histogram
    joint_hist = flat_joint_hist.reshape(N1, N2)
    
    # Normalize to probabilities
    joint_prob = joint_hist / joint_hist.sum()
    marginal_prob1 = joint_prob.sum(dim=1)
    marginal_prob2 = joint_prob.sum(dim=0)
    
    # Compute marginal cumulative distribution
    cumulative_prob1 = torch.cumsum(marginal_prob1, dim=0)
    denominator = cumulative_prob1.unsqueeze(1) * marginal_prob2.unsqueeze(0)
    
    # Compute log term safely
    valid = (joint_prob > 0) & (denominator > 0)
    log_term = torch.zeros_like(joint_prob)
    log_term[valid] = torch.log2(joint_prob[valid] / denominator[valid])
    
    # Final entropy calculation
    return -torch.sum(joint_prob * log_term).item()

def compute_intensity_diff(x, y, s, s_prev, max_shift, alpha):
    s = int(s)
    s_prev = int(s_prev)
    s_max = max(s, s_prev)
    row_length = len(x) - 2 * max_shift
    valid_length = row_length - 2 * s_max

    x_shifted = torch.roll(x, s)
    y_shifted = torch.roll(y, s_prev)

    x_shifted = x_shifted[max_shift + s_max:max_shift + row_length + s_max]
    y_shifted = y_shifted[max_shift + s_max:max_shift + row_length + s_max]

    return ccre(x_shifted, y_shifted)

def remove_line_jitter(image, max_shift, alpha=2, penalty_lambda=0):
    input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_frame = torch.tensor(input_frame, dtype=torch.float32, device='cuda')

    H, W = input_frame.shape
    C = 3

    padded_frame = torch.zeros((H, W + 2 * max_shift), device='cuda')
    padded_frame[:, max_shift:W + max_shift] = input_frame

    cost = torch.full((H, 2 * max_shift + 1), float('inf'), device='cuda')
    shifts = torch.zeros((H, 2 * max_shift + 1), dtype=torch.int32, device='cuda')

    cost[0, :] = 0

    for i in range(1, H):
        print(f"Processing row {i}/{H-1}", end="\r")
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                prev_cost = cost[i - 1, s_prev + max_shift]
                diff_i_minus_1 = compute_intensity_diff(padded_frame[i], padded_frame[i - 1], s, s_prev, max_shift, alpha)

                diff_total = diff_i_minus_1
                if i >= 2:
                    s_prev_prev = shifts[i - 1, s_prev + max_shift]
                    diff_i_minus_2 = compute_intensity_diff(padded_frame[i], padded_frame[i - 2], s, s_prev_prev, max_shift, alpha)
                    diff_total = 0.7 * diff_i_minus_1 + 0.3 * diff_i_minus_2

                shift_change_penalty = penalty_lambda * abs(s - s_prev)
                current_cost = prev_cost + diff_total + shift_change_penalty

                if current_cost < cost[i, s + max_shift]:
                    cost[i, s + max_shift] = current_cost
                    shifts[i, s + max_shift] = s_prev

    corrected_shifts = torch.zeros(H, dtype=torch.int32, device='cuda')
    corrected_shifts[H - 1] = torch.argmin(cost[H - 1]) - max_shift

    for i in range(H - 2, -1, -1):
        corrected_shifts[i] = shifts[i + 1, corrected_shifts[i + 1] + max_shift]

    corrected_frame = torch.zeros((H, W + 2 * max_shift, C), dtype=torch.float32, device='cuda')
    for i in range(H):
        shift = corrected_shifts[i].item()
        corrected_frame[i, max_shift + shift:max_shift + shift + W, :] = torch.tensor(image[i], device='cuda')

    return corrected_frame[:, max_shift:W + max_shift, :].cpu().numpy().astype(np.uint8)


input_file = sys.argv[1]
output_file = sys.argv[2]  
print(f'input file {input_file}')
print(f'output file {output_file}')
print(input_file)
print(output_file)
input_image = cv2.imread(input_file)
input_image = cv2.rotate(input_image, cv2.ROTATE_180)
penalty = lambda x: x*x
start_time = time.time()
corrected_image = remove_line_jitter(input_image, 15)
# Save the corrected image
end_time = time.time()
print(f'Time taken: {(end_time - start_time)*1000} ms')
corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_180)
cv2.imwrite(output_file, corrected_image)
