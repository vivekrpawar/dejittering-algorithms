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
    min1, max1 = int(I1.min()), int(I1.max())
    min2, max2 = int(I2.min()), int(I2.max())


    # Number of intensity levels
    N1 = int(max1 - min1 + 1)
    N2 = int(max2 - min2 + 1)

    # Compute joint histogram directly using numpy
    indices1 = (I1 - min1).astype(int)
    indices2 = (I2 - min2).astype(int)
    joint_hist = np.zeros((N1, N2), dtype=np.float64)

    # Vectorized histogram calculation
    # print(f'joint hist {joint_hist.shape} {indices1.shape} {indices1.shape}')
    np.add.at(joint_hist, (indices1, indices2), 1)

    # Normalize histograms to probabilities
    joint_prob = joint_hist / joint_hist.sum()
    marginal_prob1 = joint_prob.sum(axis=1)
    marginal_prob2 = joint_prob.sum(axis=0)

    # Compute survival function for marginal_prob1 (cumulative residual distribution)
    survival_prob1 = np.flip(np.cumsum(np.flip(marginal_prob1)))

    # Compute denominator using survival_prob1 instead of cumulative probability
    denominator = survival_prob1[:, None] * marginal_prob2[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        valid = (joint_prob > 0) & (denominator > 0)
        log_term = np.zeros_like(joint_prob)
        log_term[valid] = np.log2(joint_prob[valid] / denominator[valid])

    # Final entropy calculation
    ccre_value = -np.sum(joint_prob * log_term)
    return ccre_value


def get_cost(index1, index2, image_array, max_shift):
    vector1 = image_array[index1]
    vector2 = image_array[index2]
    alpha = 2
    n = vector1.shape[0]
    vector1 = np.pad(vector1, (max_shift, max_shift), 'constant', constant_values=0)
    vector2 = np.pad(vector2, (max_shift, max_shift), 'constant', constant_values=0) 
    cost_ccre = []  
    for shift in range(-max_shift, max_shift+1): 
        v1 = vector1[max_shift:max_shift+n]
        v2 = vector2[max_shift+shift:max_shift+shift+n]
        loss1 = ccre(v1, v2) 
        cost_ccre.append(loss1)  
    cost_ccre = np.array(cost_ccre) 
    return cost_ccre

def remove_line_jitter(img_rgb, max_shift):
    H, W, C = img_rgb.shape
    image_array = np.array(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    shifts = np.zeros(H)
    shifts[0] = max_shift
    for i in range(1, H):
        print(f'Processing row {i+1}/{H}', end='\r')
        cost_ccre = get_cost(i-1, i, image_array, max_shift) 
        shifts[i] = np.argmin(cost_ccre)
    relative_shifts = np.zeros(shifts.shape)
    for i in range(1, H):
        relative_shifts[i] = (shifts[i]-max_shift) + relative_shifts[i-1]
    mean_shift = np.round(np.mean(relative_shifts))
    inverse_shifts = -1*(relative_shifts-mean_shift)
    inverse_shifts = inverse_shifts.astype(int)
    f = np.zeros((H, W, 3))
    g = img_rgb
    for i in range(H):
        # print(f"Processing row {i+1}/{H} shift {relative_shifts[i]}")
        f[i,:, :] = np.roll(g[i, :, :], inverse_shifts[i], axis=0)
        if inverse_shifts[i] >= 0:
            f[i,:inverse_shifts[i], :] = np.zeros((1, inverse_shifts[i], C))
        else:
            f[i, inverse_shifts[i]:, :] = np.zeros((1, -1*inverse_shifts[i], C))
    corrected_image = f[:,max_shift:max_shift+W, :]
    if corrected_image.dtype != np.uint8:
            corrected_image = cv2.convertScaleAbs(corrected_image)
    return corrected_image


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
