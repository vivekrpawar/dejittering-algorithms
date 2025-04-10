import cv2
import os
import numpy as np
import sys

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

    with np.errstate(divide='ignore', invalid='ignore'):
        valid = (joint_prob > 0) & (denominator > 0)
        log_term = np.zeros_like(joint_prob)
        log_term[valid] = np.log2(joint_prob[valid] / denominator[valid])

    # Final entropy calculation
    ccre_value = -np.sum(joint_prob * log_term)
    return ccre_value

def compute_intensity_diff(x, y, s, s_prev, max_shift):
    s_max = max(s, s_prev) # Maximum shift
    s_min = min(s, s_prev)
    # s_max = abs(s_prev - s)
    row_length = len(x) - 2 * max_shift # Length of the row after removing padding
    # valid_length = row_length - abs(s_prev - s)
    valid_length = row_length - 2*s_max
    x_shifted = np.roll(x, s)
    y_shifted = np.roll(y, s_prev)
    # x_shifted = x_shifted[max_shift+s_max:max_shift+row_length+s_min] 
    # y_shifted = y_shifted[max_shift+s_max:max_shift+row_length+s_min]
    x_shifted = x_shifted[max_shift+s_max:row_length] 
    y_shifted = y_shifted[max_shift+s_max:row_length]
    return ccre(x_shifted, y_shifted)


def remove_jiiter_rgb(image):
    # Load the image in grayscale 

    # print(image.shape)
    H, W, C = image.shape
    M = 15
    N = M+1
    g = image
    f = np.zeros((H, W+2*N, C)) # Hx(W+2N)
    f[0,:, :] = np.hstack((np.zeros((1, N, C)), g[0,:, :].reshape(1, W, C), np.zeros((1, N, C))))[0,:, :] # Hx(W+2N) 
    prev_image = np.copy(f)

    gL = image[:, 0:N]
    gama = image[:, N: W-N, 0] + image[:, N: W-N, 1] + image[:, N: W-N, 2]
    gR = image[:, W-N:] 

    p0 = p1 = N+1
    # print(f'{np.zeros((1, N)).shape} {gama[0,:].reshape(1, W-2*N).shape} {np.zeros((1, N)).shape}')

    phi1 = phi2 = np.hstack((np.zeros((1, N)), gama[0,:].reshape(1, W-2*N), np.zeros((1, N))))[0,:] # (445,)

    # print(f'phi1.shape {phi1.shape}')
    alpha = 0.5
    prev_k1 = 0
    prev_k2 = 0
    for i in range(1, H):
        print(f"Processing row {i+1}/{H}", end="\r")
        min_L = 10**20
        min_k = 0
        for k in range(1, 2*N+2):
            # print(f"{np.zeros((1, k-1)).shape} {gama[i,:].reshape(1, W-2*N).shape} {np.zeros((1, 2*N-k+1)).shape}")
            hk = np.hstack((np.zeros((1, k-1)), gama[i,:].reshape(1, W-2*N), np.zeros((1, 2*N-k+1))))[0,:]
            m = max(k, max(p1, p0))
            n = min(k, max(p1, p0)) + W-1
            # L = (1/(n-m+1)) * np.sum(np.abs(hk-2*phi1+phi2)**alpha) 
            # L = 0.5*ccre(hk, phi1) + 0.5*ccre(hk, phi2)
            L1 = compute_intensity_diff(gama[i-1,:], gama[i,:], prev_k1, k, N)
            if i >= 2:
                L2 = compute_intensity_diff(gama[i-2,:], gama[i,:], prev_k2, k, N)
                L = 0.5*L1 + 0.5*L2
            else:
                L = L1
            if L < min_L:
                min_k = k
                min_L = L
        # print(f' min_k {min_k} min_L {min_L}')
        p0 = p1
        p1 = min_k
        prev_k2 = prev_k1
        prev_k1 = min_k
        phi2 = phi1
        phi1 = np.hstack((np.zeros((1, p1-1)), gama[i,:].reshape(1, W-2*N), np.zeros((1, 2*N-p1+1))))[0,:]
        f[i,:, :] = np.hstack((np.zeros((1, p1-1, C)), g[i,:, :].reshape(1, W, C), np.zeros((1, 2*N-p1+1, C))))[0,:, :]

    corrected_image = f[:,N:N+W, :]

    if corrected_image.dtype != np.uint8:
        corrected_image = cv2.convertScaleAbs(corrected_image)

    return corrected_image


if __name__ == "__main__":
    input_image = sys.argv[1]
    output_image = sys.argv[2]
    image = cv2.imread(input_image)
    corrected_image = remove_jiiter_rgb(image)
    cv2.imwrite(output_image, corrected_image)
    print(f'Output image saved to {output_image}')