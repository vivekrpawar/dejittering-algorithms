import cv2
import os
import numpy as np
import sys

def remove_jiiter_rgb(image, M=15):
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


    for i in range(1, H):
        min_L = 10**20
        min_k = 0
        for k in range(1, 2*N+2):
            # print(f"{np.zeros((1, k-1)).shape} {gama[i,:].reshape(1, W-2*N).shape} {np.zeros((1, 2*N-k+1)).shape}")
            hk = np.hstack((np.zeros((1, k-1)), gama[i,:].reshape(1, W-2*N), np.zeros((1, 2*N-k+1))))[0,:]
            m = max(k, max(p1, p0))
            n = min(k, max(p1, p0)) + W-1
            L = (1/(n-m+1)) * np.sum(np.abs(hk-2*phi1+phi2)**alpha)
            if L < min_L:
                min_k = k
                min_L = L
        # print(f' min_k {min_k} min_L {min_L}')
        p0 = p1
        p1 = min_k
        phi2 = phi1
        phi1 = np.hstack((np.zeros((1, p1-1)), gama[i,:].reshape(1, W-2*N), np.zeros((1, 2*N-p1+1))))[0,:]
        f[i,:, :] = np.hstack((np.zeros((1, p1-1, C)), g[i,:, :].reshape(1, W, C), np.zeros((1, 2*N-p1+1, C))))[0,:, :]

    corrected_image = f[:,N:N+W, :]

    if corrected_image.dtype != np.uint8:
        corrected_image = cv2.convertScaleAbs(corrected_image)

    return corrected_image

input_file = sys.argv[1]
output_file = sys.argv[2]

print(f'input file {input_file}')
print(f'output file {output_file}')
corrected_image = cv2.imread(input_file) 
# Remove jitter from the image
corrected_image = remove_jiiter_rgb(corrected_image, 15)

# Save the corrected image
cv2.imwrite(output_file, corrected_image)


