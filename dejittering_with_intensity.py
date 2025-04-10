import cv2
import os
import numpy as np
import sys

def remove_jiiter_rgb(image, neighbors, lambada=0.5, alpha=0.5):
    # Load the image in grayscale 

    # print(image.shape)
    H, W, C = image.shape
    M = 30
    N = M+1
    g = image
    f = np.zeros((H, W+2*N, C)) # Hx(W+2N)
    f[0,:, :] = np.hstack((np.zeros((1, N, C)), g[0,:, :].reshape(1, W, C), np.zeros((1, N, C))))[0,:, :] # Hx(W+2N) 
    prev_image = np.copy(f)

    gL = image[:, 0:N]
    gama = image[:, N: W-N, 0] + image[:, N: W-N, 1] + image[:, N: W-N, 2]
    gama_list = []
    for neighbor in neighbors:
        gama_list.append(neighbor[:, N: W-N, 0] + neighbor[:, N: W-N, 1] + neighbor[:, N: W-N, 2])
    gR = image[:, W-N:] 

    p0 = p1 = N+1
    # print(f'{np.zeros((1, N)).shape} {gama[0,:].reshape(1, W-2*N).shape} {np.zeros((1, N)).shape}')

    phi1 = phi2 = np.hstack((np.zeros((1, N)), gama[0,:].reshape(1, W-2*N), np.zeros((1, N))))[0,:] # (445,)

    # print(f'phi1.shape {phi1.shape}')
    alpha = 0.5


    for i in range(1, H):
        min_L = 10**20
        min_k = 0
        hk_list = []
        for j in range(len(neighbors)):
            hk_list.append(np.hstack((np.zeros((1, N)), gama_list[j][i,:].reshape(1, W-2*N), np.zeros((1, N))))[0,:])
        
        for k in range(1, 2*N+2):
            # print(f"{np.zeros((1, k-1)).shape} {gama[i,:].reshape(1, W-2*N).shape} {np.zeros((1, 2*N-k+1)).shape}")
            hk = np.hstack((np.zeros((1, k-1)), gama[i,:].reshape(1, W-2*N), np.zeros((1, 2*N-k+1))))[0,:]

            m = max(k, max(p1, p0))
            n = min(k, max(p1, p0)) + W-1
            L = (1/(n-m+1)) * np.sum(np.abs(hk-2*phi1+phi2)**alpha)
            L_neighbors = 0
            for h in hk_list:
                L_neighbors += (1/(n-m+1))*np.sum(np.abs(hk-h)**alpha)
            L_neighbors = L_neighbors/len(neighbors)
            L = L + lambada*L_neighbors
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

lambdas = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19]

# for lambada in lambdas:
# for lambada in lambdas:
#     for alpha in alphas:
#         for w in window_sizes:
lambada = 0.5
alpha = 0.5
w = 5
new_output_dir = os.path.join(output_dir, f'lambda_{lambada}_alpha_{alpha}_window_{w}')
if not os.path.exists(new_output_dir):
    os.makedirs(new_output_dir)
print(f'lambda {lambada} alpha {alpha} window {w} directory name {output_dir}')
for i in range(w, listlen-w+1):
    print(f'Processing image {i} of {listlen}')
    file_path = os.path.join(input_dir, filelist[i])
    image1 = cv2.imread(file_path)
    neighbors = []
    for j in range(i-w, i+w):
        if j != i:
            neighbors.append(cv2.imread(os.path.join(input_dir, filelist[j])))
    processed_image = remove_jiiter_rgb(image1, neighbors, lambada, alpha)
    output_path = os.path.join(new_output_dir, filelist[i]) 
    cv2.imwrite(output_path, processed_image)
 
print('Processing complete')