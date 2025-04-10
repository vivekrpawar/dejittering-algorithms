import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load video frame (you can change this to read frames from a video instead)

input_file = sys.argv[1]
frame = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

# Get frame dimensions
print(frame.shape)
H, W = frame.shape

# Step 1: Compute average intensity per row
row_means = np.mean(frame, axis=1)  # shape = (H,)

# Step 2: Compute variance per row (optional, useful for noisy flicker)
row_variances = np.var(frame, axis=1)

# Step 3: Compute difference from neighboring rows
row_diffs = np.zeros(H)

for i in range(1, H-1):
    local_avg = (row_means[i-1] + row_means[i+1]) / 2
    row_diffs[i] = abs(row_means[i] - local_avg)

# Step 4: Thresholding to detect damaged rows
# Empirical threshold; can be tuned based on content and video quality
intensity_threshold = np.mean(row_diffs) + 2 * np.std(row_diffs)

# Flickering rows will have abnormally high difference from neighbors
flicker_rows = np.where(row_diffs > intensity_threshold)[0]

print(f"Detected Flicker Rows: {flicker_rows}")
