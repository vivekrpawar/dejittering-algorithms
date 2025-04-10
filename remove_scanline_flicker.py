import cv2
import numpy as np
import sys 

def compute_intensity(row):
    """ Grayscale intensity per row (average across channels) """
    return np.mean(row, axis=1)

def color_distance(row1, row2):
    """ Compute Euclidean color distance between two rows (row-wise average) """
    return np.mean(np.linalg.norm(row1.astype(np.float32) - row2.astype(np.float32), axis=1))

def detect_and_repair_flicker_rows(frame, intensity_threshold=30, color_threshold=50):
    """
    Detect and repair flicker rows using:
        1. Intensity consistency check (grayscale)
        2. Color consistency check (full RGB)
        3. Edge-aware interpolation for repair (vectorized)
    """
    H, W, C = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Row-wise intensity (grayscale average per row)
    row_intensity = np.mean(gray_frame, axis=1)

    repaired_frame = frame.copy()
    flicker_rows = []

    # --- Detection Phase ---
    for row in range(1, H-1):
        prev_row = frame[row-1]
        next_row = frame[row+1]
        current_row = frame[row]

        prev_row_intensity = row_intensity[row-1]
        next_row_intensity = row_intensity[row+1]
        current_row_intensity = row_intensity[row]

        expected_intensity = (prev_row_intensity + next_row_intensity) / 2

        # Grayscale intensity check (row-wise)
        intensity_anomaly = abs(current_row_intensity - expected_intensity) > intensity_threshold

        # Color distance check (row-wise)
        color_dist_prev = color_distance(current_row, prev_row)
        color_dist_next = color_distance(current_row, next_row)

        color_anomaly = (color_dist_prev > color_threshold) or (color_dist_next > color_threshold)

        if intensity_anomaly or color_anomaly:
            flicker_rows.append(row)

    # --- Flicker Post-Filtering (Skip isolated flicker rows) ---
    l = len(flicker_rows)
    actual_flicker_rows = []
    for i in range(l):
        if ((i > 1 and flicker_rows[i-1] == flicker_rows[i]-1 and flicker_rows[i-2] == flicker_rows[i]-2) or
            (i < l-2 and flicker_rows[i+1] == flicker_rows[i]+1 and flicker_rows[i+2] == flicker_rows[i]+2)):
            continue
        actual_flicker_rows.append(flicker_rows[i])

    # --- Repair Phase (Vectorized Edge-Aware Interpolation) ---
    actual_flicker_rows = np.array(actual_flicker_rows)

    if len(actual_flicker_rows) > 0:
        upper_rows = frame[actual_flicker_rows - 1].astype(np.float32)
        lower_rows = frame[actual_flicker_rows + 1].astype(np.float32)

        upper_gray = gray_frame[actual_flicker_rows - 1].astype(np.float32)
        lower_gray = gray_frame[actual_flicker_rows + 1].astype(np.float32)
        current_gray = gray_frame[actual_flicker_rows].astype(np.float32)

        # Compute edge strengths for all columns at once (broadcasted subtraction)
        upper_edge_strength = np.abs(upper_gray - current_gray)  # shape (len(flicker_rows), W)
        lower_edge_strength = np.abs(lower_gray - current_gray)

        # Total edge strength per pixel (avoid div-by-zero using epsilon)
        total_edge_strength = upper_edge_strength + lower_edge_strength + 1e-6

        # Compute weights (broadcasted division)
        upper_weight = lower_edge_strength / total_edge_strength
        lower_weight = upper_edge_strength / total_edge_strength

        # Now do weighted blending for all pixels at once (vectorized interpolation)
        interpolated_rows = (upper_weight[..., np.newaxis] * upper_rows +
                             lower_weight[..., np.newaxis] * lower_rows).astype(np.uint8)

        # Apply the interpolated rows to the repaired frame
        repaired_frame[actual_flicker_rows] = interpolated_rows

    return repaired_frame, actual_flicker_rows.tolist()


# Load frame (update filename)
filename = sys.argv[1]
frame = cv2.imread(filename) 
# Process frame
repaired_frame, flicker_rows = detect_and_repair_flicker_rows(frame)
print(f'Flicker rows cound {len(flicker_rows)} \n {flicker_rows}')
# Save result
cv2.imwrite(sys.argv[2], repaired_frame) 
