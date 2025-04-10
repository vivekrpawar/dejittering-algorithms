import numpy as np
import sys
import cv2
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor
from numba import njit 
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

@njit
def compute_intensity_diff(row1, row2, shift1, shift2, max_shift, alpha):
    W = row1.shape[0] - 2 * max_shift
    diff = 0.0
    for i in range(W):
        a = row1[max_shift + shift1 + i]
        b = row2[max_shift + shift2 + i]
        diff += abs(a - b) ** alpha
    return diff

def remove_line_jitter(image, max_shift, alpha=2, penalty_lambda=0):
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
        # print(f"Processing row {i}/{H-1}", end="\r")
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                if cost[i-1][s_prev + max_shift] == float('inf'):
                    continue  # Skip invalid predecessors
                
                # Calculate cost from i-1
                prev_cost = cost[i-1][s_prev + max_shift]
                diff_i_minus_1 = compute_intensity_diff(padded_frame[i], padded_frame[i-1], s, s_prev, max_shift, alpha)
                
                # Include cost from i-2 if available
                if i >= 2:
                    s_prev_prev = shifts[i-1][s_prev + max_shift]
                    diff_i_minus_2 = compute_intensity_diff(padded_frame[i], padded_frame[i-2], s, s_prev_prev, max_shift, alpha)
                    diff_total = 0.7 * diff_i_minus_1 + 0.3 * diff_i_minus_2   
                else:
                    diff_total = diff_i_minus_1
                 
                shift_change_penalty = penalty_lambda * abs(s - s_prev)
                current_cost = prev_cost + diff_total + shift_change_penalty
                
                if current_cost < cost[i][s + max_shift]:
                    cost[i][s + max_shift] = current_cost
                    shifts[i][s + max_shift] = s_prev   
     
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


def process_single_frame(image_file, max_shift, alpha, _lambda):
    input_image = cv2.imread(image_file)
    corrected_image = remove_line_jitter(input_image, max_shift, alpha, _lambda)
    corrected_image, _ = detect_and_repair_flicker_rows(corrected_image)
    return corrected_image

def process_frames(input_dir, output_video, max_shift, alpha=0.7, _lambda=0):
    if(not os.path.exists(output_video)):
        os.makedirs(output_video) 
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))  
    if not image_files:
        print(f"No images found in directory {input_dir}")
        return
    print(f'output directory {output_video}') 
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    size = (width, height)
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(f'{output_video}.mp4', fourcc, 30.0, size) 
    total_frames = len(image_files)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_frame, image_file, max_shift, alpha, _lambda)
                for image_file in image_files]

        for count, future in enumerate(futures):
            print(f'Processing frame {count}/{total_frames}', end="\r", flush=True)
            corrected_image = future.result()
            cv2.imwrite(f'{output_video}/corrected_{count:06d}.png', corrected_image)
            out.write(corrected_image)

    out.release()
    print(f'Video saved as {output_video}')

input_dir = sys.argv[1] # Input directory
output_video = sys.argv[2] # Output filename 

start_time = time.time()
process_frames(input_dir, output_video, 15)
end_time = time.time()
total_time_in_ms = (end_time - start_time) * 1000

print(f'Input file directory {input_dir}')
print(f'Output video directory {output_video}')
print(f'Processing completed. Output saved to {output_video}.mp4')


print(f'Total time taken: {total_time_in_ms:.2f} ms')