import numpy as np
import sys
import cv2
import os
import glob
import numba as nb
import time

@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.int64, nb.int64, nb.int64, nb.int64), nogil=True)
def compute_intensity_diff_numba(x, y, s, s_prev, max_shift, alpha):
    len_x = x.shape[0]
    row_length = len_x - 2 * max_shift
    s_max = max(abs(s), abs(s_prev))
    
    valid_length = row_length - 2 * s_max
    if valid_length <= 0:
        return np.inf

    start = max_shift + s_max
    end = start + valid_length
    
    x_start = start - s
    x_end = end - s
    y_start = start - s_prev
    y_end = end - s_prev

    x_slice = x[x_start:x_end]
    y_slice = y[y_start:y_end]

    if x_slice.shape != y_slice.shape:
        return np.inf

    diff = np.abs(x_slice - y_slice)
    return np.sum(100 * (diff ** alpha)) / valid_length

@nb.njit(nb.types.Tuple((nb.float64[:,:], nb.int64[:,:]))(nb.float64[:,:], nb.int64, nb.int64, nb.int64, nb.int64, nb.int64), 
          parallel=True, nogil=True, cache=True)
def compute_dp_cost(padded_frame, max_shift, alpha, penalty_lambda, H, num_shifts):
    cost = np.full((H, num_shifts), np.inf)
    shifts = np.zeros((H, num_shifts), dtype=np.int64)
    cost[0, :] = 0.0  # Initialize first row

    for i in nb.prange(1, H):
        for s_idx in range(num_shifts):
            s = s_idx - max_shift
            min_cost = np.inf
            best_prev = 0
            
            for prev_s_idx in range(num_shifts):
                prev_cost = cost[i-1, prev_s_idx]
                if prev_cost == np.inf:
                    continue
                
                prev_s = prev_s_idx - max_shift
                diff = compute_intensity_diff_numba(
                    padded_frame[i], padded_frame[i-1], s, prev_s, max_shift, alpha
                )
                
                if diff == np.inf:
                    continue
                
                penalty = penalty_lambda * abs(s - prev_s)
                total_cost = prev_cost + diff + penalty
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_prev = prev_s_idx
            
            if min_cost != np.inf:
                cost[i, s_idx] = min_cost
                shifts[i, s_idx] = best_prev
                
    return cost, shifts

def remove_line_jitter(image, max_shift=15, alpha=2, penalty_lambda=0):
    input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W = input_frame.shape
    C = 3
    
    # Create padded array with explicit typing
    padded_frame = np.zeros((H, W + 2 * max_shift), dtype=np.float64)
    padded_frame[:, max_shift:W + max_shift] = input_frame
    num_shifts = 2 * max_shift + 1
    
    # Ensure input types match Numba signatures
    cost, shifts = compute_dp_cost(
        padded_frame.copy(),
        np.int64(max_shift),
        np.int64(alpha),
        np.int64(penalty_lambda),
        np.int64(H),
        np.int64(num_shifts))
    
    # Rest of the function remains the same...
    corrected_shifts = np.zeros(H, dtype=np.int64)
    s_opt = np.argmin(cost[H-1]) - max_shift
    corrected_shifts[H-1] = s_opt
    
    for i in range(H-2, -1, -1):
        corrected_shifts[i] = shifts[i+1, corrected_shifts[i+1] + max_shift]
    
    corrected_frame = np.zeros((H, W + 2 * max_shift, C), dtype=np.uint8)
    for i in range(H):
        shift = corrected_shifts[i]
        start = max_shift + shift
        end = start + W
        start = max(start, 0)
        end = min(end, W + 2 * max_shift)
        corrected_frame[i, start:end, :] = image[i, :end - start, :]
    
    return corrected_frame[:, max_shift:W + max_shift, :].astype(np.uint8)

def detect_and_repair_flicker_rows(frame, intensity_threshold=30, color_threshold=50):
    H, W, C = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_intensity = np.mean(gray_frame, axis=1)
    
    # Vectorized anomaly detection
    current_intensity = row_intensity[1:-1]
    prev_intensity = row_intensity[:-2]
    next_intensity = row_intensity[2:]
    expected_intensity = (prev_intensity + next_intensity) / 2
    intensity_anomaly = np.abs(current_intensity - expected_intensity) > intensity_threshold
    
    current_rows = frame[1:-1].astype(np.float32)
    prev_rows = frame[:-2].astype(np.float32)
    next_rows = frame[2:].astype(np.float32)
    
    color_dist_prev = np.mean(np.linalg.norm(current_rows - prev_rows, axis=2), axis=1)
    color_dist_next = np.mean(np.linalg.norm(current_rows - next_rows, axis=2), axis=1)
    color_anomaly = (color_dist_prev > color_threshold) | (color_dist_next > color_threshold)
    
    flicker_mask = intensity_anomaly | color_anomaly
    flicker_rows = np.where(flicker_mask)[0] + 1  # Adjust indices
    
    # Corrected post-filtering
    if len(flicker_rows) > 0:
        # Calculate differences between consecutive elements
        diffs = np.diff(flicker_rows, prepend=flicker_rows[0]-2, append=flicker_rows[-1]+2)
        
        # Identify isolated rows (not part of 3+ consecutive rows)
        isolated = np.ones(len(flicker_rows), dtype=bool)
        for i in range(1, len(flicker_rows)-1):
            if diffs[i] == 1 and diffs[i+1] == 1:
                isolated[i-1:i+2] = False
        actual_flicker_rows = flicker_rows[isolated]
    else:
        actual_flicker_rows = np.array([], dtype=int)
    
    # Repair logic remains the same
    repaired_frame = frame.copy()
    if actual_flicker_rows.size > 0:
        upper = frame[actual_flicker_rows - 1].astype(np.float32)
        lower = frame[actual_flicker_rows + 1].astype(np.float32)
        upper_gray = gray_frame[actual_flicker_rows - 1, :].astype(np.float32)
        lower_gray = gray_frame[actual_flicker_rows + 1, :].astype(np.float32)
        current_gray = gray_frame[actual_flicker_rows, :].astype(np.float32)
        
        upper_edge = np.abs(upper_gray - current_gray)
        lower_edge = np.abs(lower_gray - current_gray)
        total_edge = upper_edge + lower_edge + 1e-6
        
        upper_weight = (lower_edge / total_edge)[..., np.newaxis]
        lower_weight = (upper_edge / total_edge)[..., np.newaxis]
        interpolated = (upper * upper_weight + lower * lower_weight).astype(np.uint8)
        repaired_frame[actual_flicker_rows] = interpolated
    
    return repaired_frame, actual_flicker_rows.tolist()

def process_frames(input_dir, output_video, max_shift, alpha=0.7, _lambda=0):
    os.makedirs(output_video, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    first_image = cv2.imread(image_files[0])
    H, W, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{output_video}.mp4', fourcc, 30.0, (W, H))
    
    total_frames = len(image_files)
    for count, image_file in enumerate(image_files):
        print(f'Processing frame {count+1}/{total_frames}', end='\r')
        input_image = cv2.imread(image_file)
        corrected_image = remove_line_jitter(input_image, max_shift, alpha, _lambda)
        corrected_image, _ = detect_and_repair_flicker_rows(corrected_image)
        cv2.imwrite(os.path.join(output_video, f'corrected_{count:06d}.png'), corrected_image)
        out.write(corrected_image)
    
    out.release()
    print(f'\nVideo saved to {output_video}.mp4')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_dir> <output_video>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_video = sys.argv[2]
    
    start_time = time.time()
    process_frames(input_dir, output_video, 15)
    total_time = (time.time() - start_time) * 1000
    print(f'Total processing time: {total_time:.2f} ms')