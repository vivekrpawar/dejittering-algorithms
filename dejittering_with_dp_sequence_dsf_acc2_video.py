import numpy as np
import sys
import cv2
import os
import time
import glob
from numba import njit, prange

def color_distance(row1, row2):
    """ Compute Euclidean color distance between two rows (row-wise average) """
    return np.mean(np.linalg.norm(row1.astype(np.float32) - row2.astype(np.float32), axis=1))

def detect_and_repair_flicker_rows(frame, intensity_threshold=30, color_threshold=80):
    
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
def compute_intensity_diff_fast(x, y, s, s_prev, max_shift, alpha):
    s_max = max(s, s_prev)
    s_min = min(s, s_prev)
    row_length = len(x) - 2 * max_shift
    x_start = max_shift - s + s_max
    y_start = max_shift - s_prev + s_max
    length = row_length + s_min - s_max

    total = 0.0
    for i in range(length):
        diff = abs(x[x_start + i] - y[y_start + i])
        total += 100 * (diff ** alpha)

    return total / length 

@njit(parallel=True)
def remove_line_jitter_fast(padded_frame, H, W, max_shift, alpha, penalty_lambda):
    cost = np.full((H, 2 * max_shift + 1), np.inf)
    cost[0, :] = 0.0
    shifts = np.zeros((H, 2 * max_shift + 1), dtype=np.int32)

    for i in range(1, H):
        for s in range(-max_shift, max_shift + 1):
            s_idx = s + max_shift
            for s_prev in range(-max_shift, max_shift + 1):
                s_prev_idx = s_prev + max_shift
                if cost[i - 1, s_prev_idx] == np.inf:
                    continue

                prev_cost = cost[i - 1, s_prev_idx]
                diff_i_minus_1 = compute_intensity_diff_fast(padded_frame[i], padded_frame[i - 1],
                                                             s, s_prev, max_shift, alpha)

                if i >= 2:
                    s_prev_prev = shifts[i - 1, s_prev_idx]
                    diff_i_minus_2 = compute_intensity_diff_fast(padded_frame[i], padded_frame[i - 2],
                                                                 s, s_prev_prev, max_shift, alpha)
                    diff_total = 0.7 * diff_i_minus_1 + 0.3 * diff_i_minus_2
                else:
                    diff_total = diff_i_minus_1

                shift_penalty = penalty_lambda * abs(s - s_prev)
                current_cost = prev_cost + diff_total + shift_penalty

                if current_cost < cost[i, s_idx]:
                    cost[i, s_idx] = current_cost
                    shifts[i, s_idx] = s_prev

    corrected_shifts = np.zeros(H, dtype=np.int32)
    s_opt = np.argmin(cost[H - 1]) - max_shift
    corrected_shifts[H - 1] = s_opt

    for i in range(H - 2, -1, -1):
        corrected_shifts[i] = shifts[i + 1, corrected_shifts[i + 1] + max_shift]

    return corrected_shifts

def remove_line_jitter(image, max_shift, alpha=2, penalty_lambda=0):
    input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = input_frame.shape
    C = 3

    padded_frame = np.zeros((H, W + 2 * max_shift), dtype=np.float32)
    padded_frame[:, max_shift:W + max_shift] = input_frame.astype(np.float32)

    corrected_shifts = remove_line_jitter_fast(padded_frame, H, W, max_shift, alpha, penalty_lambda)

    corrected_frame = np.zeros((H, W + 2 * max_shift, C), dtype=np.uint8)
    for i in range(H):
        shift = corrected_shifts[i]
        start = max_shift + shift
        end = start + W
        corrected_frame[i, start:end, :] = image[i, :end - start, :]

    return corrected_frame[:, max_shift:W + max_shift, :]

def process_frames(input_dir, output_video, max_shift, alpha=2, _lambda=0):
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
    count = 0
    total_frames = len(image_files)
    for image_file in image_files:
        out_msg = f'Processing frame {count}/{total_frames}'
        print(out_msg, end="\n", flush=True)
        input_image = cv2.imread(image_file)
        start_time = time.time()
        corrected_image = remove_line_jitter(input_image, max_shift, alpha, _lambda)
        corrected_image, _ = detect_and_repair_flicker_rows(corrected_image)
        end_time = time.time()
        print(f'Processing time for frame {count}: {(end_time - start_time) * 1000:.2f} ms')
        cv2.imwrite(f'{output_video}/corrected_{count:06d}.png', corrected_image)
        out.write(corrected_image) 
        count += 1

    out.release()
    print(f'Video saved as {output_video}')

def process_video(video_path, output_path, max_shift=8, alpha=2, _lambda=0):
    if(not os.path.exists(output_video)):
        os.makedirs(output_video)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path+".mp4", fourcc, fps, (width, height))
    print(f"Processing video {video_path} with FPS: {fps}, Width: {width}, Height: {height} Frames {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"Output files will be saved to {output_path}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame {frame_idx}", end = '\r', flush=True)
        # start = time.time()
        corrected_frame = remove_line_jitter(frame, max_shift, alpha, _lambda)
        corrected_frame, _ = detect_and_repair_flicker_rows(corrected_frame)
        out.write(corrected_frame)
        # print(f"Frame {frame_idx} processed in {(time.time() - start) * 1000:.2f} ms", end="\r", flush=True)
        cv2.imwrite(f'{output_video}/corrected_{frame_idx:06d}.png', corrected_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved corrected video to {output_path}")

input_video = sys.argv[1] # Input filename
output_video = sys.argv[2] # Output filename 
alpha = float(sys.argv[3]) # Alpha value
max_shift = int(sys.argv[4]) # Max shift value
start_time = time.time()
process_video(input_video, output_video, max_shift=max_shift, alpha=alpha, _lambda=0)
end_time = time.time()
total_time_in_ms = (end_time - start_time) * 1000

print(f'Input video directory {input_video}')
print(f'Output directory directory {output_video}')
print(f'Processing completed. Output saved to {output_video}.mp4')


print(f'Total time taken: {total_time_in_ms:.2f} ms')