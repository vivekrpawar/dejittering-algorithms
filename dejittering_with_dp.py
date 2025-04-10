def remove_line_jitter(frame, max_shift, intensity_diff, penalty):
    H, W = frame.shape
    cost = [[float('inf')] * (2 * max_shift + 1) for _ in range(H)]
    shifts = [[0] * (2 * max_shift + 1) for _ in range(H)]
    
    # Initialize cost for the first line
    for s in range(-max_shift, max_shift + 1):
        cost[0][s + max_shift] = intensity_diff(frame[0], s)
    
    # Fill the DP table
    for i in range(1, H):
        for s in range(-max_shift, max_shift + 1):
            for s_prev in range(-max_shift, max_shift + 1):
                shift_cost = cost[i-1][s_prev + max_shift] + penalty(s - s_prev)
                current_cost = shift_cost + intensity_diff(frame[i], s)
                if current_cost < cost[i][s + max_shift]:
                    cost[i][s + max_shift] = current_cost
                    shifts[i][s + max_shift] = s_prev

    # Backtrack to find optimal shifts
    corrected_shifts = [0] * H
    s_opt = min(range(-max_shift, max_shift + 1), key=lambda s: cost[H-1][s + max_shift])
    corrected_shifts[H-1] = s_opt
    for i in range(H-2, -1, -1):
        corrected_shifts[i] = shifts[i+1][corrected_shifts[i+1] + max_shift]

    # Apply shifts to the frame
    corrected_frame = np.zeros_like(frame)
    for i in range(H):
        corrected_frame[i] = np.roll(frame[i], corrected_shifts[i])
    
    return corrected_frame
