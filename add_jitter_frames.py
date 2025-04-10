import numpy as np
from PIL import Image 
import cv2
import os
import math
import random
import sys
import shutil
import time


def generate_random_polygon(max_radius, center_x, center_y, num_points=6):
    """
    Generates random polygon points around a given center with varying radii.
    
    Args:
        max_radius (int): Maximum radius of the polygon from the center.
        center_x (int): X-coordinate of the polygon's center.
        center_y (int): Y-coordinate of the polygon's center.
        num_points (int): Number of vertices for the polygon.

    Returns:
        np.array: Array of points forming the polygon.
    """
    points = []
    for _ in range(num_points):
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(max_radius * 0.5, max_radius)
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        points.append([x, y])
    return np.array(points, dtype=np.int32)


def add_irregular_blotches(image, num_blotches=10, max_radius=30):
    """
    Adds irregular blotch artifacts to the given image by drawing random polygons.
    
    Args:
        image (np.array): Input image in BGR format.
        num_blotches (int): Number of blotch artifacts to add.
        max_radius (int): Maximum radius for the blotch size.

    Returns:
        np.array: Image with blotch artifacts.
    """
    blotched_image = image.copy()
    height, width, _ = blotched_image.shape

    for _ in range(num_blotches):
        # Random center for the polygon
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)

        # Generate a random polygon with irregular shape
        polygon = generate_random_polygon(max_radius, center_x, center_y)

        # Random color (grayscale) for the blotch
        color = random.randint(50, 255)

        # Draw the filled polygon (blotch) on the image
        cv2.fillPoly(blotched_image, [polygon], (color, color, color))

    return blotched_image


def add_chroma_loss_artifact(frame, line_thickness=1, line_gap=8, alpha=1):
     height, width, _ = frame.shape

    # Create a copy to modify
     artifact_frame = frame.copy()

    # Colors: Cyan, Magenta, Green (in BGR format)
     colors = [(255, 255, 0), (255, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]  # Cyan, Magenta, Green
     random.shuffle(colors)
    # Add tint to rows at regular intervals 
     random_offset = random.randint(0, line_gap-1)
    #  print(random_offset)
     i = random_offset
     while i < height-line_gap: 
        color = colors[(i // (line_gap + line_thickness)) % 3]  # Cycle through colors

        # Create a solid-colored strip matching the row width
        tint_strip = np.full((line_thickness, width, 3), color, dtype=np.uint8)

        # Blend the tint with the original frame rows using weighted sum
        artifact_frame[i:i + line_thickness] = cv2.addWeighted(
            frame[i:i + line_thickness], 1 - alpha,  # Original frame contribution
            tint_strip, alpha,  # Tint contribution
            0  # Scalar offset
        )  
        i += line_gap + random.randint(0, 5)
     return artifact_frame

def add_jitter_frame(frame, max_jitter=8, jitter_probability=1):
    rows, cols, C = frame.shape
    jittered_frame = np.zeros((rows, cols+2*max_jitter, C))
    for i in range(rows):
        if np.random.rand() < jitter_probability:  # Apply jitter with a certain probability
            dx = np.random.randint(-max_jitter, max_jitter)
            jittered_frame[i, max_jitter+dx:max_jitter+dx+cols, :] = frame[i, :, :]
    
    return jittered_frame[:, max_jitter:max_jitter+cols, :].astype(np.uint8)


def add_jitter(input_dir_path, output_dir_path, max_frames=500, line_gap=8):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    files = sorted(os.listdir(input_dir_path))
    count = 0
    for filename in files:
        print(f"Processing frame: {count}", end="\r")
        count += 1
        if count > max_frames:
            break
        if filename.endswith('.jpg') or filename.endswith('.png'): 
            inp_path = os.path.join(input_dir_path, filename)
            out_path = os.path.join(output_dir_path, filename)
            frame = cv2.imread(inp_path) 
            frame_jittered = add_chroma_loss_artifact(add_jitter_frame(frame), 1, line_gap)
            cv2.imwrite(out_path, frame_jittered)

max_frames = 500
input_dir_path = sys.argv[1]
output_dir_path = sys.argv[2]
line_gap = int(sys.argv[3])
jittered_frame = add_jitter(input_dir_path, output_dir_path, max_frames, line_gap)
print(f'Output directory name: {output_dir_path}')