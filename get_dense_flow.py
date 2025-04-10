import cv2
import numpy as np
import sys
from utils import flow_viz
def viz(img, flo, out_path): 
     
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    print(f"Saving the file", end='\r')
    cv2.imwrite(out_path, img_flo)
    
prev_frame = cv2.imread(sys.argv[1])
curr_frame = cv2.imread(sys.argv[2])

# Convert frames to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

# Compute dense optical flow using Farneback
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                    None, 0.5, 3, 15, 3, 5, 1.2, 0) 
# Visualize the flow
viz(prev_frame, flow, sys.argv[3])
print("Done.")
