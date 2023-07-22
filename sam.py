import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt

# Initialize the SAM model
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Open the video feed
cap = cv2.VideoCapture("https://t1.arcischain.io:8443/live/5/index.m3u8")

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # If the frame was read successfully, segment it and display the result
    if ret:
        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Generate masks for the frame
        results = mask_generator.generate(frame_rgb)

        for i, result in enumerate(results):
            # print(f"Result {i} keys: {result.keys()}")
            # I want to print the area of the mask
            print(f"Result {i} bbox: {result['bbox']}")
            
            # print(f"Result {i} area keys: {result['area']}")
            # print(f"Result {i} area values: {result['area'].values()}")

        segmentations = [result['segmentation'] for result in results]

        #  Extract the masks from the results

        # Display the original frame and the masks
        plt.subplot(1, 2, 1)
        plt.imshow(frame_rgb)
        plt.title("Original Frame")

        plt.subplot(1, 2, 2)
        plt.imshow(np.sum(segmentations, axis=0))  # Sum up all masks to visualize them together
        plt.title("Segmented Masks")

        plt.show()

    # If the frame was not read successfully, break the loop
    else:
        break

# Release the video feed
cap.release()
