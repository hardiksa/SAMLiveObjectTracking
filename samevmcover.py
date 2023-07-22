import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt

# Initialize the SAM model
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Open the video feed
cap = cv2.VideoCapture("https://t1.arcischain.io:8443/live/5/index.m3u8")

# Define the coordinates of interest
x_interest = 500
y_interest = 50

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # If the frame was read successfully, segment it and display the result
    if ret:
        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Generate masks for the frame
        results = mask_generator.generate(frame_rgb)

        # Initialize a list to store the segmentations of interest
        segmentations_of_interest = []

        for i, result in enumerate(results):
            # Get the bounding box of the result
            bbox = result['bbox']

            # Check if the coordinates of interest are within the bounding box
            if bbox[0] <= x_interest <= bbox[0] + bbox[2] and bbox[1] <= y_interest <= bbox[1] + bbox[3]:
                print(f"Result {i} bbox: {bbox}")
                segmentations_of_interest.append(result['segmentation'])

                # Display the mask of the current result
                plt.figure(figsize=(10, 10))
                plt.imshow(result['segmentation'])
                plt.title(f"Segmentation of Result {i}")
                plt.show()

    # If the frame was not read successfully, break the loop
    else:
        break

# Release the video feed
cap.release()
