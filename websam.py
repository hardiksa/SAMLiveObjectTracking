from flask import Flask, Response, render_template
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import threading
import time


app = Flask(__name__, template_folder='.')

# Initialize the SAM model
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Open the video feed
cap = cv2.VideoCapture('https://t1.arcischain.io:8443/live/5/index.m3u8')

# Global state
frame = None
segmented_frame = None

def update_frames():
    global frame, segmented_frame
    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()

        # If the frame was read successfully, segment it and display the result
        if ret:
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate masks for the frame
            results = mask_generator.generate(frame_rgb)
            segmentations = [result['segmentation'] for result in results]

            # Check if segmentations is empty
            if not segmentations:
                app.logger.error("No segmentations generated")
                continue

            # Create the segmented frame
            segmented_frame = np.sum(segmentations, axis=0)

            # Check if segmented_frame is None or empty
            if segmented_frame is None or not np.any(segmented_frame):
                app.logger.error("segmented_frame is None or empty")
        else:
            app.logger.error("Failed to read frame from video feed")

        # Sleep for approximately 8.57 seconds to process about 7 frames per minute
        time.sleep(60 / 7)
        
# Start the update_frames function in a separate thread
threading.Thread(target=update_frames).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    # Encode the frame as JPEG
    ret, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/segmented_video')
def segmented_video():
    try:
        # Check if segmented_frame is None or empty
        if segmented_frame is None or not np.any(segmented_frame):
            app.logger.error("segmented_frame is None or empty in /segmented_video")
            return Response(status=204)  # Return "No Content" status

        # Convert the segmented frame to the CV_8UC1 type
        segmented_frame_8u = cv2.convertScaleAbs(segmented_frame, alpha=(255.0/segmented_frame.max()))

        # Apply a color map to the segmented frame
        color_segmented_frame = cv2.applyColorMap(segmented_frame_8u, cv2.COLORMAP_JET)

        # Encode the color segmented frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', color_segmented_frame)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        app.logger.error(f"Error in /segmented_video: {e}")
        raise



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
