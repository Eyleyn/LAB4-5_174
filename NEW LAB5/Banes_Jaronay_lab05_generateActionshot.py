import cv2
import os

def capture_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize variables
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = fps // 2  # Capture frames at every half second
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read and capture frames at specified intervals
    count = 0
    while count < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), frame)
        
        count += 1
    
    # Release video capture
    cap.release()

def main():
    # Path to the video file
    video = 'spike.mp4'

    # Output folder for captured frames
    frames_folder = 'captured_frames'

    # Capture frames at specific intervals
    capture_frames(video, frames_folder)
    