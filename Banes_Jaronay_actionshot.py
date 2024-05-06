import cv2 
import numpy as np  
import os  

def capture_frames_and_stitch(video_path, output_path):
    # find matches between two images using SIFT features
    def find_matches(base_image, sec_image):
        # Create a SIFT detector object
        sift = cv2.SIFT_create()
        # Detect and compute keypoints and descriptors for base image
        base_image_kp, base_image_des = sift.detectAndCompute(cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY), None)
        # Detect and compute keypoints and descriptors for secondary image
        sec_image_kp, sec_image_des = sift.detectAndCompute(cv2.cvtColor(sec_image, cv2.COLOR_BGR2GRAY), None)

        # Create Brute-Force Matcher object
        bf_matcher = cv2.BFMatcher()
        # Match keypoints between base and secondary images
        initial_matches = bf_matcher.knnMatch(base_image_des, sec_image_des, k=2)

        # Select good matches based on Lowe's ratio test where the 
        # number of false matches between features are reduced
        good_matches = []
        for m, n in initial_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract keypoints corresponding to good matches
        base_image_pts = np.float32([base_image_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        sec_image_pts = np.float32([sec_image_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix using RANSAC algorithm
        homography_matrix, _ = cv2.findHomography(sec_image_pts, base_image_pts, cv2.RANSAC, 4.0)

        return homography_matrix

    # Stitch two images using a homography matrix
    def stitch_images(base_image, sec_image, homography_matrix):
        # Get dimensions of base image
        height, width = base_image.shape[:2]
        # Warp the secondary image using the homography matrix to align with base image
        sec_image_transformed = cv2.warpPerspective(sec_image, homography_matrix, (width, height))
        # Threshold the warped image to create a binary mask
        _, mask = cv2.threshold(sec_image_transformed, 1, 255, cv2.THRESH_BINARY)

        # Convert mask to uint8 type
        mask = mask.astype(np.uint8)

        alpha = 0.5  # Set blending ratio (adjust as needed)
        # Blend the base and transformed secondary images using alpha blending
        stitched_image = cv2.addWeighted(base_image, alpha, sec_image_transformed, 1 - alpha, 0)
        return stitched_image, mask

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize variables for frame count and frames per second (FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define interval for capturing frames (every half second)
    interval = fps // 2  
    
    # Create output folder for captured frames
    output_folder = 'captured_frames'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read and capture frames at specified intervals
    count = 0
    while count < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skips redundant frames based on interval
        if count % interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), frame)
        
        count += 1
    
    # Release video capture
    cap.release()

    # Define paths for captured frames folder and output action shot image
    frames_folder = 'captured_frames'
    output_image = output_path

    # Get list of frame files in the captured frames folder
    frame_files = sorted(os.listdir(frames_folder))
    
    # Read frames and stitch them together
    frames = []
    for file in frame_files:
        frame_path = os.path.join(frames_folder, file)
        frame = cv2.imread(frame_path)
        frames.append(frame)
    
    # Extract person from each frame and stitch frames
    base_image = frames[0]  # Initialize base image with the first frame
    for i in range(1, len(frames)):
        homography_matrix = find_matches(base_image, frames[i])
        stitched_image, _ = stitch_images(base_image, frames[i], homography_matrix)
        base_image = stitched_image  # Update base image for next iteration

    # Save the final stitched action shot image
    cv2.imwrite(output_image, stitched_image)
    print("Action shot image generated successfully!")

def main():
    # Define input video file path
    video_file = 'spike.mp4'

    # Define output path for action shot image
    output_image = 'Banes_Jaronay_lab05_actionshot.png'

    # Check if video file exists
    if not os.path.exists(video_file):
        print("Error: Video file not found.")
    else:
        # Capture frames and stitch them together into action shot image
        capture_frames_and_stitch(video_file, output_image)

if __name__ == "__main__":
    main()
