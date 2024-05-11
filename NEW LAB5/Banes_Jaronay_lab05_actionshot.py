import cv2
import numpy as np

def stitch_images(image1, image2, output_name):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the feature detector and extractor (e.g., SIFT)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize the feature matcher using brute-force matching
    bf = cv2.BFMatcher()

    # Match the descriptors using brute-force matching
    matches = bf.match(descriptors1, descriptors2)

    # Select the top N matches
    num_matches = 50
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    # Extract matching keypoints
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estimate the homography matrix
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp the first image using the homography
    result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

    # Blending the warped image with the second image using alpha blending
    alpha = 0.6  # blending factor
    blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

    # Save the blended image
    cv2.imwrite(output_name, blended_image)

# Load the images
image1 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\frame_0.jpg')
image2 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\frame_15.jpg')
image3 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\frame_30.jpg')
image4 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\frame_45.jpg')
image5 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\frame_60.jpg')

# Stitch image1 and image2
stitch_images(image1, image2, 'Stitched1.jpg')

# Stitch Stitched1 and image3
stitch_images(cv2.imread('Stitched1.jpg'), image3, 'Stitched2.jpg')

# Stitch Stitched2 and image4
stitch_images(cv2.imread('Stitched2.jpg'), image4, 'Stitched3.jpg')

# Stitch Stitched3 and image5
stitch_images(cv2.imread('Stitched3.jpg'), image5, 'Final_Stitched_Image.jpg')

# Apply histogram equalization to the final stitched image
final_stitched_image = cv2.imread('Final_Stitched_Image.jpg')

# Save the equalized image
cv2.imwrite('Banes_Jaronay_lab05_actionshot.jpg', final_stitched_image)
