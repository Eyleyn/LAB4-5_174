
#Reference: https://kediarahul.medium.com/panorama-stitching-stitch-multiple-images-using-opencv-python-c-875a1d11236d
#Based: https://kediarahul.medium.com/panorama-stitching-stitch-multiple-images-using-opencv-python-c-875a1d11236d
import os
import cv2
import math
import numpy as np

def find_matches(image1, image2):

    # Create SIFT detector object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the base and secondary image
    image1_keypoints, image1_descriptors = sift.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
    image2_keypoints, image2_descriptors = sift.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)
    
    # Create a Brute-Force Matcher object
    bf_matcher = cv2.BFMatcher()

    # Perform K-Nearest Neighbors (KNN) matching
    Initialmatches = bf_matcher.knnMatch(image1_descriptors, image2_descriptors, k=3)

    # Filter out good matches
    good_matches = []
    for matches in Initialmatches:
        m,n = matches [:2]
        if m.distance < 0.80 * n.distance:
            good_matches.append([m])

     # Return the good matches along with keypoints of base and secondary images
    return good_matches, image1_keypoints, image2_keypoints


def find_homography(matches, image1_keypoints, image2_keypoints):
    #Not sure if may reshape pa or nah
    
    # Extract keypoints list 
    image1_points = np.float32([image1_keypoints[m[0].queryIdx].pt for m in matches])
    image2_points = np.float32([image2_keypoints[m[0].trainIdx].pt for m in matches])

    # Use RANSAC algorithm to estimate the homography 
    homography, _ = cv2.findHomography(image2_points, image1_points, cv2.RANSAC, 5.0)

    # Return the computed homography  
    return homography

    
def get_new_frame_size(homography, image2, image1):
    # Extract height and width from the secondary image 
    (height, width) = image2
    
    # Define the initial matrix
    initial_matrix = np.array([[0, width - 1, width - 1, 0],
                              [0, 0, height - 1, height - 1],
                              [1, 1, 1, 1]])
    
    # Calculate the final matrix
    final_matrix = np.dot(homography, initial_matrix)

    # Extract x, y, and c coordinates
    [x, y, c] = final_matrix

     # Normalize x and y coordinates
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Calculate the minimum and maximum x and y coordinates
    min_x, new_width = int(round(min(x))), int(round(max(x)))
    min_y, new_height = int(round(min(y))), int(round(max(y)))

    correction = [0, 0]

    # Calculate the new width and height of the frame
    if min_x < 0:
        new_width -= min_x
        correction[0] = abs(min_x)
    if min_y < 0:
        new_height -= min_y
        correction[1] = abs(min_y)
    
    #Ensure that the new size is sufficient for two images
    if new_width < image1[1] + correction[0]:
        new_width = image1[1] + correction[0]
    if new_height < image1[0] + correction[1]:
        new_height = image1[0] + correction[1]

    # Add correction values to x and y coordinates
    x = np.add(x, correction[0])
    y = np.add(y, correction[1])

    # Define old initial points and new final points
    old_initial_points = np.float32([[0, 0],
                                   [width - 1, 0],
                                   [width - 1, height - 1],
                                   [0, height - 1]])
    new_final_points = np.float32(np.array([x, y]).transpose())

    # Update the homography 
    homography = cv2.getPerspectiveTransform(old_initial_points, new_final_points)
    
    # Return the new height, width, correction, and updated homography 
    return [new_height, new_width], correction, homography


def stitch_images(image1, image2):

    # Find matches between key points in the image 1 and secondary images
    matches, image1_keypoints, image2_keypoints = find_matches(image1, image2)

    # Calculate the homography
    homography = find_homography(matches, image1_keypoints, image2_keypoints)

    # Calculate the new frame size and correction, and homography  
    new_frame_size, correction, homography = get_new_frame_size(homography, image2.shape[:2], image1.shape[:2])

    # Warp the secondary image
    image2_transformed = cv2.warpPerspective(image2, homography, (new_frame_size[1], new_frame_size[0]))

    # Create a black image 1 with the new frame size
    image1_transformed = np.zeros((new_frame_size[0], new_frame_size[1], 3), dtype=np.uint8)

    # Place the image 1 onto the black canvas at the corrected position
    image1_transformed[correction[1]:correction[1]+image1.shape[0], correction[0]:correction[0]+image1.shape[1]] = image1

    # Convert the transformed secondary image to grayscale
    image2_transformed_gray = cv2.cvtColor(image2_transformed, cv2.COLOR_BGR2GRAY)

    # Create a binary mask 
    _, mask = cv2.threshold(image2_transformed_gray, 1, 255, cv2.THRESH_BINARY)

    #Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Use the mask to blend the images
    fg = cv2.bitwise_and(image2_transformed, image2_transformed, mask=mask)
    bg = cv2.bitwise_and(image1_transformed, image1_transformed, mask=mask_inv)
    
    stitched_image = cv2.add(fg, bg)

    # Return the final stitched image
    return stitched_image

def main():
    # Load the images
    image1 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\1.jpg')
    image2 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\2.jpg')
    image3 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\3.jpg')
    image4 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\4.jpg')
    image5 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\5.jpg')
    image6 = cv2.imread('C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames2\\6.jpg')

    stitched_image = stitch_images(image1, image2)
    cv2.imwrite('Seamless Stitch1.jpg', stitched_image)

    stitched_image2 = stitch_images(image3, image4)
    cv2.imwrite('Seamless Stitch2.jpg', stitched_image2)

    stitched_image3 = stitch_images(image5, image6)
    cv2.imwrite('Seamless Stitch3.jpg', stitched_image3)

    final_stitched_image = stitch_images(stitched_image, stitched_image2)
    final_stitched_image = stitch_images(final_stitched_image, stitched_image3)
        
    # Write the final stitched image to a file
    cv2.imwrite("Banes_Jaronay_lab05_actionshot.jpg", final_stitched_image)

if __name__ == "__main__":
    main()
