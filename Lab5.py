import cv2
import numpy as np

# Load the images
image_paths = [
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\1.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\2.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\3.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\4.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\5.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\6.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\7.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\8.jpg',
    'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\data\\9.jpg',
]

sift = cv2.SIFT_create()

stitched_image = None

for i in range(len(image_paths) - 1):
    # Load the current and next images
    image1 = cv2.imread(image_paths[i])
    image2 = cv2.imread(image_paths[i + 1])

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize the feature matcher using brute-force matching
    bf = cv2.BFMatcher()

    matches = bf.match(descriptors1, descriptors2)

    num_matches = 50
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estimate the homography matrix
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp the first image using the homography
    result = cv2.warpPerspective(image1, homography, (image2.shape[1] + image1.shape[1], image2.shape[0]))

    # Blend the warped image with the second image using alpha blending
    result[0:image2.shape[0], 0:image2.shape[1]] = image2

    if stitched_image is None:
        stitched_image = result
    else:
        stitched_image = cv2.addWeighted(stitched_image, 0.5, result, 0.5, 0)

cv2.imshow('Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
