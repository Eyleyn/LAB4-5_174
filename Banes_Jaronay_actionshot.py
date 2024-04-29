import cv2
import numpy as np

def find_matches(base_image, sec_image):
    sift = cv2.SIFT_create()
    base_image_kp, base_image_des = sift.detectAndCompute(cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY), None)
    sec_image_kp, sec_image_des = sift.detectAndCompute(cv2.cvtColor(sec_image, cv2.COLOR_BGR2GRAY), None)

    bf_matcher = cv2.BFMatcher()
    initial_matches = bf_matcher.knnMatch(base_image_des, sec_image_des, k=2)

    good_matches = []
    for m, n in initial_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    base_image_pts = np.float32([base_image_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sec_image_pts = np.float32([sec_image_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, _ = cv2.findHomography(sec_image_pts, base_image_pts, cv2.RANSAC, 5.0)

    return homography_matrix

def stitch_images(base_image, sec_image, homography_matrix):
    height, width = base_image.shape[:2]
    sec_image_transformed = cv2.warpPerspective(sec_image, homography_matrix, (width, height))
    mask = np.where(sec_image_transformed != 0, 255, 0).astype(np.uint8)
    stitched_image = cv2.addWeighted(base_image, 0.5, sec_image_transformed, 0.5, 0)
    return stitched_image, mask

if __name__ == "__main__":
    image_paths = [
        'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames\\frame_0.jpg',
        'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames\\frame_15.jpg',
        'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames\\frame_30.jpg',
        'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames\\frame_45.jpg',
        'C:\\Users\\User\\Desktop\\KOMSAI\\UP202004905\\4TH_YEAR_NOTES\\2nd_SEM\\CMSC 174 Sec 2\\LAB\\lab05\\captured_frames\\frame_60.jpg',
        ]

    images = [cv2.imread(image_path) for image_path in image_paths]

    # Initialize the base image as the first image
    base_image = images[0]

    for i in range(1, len(images)):
        homography_matrix = find_matches(base_image, images[i])
        stitched_image, _ = stitch_images(base_image, images[i], homography_matrix)
        base_image = stitched_image

    #stitched_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("Banes_Jaronay_lab05_actionshot.png", stitched_image)
