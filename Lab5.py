#Reference: https://kediarahul.medium.com/panorama-stitching-stitch-multiple-images-using-opencv-python-c-875a1d11236d
#Based: https://kediarahul.medium.com/panorama-stitching-stitch-multiple-images-using-opencv-python-c-875a1d11236d
import os
import cv2
import math
import numpy as np

def FindMatches(BaseImage, SecImage):
    # Create SIFT detector object
    Sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the base and secondary image
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)
    
    # Create a Brute-Force Matcher object
    BF_Matcher = cv2.BFMatcher()

    # Perform K-Nearest Neighbors (KNN) matching
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Filter out good matches
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])

     # Return the good matches along with keypoints of base and secondary images
    return GoodMatches, BaseImage_kp, SecImage_kp


def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    # Initialize empty list for base and secondary image
    BaseImage_pts = []
    SecImage_pts = []

    # Iterate through matches to extract corresponding points
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Convert the lists of points to numpy arrays
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # Use RANSAC algorithm to estimate the homography matrix
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    # Return the computed homography matrix and the status
    return HomographyMatrix, Status

    
def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    # Extract height and width from the secondary image shape
    (Height, Width) = Sec_ImageShape
    
    # Define the initial matrix
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    # Calculate the final matrix
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    # Extract x, y, and c coordinates
    [x, y, c] = FinalMatrix

     # Normalize x and y coordinates
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    # Calculate the new width and height of the frame
    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]

    # Calculate the new width and height of the frame
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    # Add correction values to x and y coordinates
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])

    # Define old initial points and new final points
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    # Update the homography matrix
    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    # Return the new height, width, correction, and updated homography matrix
    return [New_Height, New_Width], Correction, HomographyMatrix


def StitchImages(BaseImage, SecImage):

    # Find matches between key points in the base and secondary images
    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage)
    # Calculate the homography matrix and its status 
    HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    # Calculate the new frame size and correction matrix 
    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage.shape[:2], BaseImage.shape[:2])

    # Warp the secondary image
    SecImage_Transformed = cv2.warpPerspective(SecImage, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    # Create a black base image with the new frame size
    BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    # Place the base image onto the black canvas at the corrected position
    BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

    # Convert the transformed secondary image to grayscale
    SecImage_Transformed_gray = cv2.cvtColor(SecImage_Transformed, cv2.COLOR_BGR2GRAY)

    # Create a binary mask 
    _, mask = cv2.threshold(SecImage_Transformed_gray, 1, 255, cv2.THRESH_BINARY)

    #Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Use the mask to blend the images
    fg = cv2.bitwise_and(SecImage_Transformed, SecImage_Transformed, mask=mask)
    bg = cv2.bitwise_and(BaseImage_Transformed, BaseImage_Transformed, mask=mask_inv)
    StitchedImage = cv2.add(fg, bg)

    # Return the final stitched image
    return StitchedImage



def Convert_xy(x, y, center, f):
    # Calculate the transformed x-coordinate 
    xt = (f * np.tan((x - center[0]) / f)) + center[0]

     # Calculate the transformed y-coordinate
    yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]
    
    # Return the transformed coordinates 
    return xt, yt


def ProjectOntoCylinder(InitialImage):
    # Get the height and width of the image
    h, w = InitialImage.shape[:2]
    # Get the center of the image
    center = [w // 2, h // 2]
    # Define focal length
    f = 1100
    
    # Initialize array
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)
    
    #Loop in every pixel of width 
    for i in range(w):
        #Loop in every pixel of height 
        for j in range(h):
            # Convert pixel coordinates to cylindrical coordinates
            xt, yt = Convert_xy(i, j, center, f)

            # Calculate the top-left pixel coordinates
            ii_tl_x = int(xt)
            ii_tl_y = int(yt)
            
            # Check if within the valid range
            if ii_tl_x >= 0 and ii_tl_x <= w - 2 and ii_tl_y >= 0 and ii_tl_y <= h - 2:
                
                # Calculate the fractional part of the coordinates 
                dx = xt - ii_tl_x
                dy = yt - ii_tl_y

                # Calculate weights for bilinear interpolation
                weight_tl = (1.0 - dx) * (1.0 - dy)
                weight_tr = dx * (1.0 - dy)
                weight_bl = (1.0 - dx) * dy
                weight_br = dx * dy

                 # Perform bilinear interpolation 
                TransformedImage[j, i] = (weight_tl * InitialImage[ii_tl_y, ii_tl_x] +
                                          weight_tr * InitialImage[ii_tl_y, ii_tl_x + 1] +
                                          weight_bl * InitialImage[ii_tl_y + 1, ii_tl_x] +
                                          weight_br * InitialImage[ii_tl_y + 1, ii_tl_x + 1])

     # Find the minimum and maximum non-zero columns in the transformed image
    min_x = np.min(np.where(TransformedImage.sum(axis=2).sum(axis=0) != 0))
    max_x = np.max(np.where(TransformedImage.sum(axis=2).sum(axis=0) != 0))

    # Crop the transformed image to remove excess black space
    TransformedImage = TransformedImage[:, min_x:max_x]
    
     # Return the transformed image along with the range of non-zero columns and the range of rows
    return TransformedImage, np.arange(min_x, max_x), np.arange(h)




def main():
    # Images directory
    image_directory = "C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data"

    # Empty image array
    image_paths = []


    for filename in os.listdir(image_directory):
        # Check if the file ends with ".jpg"
        if filename.endswith(".jpg"):
             # Get the image path
            image_path = os.path.join(image_directory, filename)
            #Append the resized image
            image_paths.append(cv2.resize(cv2.imread(image_path), (800, 800)))   

    # Project the first image onto a cylinder
    BaseImage, _, _ = ProjectOntoCylinder(image_paths[0])

    #Iterate all the images expect the first
    for i in range(1, len(image_paths)):

        # Stitch the current image with the base image
        StitchedImage = StitchImages(BaseImage, image_paths[i])

        # The stitched image will be the new based image
        BaseImage = StitchedImage.copy()    

    # Resize the final stitched image
    final_stitched_image = cv2.resize(BaseImage, (800, 800))
    
    # Write the final stitched image to a file
    cv2.imwrite("Banes_Jaronay_lab05_stitch.png", final_stitched_image)

if __name__ == "__main__":
    main()