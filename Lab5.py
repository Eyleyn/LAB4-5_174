#Reference: https://kediarahul.medium.com/panorama-stitching-stitch-multiple-images-using-opencv-python-c-875a1d11236d
#Based: https://kediarahul.medium.com/panorama-stitching-stitch-multiple-images-using-opencv-python-c-875a1d11236d
import os
import cv2
import math
import numpy as np

def ReadImage(ImageFolderPath):
    Images = []
    if os.path.isdir(ImageFolderPath):
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x:x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
        
        for i in range(len(ImageNames_Sorted)):
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(os.path.join(ImageFolderPath, ImageName))

            Images.append(InputImage)
    
    return Images

    
def FindMatches(BaseImage, SecImage):
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])

    return GoodMatches, BaseImage_kp, SecImage_kp


def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix, Status

    
def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    (Height, Width) = Sec_ImageShape
    
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
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

    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix


def StitchImages(BaseImage, SecImage):
    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage)
    HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage.shape[:2], BaseImage.shape[:2])

    SecImage_Transformed = cv2.warpPerspective(SecImage, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

    # Create a mask to identify non-black pixels in the transformed secondary image
    SecImage_Transformed_gray = cv2.cvtColor(SecImage_Transformed, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(SecImage_Transformed_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Use the mask to blend the images
    fg = cv2.bitwise_and(SecImage_Transformed, SecImage_Transformed, mask=mask)
    bg = cv2.bitwise_and(BaseImage_Transformed, BaseImage_Transformed, mask=mask_inv)
    StitchedImage = cv2.add(fg, bg)

    return StitchedImage



def Convert_xy(x, y, center, f):
    xt = (f * np.tan((x - center[0]) / f)) + center[0]
    yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]
    return xt, yt


def ProjectOntoCylinder(InitialImage):
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1100
    
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)
    
    for i in range(w):
        for j in range(h):
            xt, yt = Convert_xy(i, j, center, f)
            ii_tl_x = int(xt)
            ii_tl_y = int(yt)
            
            if ii_tl_x >= 0 and ii_tl_x <= w - 2 and ii_tl_y >= 0 and ii_tl_y <= h - 2:
                dx = xt - ii_tl_x
                dy = yt - ii_tl_y

                weight_tl = (1.0 - dx) * (1.0 - dy)
                weight_tr = dx * (1.0 - dy)
                weight_bl = (1.0 - dx) * dy
                weight_br = dx * dy

                TransformedImage[j, i] = (weight_tl * InitialImage[ii_tl_y, ii_tl_x] +
                                          weight_tr * InitialImage[ii_tl_y, ii_tl_x + 1] +
                                          weight_bl * InitialImage[ii_tl_y + 1, ii_tl_x] +
                                          weight_br * InitialImage[ii_tl_y + 1, ii_tl_x + 1])

    min_x = np.min(np.where(TransformedImage.sum(axis=2).sum(axis=0) != 0))
    max_x = np.max(np.where(TransformedImage.sum(axis=2).sum(axis=0) != 0))

    TransformedImage = TransformedImage[:, min_x:max_x]
    
    return TransformedImage, np.arange(min_x, max_x), np.arange(h)




def main():

    image_paths = [
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\1.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\2.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\3.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\4.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\5.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\6.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\7.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\8.jpg',
    'C:\\Users\\Jewy\\Documents\\LAB4-5_174\\data\\9.jpg',
    ]

      # Define the target width and height for resizing
    target_width = 800
    target_height = 800
    
    Images = [cv2.resize(cv2.imread(image_path), (target_width, target_height)) for image_path in image_paths]
    Images[2] = cv2.resize(Images[2], (6020, 6000)) 
    
    BaseImage, _, _ = ProjectOntoCylinder(Images[0])
    for i in range(1, len(Images)):
        StitchedImage = StitchImages(BaseImage, Images[i])
        BaseImage = StitchedImage.copy()    

    # Resize the final stitched image
    final_stitched_image = cv2.resize(BaseImage, (800, 800))
    
    #smoothed_image = cv2.GaussianBlur(final_stitched_image, (5, 5), 0)
    
    cv2.imwrite("Banes_Jaronay_lab05_stitch.png", final_stitched_image)

if __name__ == "__main__":
    main()