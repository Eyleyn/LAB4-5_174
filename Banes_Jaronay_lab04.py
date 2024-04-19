import cv2
from matplotlib import pyplot as plt
import imutils
import os
import numpy as np

def process_images(image):
    
    #Converts image from RGB to LAB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    #Converts image to grayscale
    gray_image = image[:,:,1]

    #Blur the image
    blurred_gray_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

    #Perfrom thresholding to the the blurred image
    (T, bottle_threshold) = cv2.threshold(blurred_gray_image,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #Create rectangle structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    #Morph the image
    morph_image = cv2.morphologyEx(bottle_threshold, cv2.MORPH_OPEN, kernel)

    #Find the contour in the image
    find_contours = cv2.findContours(morph_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Extract contours
    contours = imutils.grab_contours(find_contours)



    #Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    #Computes the heigth of the largest contour 
    x, y, w, h = cv2.boundingRect(largest_contour)

   
    
    return h

def mapped_height(amount_directories):
    height_mapping = {}
    for directory in amount_directories:
        #Empty height list
        heights = []
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (500, 500))
            h = process_images(image)
            #Append the h(height) to the list
            heights.append(h)
        avg_height = sum(heights) / len(heights)
        height_mapping[avg_height] = os.path.basename(directory)
    return height_mapping

def guess_image(guess_directories, height_mapping):
    for directory in guess_directories:
        heights = []
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (500, 500))
            h = process_images(image)
            #Append the h(height) to the list
            heights.append(h)
        avg_height = sum(heights) / len(heights)

            # Perform interpolation
        heights = list(height_mapping.keys())
        labels = list(height_mapping.values())
        
        # Sort labels based on heights
        sorted_labels = [x for _,x in sorted(zip(heights, labels))]
        
        # Find the two closest heights for interpolation
        idx = np.argsort(heights)
        closest_idx = np.searchsorted(heights, avg_height, side="left", sorter=idx)
        if closest_idx == 0:
            interp_idx = [closest_idx, closest_idx + 1]
        elif closest_idx == len(heights):
            interp_idx = [closest_idx - 1, closest_idx]
        else:
            interp_idx = [closest_idx - 1, closest_idx]
        
        # Perform ordinal interpolation
        interp_labels = sorted_labels[interp_idx[0]:interp_idx[1]+1]
        ordinal_values = np.arange(len(interp_labels))
        predicted_label = np.interp(avg_height, [heights[interp_idx[0]], heights[interp_idx[1]]], ordinal_values)
        
        # Map the interpolated ordinal value back to the original label
        predicted_label = interp_labels[int(round(predicted_label))]
        
        print(f"Guessed amount for {directory}: {predicted_label}")


def main():

    amount_directories = [
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\50ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\100ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\150ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\200ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\250ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\300ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\350ml',
    ]
    height_mapping = mapped_height (amount_directories)

    guess_directories = [
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\guess\\A',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\guess\\B',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\guess\\C',
    ]

    guess_image (guess_directories, height_mapping)

if __name__ == "__main__":
    main()
