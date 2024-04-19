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

    #return the height    
    return h

def mapped_height(amount_directories):

    #Empty height mapping dictionary
    height_mapping = {}

    #Iterate over the amount directories
    for directory in amount_directories:

        #Empty height list
        heights = []

        #Iterate over each file in the directory
        for filename in os.listdir(directory):

            #Get the path of the image
            image_path = os.path.join(directory, filename)

            #Reads the image
            image = cv2.imread(image_path)

            #Resize the image
            image = cv2.resize(image, (500, 500))

            #Get the heigth of the liquid
            h = process_images(image)

            #Append the h(height) to the list
            heights.append(h)
        
        #Calculate the average height
        avg_height = sum(heights) / len(heights)

        # Mapped the height 
        height_mapping[avg_height] = os.path.basename(directory)
   
    #Return the height mapping
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

        # List the heights and labels
        heights_list= list(height_mapping.keys())
        labels_list = list(height_mapping.values())
        
        # Sort labels based on heights
        sorted_labels = [x for y,x in sorted(zip(heights_list, labels_list))]
        
        # Get the index of the heights list
        index = np.argsort(heights_list)

        #Find the closest index to average height
        closest_index = np.searchsorted(heights_list, avg_height, side="left", sorter=index)

        # If the average height is in 0 index
        if closest_index == 0:

            #Interpolate between the first 2 elements
            interp_index = [closest_index, closest_index + 1]

        # If the average height is in the end
        elif closest_index == len(heights_list):

            #Interpolate between the last 2 elements
            interp_index = [closest_index - 1, closest_index]
        
        # If the average height is in between elements
        else:

            # #Interpolate between the elements
            interp_index = [closest_index - 1, closest_index]
        
        #Extract the labels between the 2 elements
        interp_labels = sorted_labels[interp_index[0]:interp_index[1]+1]

        #Extract the indices of the labels
        ordinal_values = np.arange(len(interp_labels))

        #Get the interpolated ordinal value
        interp_value = np.interp(avg_height, [heights_list[interp_index[0]], heights_list[interp_index[1]]], ordinal_values)
        
        #Predict the label
        predicted_label = interp_labels[int(round(interp_value))]
        
        #Display the results
        print(f"Guessed amount for {os.path.basename(directory)}: {predicted_label}")


def main():
    #Amount directories
    amount_directories = [
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\50ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\100ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\150ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\200ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\250ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\300ml',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\350ml',
    ]

    # Get the height mapping
    height_mapping = mapped_height (amount_directories)

    # Unkown directories
    guess_directories = [
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\guess\\A',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\guess\\B',
        'C:\\Users\\Jewy\\Desktop\\CMSC 174\\lab04\\guess\\C',
    ]

    # Guess the unknown amount
    guess_image (guess_directories, height_mapping)

if __name__ == "__main__":
    main()
