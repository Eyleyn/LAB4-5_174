import cv2
from matplotlib import pyplot as plt
import imutils
import os

def process_images(image):
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray_image = lab[:,:,1]

    blurred_gray_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

    (T, bottle_threshold) = cv2.threshold(blurred_gray_image,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph_image = cv2.morphologyEx(bottle_threshold, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(morph_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    heights = []
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    heights.append(h)
    avg_height = sum(heights) / len(heights)
    return avg_height

def mapped_height(amount_directories):
    height_mapping = {}
    for directory in amount_directories:
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (500, 500))
            avg_height = process_images(image) 
            height_mapping[avg_height] = os.path.basename(directory)
    return height_mapping

def guess_image(guess_directories, height_mapping):
    for directory in guess_directories:
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (500, 500))
        avg_height = process_images(image)
        closest_height = min(height_mapping.keys(), key=lambda x: abs(x - avg_height))
        print(f"Guessed amount for {directory}: {height_mapping[closest_height]}")

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