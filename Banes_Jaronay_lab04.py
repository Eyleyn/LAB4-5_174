import cv2
from matplotlib import pyplot as plt
import imutils
import os

def process_images(image):
    gray_image = cv2.split(image)[0]
    gray_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

    (T, threshold_image) = cv2.threshold(gray_image, 25, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    morph_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(morph_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    heights = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
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