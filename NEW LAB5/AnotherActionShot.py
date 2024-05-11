import cv2
import numpy as np
import os

# Path to the directory containing the images
image_dir = 'captured_frames2'

# Function to load images from directory
def load_images(image_dir):
    images = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            images.append(img)
    return images

# Load images from directory
images = load_images(image_dir)

# Ensure images are of the same height
max_height = max(img.shape[0] for img in images)
images = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in images]

# Create a black canvas with same size as the stitched image
stitched_image = np.zeros((max_height, sum(img.shape[1] for img in images), 3), dtype=np.uint8)

# Start with the first image on the left
x_offset = 0
for img in images:
    # Place the current image on the canvas at the appropriate offset
    stitched_image[:, x_offset:x_offset+img.shape[1]] = img
    x_offset += img.shape[1]  # Move the offset to the right for the next image

# Apply seamless blending
for i in range(len(images)-1):
    mask = np.zeros(images[i].shape[:2], dtype=np.uint8)
    mask[:, -50:] = 255  # Example mask: blend last 50 pixels of each image
    blended = cv2.seamlessClone(images[i], stitched_image, mask, (int(stitched_image.shape[1]/2), int(stitched_image.shape[0]/2)), cv2.NORMAL_CLONE)
    stitched_image[:, :images[i].shape[1]] = blended[:, :images[i].shape[1]]
stitched_image = cv2.resize(stitched_image,(800,600))
# Display stitched image
cv2.imshow('Seamless Stitch', stitched_image)
cv2.imwrite('Banes_Jaronay_lab05_actionshot.jpg', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()