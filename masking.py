import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"]) 
cv2.imshow("Original", image) 

# Create empty numpy array 
# Same width and height as image 

# image.shape --> COLS , ROWS 

mask = np.zeros(image.shape[:2], dtype = "uint8") 
# Find center of the image 
(cX, cY) = (image.shape[1]/2, image.shape[0]/2)
# Examine pixels only a part of the white rectangle  
cv2.rectangle(mask, (cX-75, cY-75), (cX+75, cY+75), 255, -1) 
cv2.imshow("Mask", mask) 

masked = cv2.bitwise_and(image, image, mask = mask) 
cv2.imshow("Mask applied", masked) 
cv2.waitKey(0) 

# New mask
mask = np.zeros(image.shape[:2], dtype = "uint8") 
#          mask,  center, radius, color, fill
cv2.circle(mask, (cX, cY), 100,    255,   -1)
masked = cv2.bitwise_and(image, image, mask = mask) 
cv2.imshow("Mask", mask) 
cv2.imshow("Mask applied", masked) 
cv2.waitKey(0) 
