import numpy as np
import argparse
import time 
import cv2 

ap = argparse.ArgumentParser() 
ap.add_argument("-i", "--image", help = "Path to the image file") 
args = vars(ap.parse_args())

# Define lower and upper limit for color you want to track 
blueLower = np.array([100, 67, 0], dtype = "uint8")
blueUpper = np.array([255, 128, 50], dtype = "uint8") 

frame = cv2.imread(args["image"]) 
cv2.imshow("Original", frame) 
cv2.waitKey(0) 

blue = cv2.inRange(frame, blueLower, blueUpper)
blue = cv2.GaussianBlur(blue, (3, 3), 0) 

# Find contours of thresholded image 
(_, cnts, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 


# Check to make sure contours were found 
if len(cnts) > 0:
    cnt = sorted(cnts, key = cv2.contourArea, reverse = True) [0]
    rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
    cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2) 

cv2.imshow("Tracking", frame)
cv2.imshow("Binary", blue) 
cv2.waitKey(0) 
