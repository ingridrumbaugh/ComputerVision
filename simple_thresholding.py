import numpy as np
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# apply gaussian blur with sigma = 5 radius 
# this helps remove some of the high frequency edges 
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image) 
cv2.waitKey(0) 
# Anything greater than T is set to the max, otherwise set to 0 
#                   grayscale image, T  , max, thresholding method
(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY) 
# returns value of T and the image ^
cv2.imshow("Threshold Binary", thresh) 
cv2.waitKey(0) 
(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV) 
cv2.imshow("Threshold Binary INV", threshInv)
cv2.waitKey(0)
