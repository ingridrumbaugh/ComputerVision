import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

# Import frame with archerfish 
archfish = cv2.imread('fish3.png') # <-- remove background!!

# x and y coords where archerfish is in image 
tempx = 690
tempy = 370 

# crop just the archerfish 
croparchfish = archfish[tempy:tempy+80, tempx:tempx+140, :]

# convert image to grayscale to remove background 
gray = cv2.cvtColor(croparchfish, cv2.COLOR_BGR2GRAY) 
# sigma = 11 
blurred = cv2.GaussianBlur(gray, (11, 11), 0) 
# Use canny edge detection on blurred, grayscale image of archerfish
edged = cv2.Canny(blurred, 30, 150) 
cv2.imshow("Canny edge",edged)

# Returns a tuple of the contours 
# Type of contours: RETR_EXTERNAL, retrieve only outermost contours
# All contours would be: RETR_LIST 
# How to approx contour: CHAIN_APPROX_SIMPLE 
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

fish = croparchfish.copy()
cv2.drawContours(fish, cnts, -1, (0, 255, 0), 2) 
cv2.imshow("contours", fish) 
cv2.waitKey(0) 