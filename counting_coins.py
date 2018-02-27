'''
Use edges to help find actual coins in the image
and count them. 
openCV has methods to find "curves" in an image,
called contours. 

To find contours in an image, get a binarization of
the image, using edge detection or thresholding. 
'''

import numpy as np
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"]) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# sigma = 11 
blurred = cv2.GaussianBlur(gray, (11, 11), 0) 
cv2.imshow("Image", image) 

edged = cv2.Canny(blurred, 30, 150) 
cv2.imshow("Edges", edged) 

# Returns a tuple of the contours 
# Type of contours: RETR_EXTERNAL, retrieve only outermost contours
# All contours would be: RETR_LIST 
# How to approx contour: CHAIN_APPROX_SIMPLE 
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

print "There are %d coins in this image" % (len(cnts)) 

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2) 
cv2.imshow("Coins", coins) 
cv2.waitKey(0) 

# Crop each individual coin from the image 
# Below line 25 

# Loop over the contours
for (i, c) in enumerate(cnts):
    # Finds the "enclosing box" that the contour fits into
    (x, y, w, h) = cv2.boundingRect(c) 
    
    print "Coin # %d" % (i+1) 
    # Crop the coin from the image using bounding box coords 
    coin = image[y:y+h,x:x+w]
    cv2.imshow("Coin", coin) 
    # Initialize mask 
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    # Fit a circle to the contour 
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y+h, x:x+w]
    
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask))
    cv2.waitKey(0) 


