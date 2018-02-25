'''
Canny edge detector is a multistep process 
Blur the image, compute sobel gradient, 
supress edges, then hystseresis thresholding to
determine if a pixel is an edge or not. 

Provides more crisp edges that have less noise
than the Laplacian or Sobel gradient images. 
'''

import numpy as np
import argparse 
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args())

image = cv2.imread(args["image"]) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
image = cv2.GaussianBlur(image, (5, 5), 0) 
cv2.imshow("Blurred", image) 

# blurred image, threshold1, threshold2
# any gradient > threshold2 are edges 
# any value < threshold1 not edges 
canny = cv2.Canny(image, 30, 150)
cv2.imshow("Canny", canny)
cv2.waitKey(0) 
