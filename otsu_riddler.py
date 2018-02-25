'''
Otsu: Automatically compute threshold value T 
Assumes there are 2 peaks in the grayscale histogram 
Then tries to find an optimal value to separate these 2 peaks
'''

import numpy as np 
import argparse
import mahotas
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0) 
cv2.imshow("Image", image) 

# OTSU 
T = mahotas.thresholding.otsu(blurred)
print "Otsu's threshold: %d" % (T) 

# Copy grayscale image 
thresh = image.copy()
thresh[thresh > T] = 255
thresh[threst < T] = 0 
# Invert threshold: Equiv to cv2.THRESH_BINARY_INV
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh) 

# RIDDLER-CALVARD 
T = mahotas.thresholding.rc(blurred)
print "Riddler-Calvard: %d" % (T) 

thresh = image.copy() 
thresh[thresh > T] = 255
thresh[thresh < T] = 0
thresh = cv2.bitwise_not(thresh) 
cv2.imshow("Riddler-Calvard", thresh) 
cv2.waitKey(0) 
