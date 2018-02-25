import numpy as np
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image"0 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"])
cv2.imshow("original", image) 

# create arrays of 8bit uints 
# cv2.add takes care of clipping, ensuring that the 
# addition produces a max of 255. 
print "max of 255: "+ str(cv2.add(np.uint8([200]), np.uint8([100])))

print "min of 0: "+ str(cv2.subtract(np.uint8([50]), np.uint8([100])))

print "wrap around: "+ str(np.uint8([200]) + np.uint8([100]))
print "wrap around: "+ str(np.uint8([50]) - np.uint8([100])) 

# Increase each pixel intensity by 100, but ensuring
# that all pixels are clipped to [0, 255] 
M = np.ones(image.shape, dtype = "uint8") * 100 
added = cv2.add(image, M)
cv2.imshow("added", added) 

# Decrease each pixel intensity by 50 but ensuring
# that all pixels are clipped to [0, 255] 
M = np.ones(image.shape, dtype = "uint8") * 50
subtracted = cv2.subtract(image, M) 
cv2.imshow("subtracted", subtracted)
cv2.waitKey(0) 
