'''
Edge Detection:
Mathematical methods to find points in an image
where the brightness of pixel intensities change
distinctly. 

1st find gradient of grayscale image, allowing us 
to find edge-like regions in the image. 

Canny Edge Detection: noise reduction (blurring), 
finding gradient of the image, non-maximum suppression, and
hysteresis thresholding. 
'''
import numpy as np
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"]) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imshow("Original", image) 
# Compute gradient magnitude image
# cv2.CV_64F is the data type 
lap = cv2.Laplacian(image, cv2.CV_64F)
# Take the absolute value of the gradient image
# and convert it back to an 8bit uint 
lab = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap) 
cv2.waitKey(0) 

# Sobel gradient representation 
# Allows you to find both horizontal and vertical edge-like regions 
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1) 
# [ 1 0 ] --> vertical edge-like regions 
# [ 0 1 ] --> horizontal edge-like regions 

# Take abs value and convert it to 8bit uint 
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY)) 

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY) 
cv2.imshow("Sobel Combined", sobelCombined) 
cv2.waitKey(0) 



