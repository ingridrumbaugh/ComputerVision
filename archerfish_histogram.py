from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"]) 
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
#blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Original", image) 

'''
# Use adaptive thresholding with gaussian methods for archer fish 
thresh = cv2.adaptiveThreshold(blurred, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)

cv2.waitKey(0) 
'''
# Split the image into 3 channels: RGB 
chans = cv2.split(image)
colors = ('b', 'g', 'r') 

plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 
features = [] 

# Loop over each of  the channels in the image 
# For each channel compute a histogram 
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist) 
    plt.plot(hist, color = color) 
    plt.xlim([0, 256]) 
    
plt.show()
cv2.waitKey(0)   
