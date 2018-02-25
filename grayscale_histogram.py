'''
HISTOGRAMS 

x-axis: 256 bins, one for each val
counting the # of times each pixel value occurs 

Using cv2.calcHist:
cv2.calcHist(images, channels, mask, histSize, ranges)

channels: List of indexes. Gray --> [0] RGB --> [0, 1, 2] 
mask:     Can compute for masked pixels only. if not, provide None 
histSize: # bins to use. IE 32 for each channel: [32, 32, 32] 
ranges:   Range of possible pixel vals. Normally [0, 256] 
'''

from matplotlib import pyplot as plt
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"]) 

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imshow("Original", image)

hist = cv2.calcHist([image], [0], None, [256], [0,256]) 

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 
plt.plot(hist) 
plt.xlim([0,256]) 
plt.show()

cv2.waitKey(0) 
