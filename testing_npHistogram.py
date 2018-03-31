import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

'''
Some code taken from: 
https://stackoverflow.com/questions/43099734/combining-cv2-imshow-with-matplotlib-plt-show-in-real-time
As well as:
https://github.com/mpatacchiola/deepgaze/blob/master/examples/ex_color_classification_images/ex_histogram_intersection.py
'''

def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1,hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection 

# Import frame with archerfish 
archfish = cv2.imread('fish3.png') 

# x and y coords where archerfish is in image 
tempx = 690
tempy = 370 

# crop just the archerfish 
croparchfish = archfish[tempy:tempy+80, tempx:tempx+140, :]
#cv2.imshow("Archer Fish", croparchfish)
#cv2.waitKey(0)

# split ground truth image into channels 
ogchans = cv2.split(croparchfish)
colors = ('b', 'g', 'r') 
'''
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 
'''

# Loop over each of  the channels in the image 
# For each channel compute a histogram 
for (chan, color) in zip(ogchans, colors):
    #print(chan.shape)
    #                     channels, mask, size, ranges,
    # oghist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    gt_hist, gt_bins = np.histogram(chan, bins = 100, range = [0,256])
    #plt.plot(gt_hist, color = color, linewidth = 2.0) 
    #plt.xlim([0, 256]) 

#plt.show()

intersection = histogram_intersection(gt_hist, gt_hist)
print("Intersection Value: "+str(intersection))

#cv2.waitKey(0) 

