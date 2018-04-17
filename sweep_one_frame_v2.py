import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection 

plt.figure()
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("# pixels")

realframe = cv2.imread('actualframe.png')
tinyfish  = cv2.imread('tinyfish.png') 

fishchannels = cv2.split(tinyfish)
colors = ('b', 'g', 'r')
height, width, channels = realframe.shape

for (chan, color) in zip(fishchannels, colors):
    gt_hist, gt_bins = np.histogram(chan, bins = 100, range = [0, 256])
    plt.plot(gt_hist, color = color, linewidth = 2.0)
    plt.xlim([0, 256])

plt.show()

print("Intersection: ")
intersection = histogram_intersection(gt_hist, gt_hist)
print(str(intersection))