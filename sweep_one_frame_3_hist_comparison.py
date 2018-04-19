import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 


def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection 

realframe = cv2.imread('actualframe.png')
fish1     = cv2.imread('fesh1.png')
fish2     = cv2.imread('fesh2.png')
fish3     = cv2.imread('fesh3.png')

fchan1 = cv2.split(fish1)
fchan2 = cv2.split(fish2)
fchan3 = cv2.split(fish3)
colors = ('b', 'g', 'r')

height, width, channels = realframe.shape 

# Histogram #1 
for (chan, color) in zip(fchan1, colors):
    gt_hist1, gt_bins1 = np.histogram(chan, bins = 100, range = [0, 256])

# Histogram #2
for (chan, color) in zip(fchan2, colors):
    gt_hist2, gt_bins2 = np.histogram(chan, bins = 100, range = [0, 256])

# Histogram #3
for (chan, color) in zip(fchan3, colors):
    gt_hist3, gt_bins3 = np.histogram(chan, bins = 100, range = [0, 256])

for y in range(0, height-40, 5):
    for x in range(0, width-40, 5):
        testimg   = realframe[y:y+40, x:x+40, :]
        testchans = cv2.split(testimg)

        for (chan, color) in zip(testchans, colors):
            test_hist, test_bins = np.histogram(chan, bins = 100, range = [0, 256])

        intersection1 = histogram_intersection(gt_hist1, test_hist)
        intersection2 = histogram_intersection(gt_hist2, test_hist)
        intersection3 = histogram_intersection(gt_hist3, test_hist)

        if ((intersection1 >= 0.75 and intersection2 >= 0.75) or (intersection1 >= 0.75 and intersection3 >= 0.75) or (intersection2 >= 0.75 and intersection3 >= 0.75)):
            print("Fish found at: "+str(x)+"  ,  "+str(y))
            cv2.rectangle(realframe,(x,y),(x+40, y+40), (0,255,0),3)

cv2.imshow("Frame", realframe)
cv2.waitKey(0)