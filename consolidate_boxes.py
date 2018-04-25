import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection 

def consolidate_rectangles(arr):
    # arr is an array of coordinates 
    # do some shit here 
    return 0

realframe = cv2.imread('actualframe.png')
tinyfish  = cv2.imread('tinyfish.png') 
gray = cv2.cvtColor(tinyfish, cv2.COLOR_BGR2GRAY) 
tinyfish = gray

fishchannels  = cv2.split(tinyfish)
colors = ('b', 'g', 'r')
height, width, channels = realframe.shape 

rect_coords = {} 
num_fish_found = 0

# Histogram of the tiny fish 
for (chan, color) in zip(fishchannels, colors):
    gt_hist, gt_bins = np.histogram(chan, bins = 100, range = [0, 256])

for y in range(0, height-32, 5):
    for x in range(0, width-38, 5):
        #print("Height: "+str(height)+"Width: "+str(width))
        testimg   = realframe[y:y+32, x:x+38, :]
        testchans = cv2.split(testimg) 

        for (chan, color) in zip(testchans, colors):
            test_hist, test_bins = np.histogram(chan, bins = 100, range = [0, 256])
        
        intersection = histogram_intersection(gt_hist, test_hist)

        if (intersection >= 0.8):
            num_fish_found = num_fish_found + 1;
            print("Fish found at: "+str(x)+" , "+str(y))
            cv2.rectangle(realframe,(x,y),(x+40, y+40), (0,255,0),3)
            rect_coords[num_fish_found] = 
cv2.imshow("Frame", realframe)
cv2.waitKey(0)