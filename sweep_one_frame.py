import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1,hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection 

realframe = cv2.imread('actualframe.png')
tinyfish  = cv2.imread('tinyfish.png') 

fishchannels  = cv2.split(tinyfish)
colors = ('b', 'g', 'r')
height, width, channels = realframe.shape 

plt.figure()
plt.ion()
plt.title("Ground Truth Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 

# Histogram of the tiny fish 
for (chan, color) in zip(fishchannels, colors):
    #ground_truth_histogram = cv2.calcHist([chan], [0], None, [256], [0,256]) 
    gt_hist, gt_bins = np.histogram(chan, bins = 100, range = [0, 256])
    plt.plot(gt_hist, color = color, linewidth = 2.0)
    plt.xlim([0,256])

for y in range(0, height-32, 5):
    for x in range(0, width-38, 5):
        print("Height: "+str(height)+"Width: "+str(width))
        testimg   = realframe[y:y+32, x:x+38, :]
        testchans = cv2.split(testimg) 

        for (chan, color) in zip(testchans, colors):
            test_hist, test_bins = np.histogram(chan, bins = 100, range = [0, 256])
        
        intersection = histogram_intersection(gt_hist, test_hist)
        print("Intersection Value: "+str(intersection))
        print("X: "+str(x)+"Y: "+str(y))
        
        if (intersection >= 0.99):
            print("Fish found at: "+str(x)+" , "+str(y))
            cv2.rectangle(realframe,(x,y),(y+32, x+38), (0,255,0),3)


#cv2.imshow("Frame", realframe)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()