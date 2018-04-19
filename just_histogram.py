import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

realframe = cv2.imread('tinyfish.png')
framechannels = cv2.split(realframe)
height, width, channs = realframe.shape
print("Height of Tiny Fish: "+str(height)+"  Width of Tiny Fish: "+str(width))
colors = ('b', 'g', 'r')

plt.figure()
plt.title("Ground Truth Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 

for (chan, color) in zip(framechannels, colors):
    histogram = cv2.calcHist([chan], [0], None, [256], [0,256]) 
    plt.plot(histogram, color = color, linewidth = 2.0)
    plt.xlim([0,256])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()