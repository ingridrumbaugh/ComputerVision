import matplotlib
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 
'''
Some code taken from: 
https://stackoverflow.com/questions/43099734/combining-cv2-imshow-with-matplotlib-plt-show-in-real-time

More code taken from: 
https://stackoverflow.com/questions/38636520/histogram-of-my-cam-in-real-time
'''
cv2.namedWindow('colorhist') 

hist_height = 64
hist_width  = 256
nbins       = 32
bin_width = hist_width/nbins 

cameraWidth = 320
cameraHeight = 240
'''
fig = plt.figure()
plt.ion()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 
'''

cap = cv2.VideoCapture(0)
cap.set(3, cameraWidth)
cap.set(4, cameraHeight) 

mask = np.zeros((cameraHeight, cameraWidth), np.uint8)
cv2.circle(mask,(cameraWidth/2, cameraHeight/2), 50, 255, -1) 

# empty histogram 
h = np.zeros((hist_height, hist_width))

bins = np.arange(nbins, dtype = np.int32).reshape(nbins, 1) 

while(True):
    (grabbed, frame) = cap.read() 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    hist_temp = cv2.calcHist([hsv],[0],mask,[nbins],[2, 245]) 
    cv2.normalize(hist_temp, hist_temp, hist_height, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_temp))
    pts = np.column_stack((bins, hist)) 
    
    for x,y in enumerate(hist):
        cv2.rectangle(h,(x*bin_width,y),(x*bin_width+bin_width+1,hist_height),(255),-1)
    h = np.flipud(h)
    cv2.imshow('Color Histogram', h) 
    h = np.zeros((hist_height,hist_width)) 
    
    frame = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow('image', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()








