import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2
# import imutils 
# Possible imports I might need:
# from __future__ import print_function
# import datetime

'''
Helpful Hist. Oriented Grad. Links:
https://www.learnopencv.com/histogram-of-oriented-gradients/

'''
realframe = cv2.imread('actualframe.png')
fish1     = cv2.imread('fesh1.png')
fish2     = cv2.imread('fesh2.png')
fish3     = cv2.imread('fesh3.png')

fchan1 = cv2.split(fish1)
fchan2 = cv2.split(fish2)
fchan3 = cv2.split(fish3)
colors = ('b', 'g', 'r')

height, width, channels = realframe.shape 
h1,     w1,    ch1      = fish1.shape

print("Height of Frame: "+str(height)+"  Width of Frame: "+str(width))
print("Height of Fish1: "+str(h1)+"  Width of Fish1: "+str(w1))

# Step 1: PREPROCESSING 
# Use Sobel Operator with Kernel size 1
fish1 = np.float32(fish1) / 255.0
realframe = np.float32(realframe) / 255.0

# Calculate gradient 
# [ 1 0 ] --> vertical edge-like regions 
# [ 0 1 ] --> horizontal edge-like regions 
gx = cv2.Sobel(fish1, cv2.CV_32F, 1, 0, ksize = 1)
gy = cv2.Sobel(fish1, cv2.CV_32F, 0, 1, ksize = 1) 
gxframe = cv2.Sobel(realframe, cv2.CV_32F, 1, 0, ksize = 1)
gyframe = cv2.Sobel(realframe, cv2.CV_32F, 0, 1, ksize = 1)

# Take abs val and convert to uint_8 
sobelX = np.uint8(np.absolute(gx))
sobelY = np.uint8(np.absolute(gy)) 
sobelXframe = np.uint8(np.absolute(gxframe))
sobelYframe = np.uint8(np.absolute(gyframe))

sobelCombined = cv2.bitwise_or(sobelX, sobelY) 
sobelframeCombined = cv2.bitwise_or(sobelXframe, sobelYframe)

cv2.imshow("Frame", realframe)
cv2.imshow("Sobel X", sobelXframe)
cv2.imshow("Sobel Y", sobelYframe)
cv2.imshow("Sobel Combined", sobelframeCombined)

# Find mag and dir of gradient  (in degrees)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
'''
NOTE: Angles are between 0-180. Called "Unsigned" gradients 
'''

# Making a HOG Descriptor 
# FROM: https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True 

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
descriptor = hog.compute(fish1)

# Image, Win-Stride, Padding, Scale, Mean-Shift 
# Win-Stride: step size in x and y of sliding window 
# Padding: controls the amount of pixels the ROI is padded w/ prior
#          to HOG feature vector extraction
# Mean-Shift: Specified if we want to apply mean-shift grouping to 
#             the detected bounding boxes 
cv2.waitKey(0)