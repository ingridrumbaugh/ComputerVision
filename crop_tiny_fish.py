import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

newtinyfish = cv2.imread('newtinyfish.png')
#height, width, channels = newtinyfish.shape
newtinyfish2 = cv2.imread('newtinyfish2.png')
newtinyfish3 = cv2.imread('tinyfish3.png')

# height, width, channels = newtinyfish3.shape
# print("Height: "+str(height)+"  Width: "+str(width))

tiny3 = newtinyfish3[23:63, 37:77]
#cv2.imshow("third tineh feesh", tiny3)
cv2.imwrite('fesh3.png', tiny3)

tiny2 = newtinyfish2[0:40, 9:49]
#cv2.imshow("New tineh feesh", tiny2)
cv2.imwrite('fesh2.png', tiny2)

tiny1 = newtinyfish[1:41, 19:59]
# cv2.imshow("Tineh fesh", tiny1)
cv2.imwrite('fesh1.png', tiny1)

cv2.waitKey(0)