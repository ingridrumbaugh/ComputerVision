from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image") 
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"]) 
matrix = np.zeros((75, 75), np.float32) 
green = (0, 255, 0) 
#                  start x/y  end x/y
cv2.rectangle(image, (500, 500), (575, 575), green)
cv2.imshow("Image", image)

cv2.waitKey(0)
