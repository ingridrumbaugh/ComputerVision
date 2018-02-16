import numpy as np
import cv2 

# Init image
# Construct array w 300 rows and 300 cols
# Allocate 3 channels (RGB) + UINT8 --> 255 RGB Value is 1 Byte 
canvas = np.zeros((300, 300, 3), dtype = "uint8")

# Define colors 
green = (0, 255, 0)
# Draw a green line from 0,0 to 300, 300
cv2.line(canvas, (0, 0), (300, 300), green) 
cv2.imshow("Canvas", canvas) 
cv2.waitKey(0) 

red = (0, 0, 255)
# Draw red line w/ thickness 3 pixels 
cv2.line(canvas, (300, 0), (0, 300), red, 3) 
cv2.imshow("Canvas", canvas) 
cv2.waitKey(0) 

# What to draw on, start X/Y, end X/Y, color 
cv2.rectangle(canvas, (10, 10), (60, 60), green) 
cv2.imshow("Canvas", canvas) 
cv2.waitKey(0) 

cv2.rectangle(canvas, (50, 200), (200, 225), red, 5) 
cv2.imshow("Canvas", canvas) 
cv2.waitKey(0) 

blue = (255, 0, 0) 
# Negative thickness fills in the rectangle 
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv2.imshow("Canvas", canvas) 
cv2.waitKey(0) 

# Start with a blank canvas again 
canvas = np.zeros((300, 300, 3), dtype = "uint8")
# Calculate center 
(centerX, centerY) = (canvas.shape[1] / 2, canvas.shape[0] / 2) 
white = (255, 255, 255) 
# Start at 0, to 175 increment 25
# Draws concentric circles 
for r in xrange(0, 175, 25):
    cv2.circle(canvas, (centerX, centerY), r, white) 

cv2.imshow("Canvas", canvas) 
cv2.waitKey(0) 

# Abstract drawing 
for i in xrange(0, 25):
    radius = np.random.randint(5, high = 200)
    color = np.random.randint(0, high = 256, size = (3,)).tolist() 
    pt = np.random.randint(0, high = 300, size = (2,))
    
    cv2.circle(canvas, tuple(pt), radius, color, -1) 

cv2.imshow("Canvas", canvas)
cv2.waitKey(0) 

