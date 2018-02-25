'''
A given pixel is turned OFF if it has a value of 0
A given pixel is turned ON if it has a value > 0 
'''

import numpy as np
import cv2 

'''
AND --> True if/only if both pixels > 0 
OR  --> True if either of two pixels are > 0 
XOR --> True if/only if EITHER of two pixels are > 0 (NOT BOTH) 
NOT --> Inverts "on" and "off" pixels 
'''

# 300 X 300 Numpy Array 
rectangle = np.zeros((300, 300), dtype = "uint8")
# Draw a 250 X 250 white rectangle @ center of image 
# Note -1 --> rectangle filled in 
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1) 
cv2.imshow("Rectangle", rectangle) 

# Init 300 X 300 Numpy Array 
circle = np.zeros((300, 300), dtype = "uint8") 
# Radius = 150 (@ center) :: also white and filled in 
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle) 

bitwiseAND = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAND)
cv2.waitKey(0) 

bitwiseOR = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOR)
cv2.waitKey(0) 

bitwiseXOR = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXOR) 
cv2.waitKey(0) 

bitwiseNOT = cv2.bitwise_not(circle) 
cv2.imshow("NOT", bitwiseNOT) 
cv2.waitKey(0) 
