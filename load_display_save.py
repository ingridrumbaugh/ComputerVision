import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args()) 

image = cv2.imread(args["image"]) 
print "width: %d pixels" % (image.shape[1])
print "height: %d pixels" % (image.shape[0])
print "channels: %d" % (image.shape[2]) 
# Display image 
cv2.imshow("Image", image)
cv2.waitKey(0) 
# Write to a new image 
# Can use this to convert file types 
cv2.imwrite("newimage.jpg", image) 

'''
NOTE: when writing matricies in Numpy, args 
is # Columns, # Rows
'''
