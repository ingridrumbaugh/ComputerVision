# Fishtracker imports 
import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 
# CV Bridge / ROS imports 
import roslib 
roslib.load_manifest('ingrid_fishtracker') 
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError 


class image_converter:

	def __init__(self):
		self.image_pub = rospy.Publisher("fishes", Image)

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback) 

	def callback(self, data):
		# try/catch to catch conversion errors 
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e) 

		# Put fish tracker code here: 

		# Import frame with archerfish 
		archfish = cv2.imread('fish3.png') 

		# x and y coords where archerfish is in image 
		tempx = 690
		tempy = 370 

		# crop just the archerfish 
		croparchfish = archfish[tempy:tempy+80, tempx:tempx+140, :]

		# split ground truth image into channels 
		ogchans = cv2.split(croparchfish)
		colors = ('b', 'g', 'r') 
		# Loop over each of  the channels in the image 
		# For each channel compute a histogram 
		for (chan, color) in zip(ogchans, colors):
		    # oghist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		    gt_hist, gt_bins = np.histogram(chan, bins = 100, range = [0,256])
		    plt.plot(gt_hist, color = color, linewidth = 2.0) 
		    plt.xlim([0, 256]) 

		def histogram_intersection(hist_1, hist_2):
		    minima = np.minimum(hist_1,hist_2)
		    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
		    return intersection 

		# for the active histogram of small rectangle in webcam frame 
		fig2 = plt.figure()
		plt.ion()
		plt.title("'Flattened' Color Histogram")
		plt.xlabel("Bins")
		plt.ylabel("# of Pixels") 

		# Capture a frame from webcam video 
		cap = cv2.VideoCapture(0)
		camwidth  = cap.get(3) # float  
		camheight = cap.get(4) # float 

		# Background subtractor 
		fgbg = cv2.createBackgroundSubtractorMOG2()
		isFish = True

		# For each frame while webcam is active 
		while(True):
		    # redraw the canvas 
		    fig2.canvas.draw()
		    # convert canvas to image 
		    img = np.fromstring(fig2.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
		    img = img.reshape(fig2.canvas.get_width_height()[::-1]+(3,))
		    # img is rgb, conver to opencv bgr 
		    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		    ret, frame = cap.read() 
		    height, width, channels = frame.shape
		    # dimensions for cropping the image 
		    starty = height/8
		    endy   = (7*height)/8
		    startx = 0
		    endx   = width 
		    # Show both the cropped version of the frame
		    cropped = frame[starty:endy, startx:endx]
		    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) 
		    # blur the image first 
		    gray = cv2.GaussianBlur(gray, (5, 5), 0) 
		    
		    # black out everything except things that are moving 
		    fgmask = fgbg.apply(cropped) 
		    finalframe = cv2.bitwise_and(cropped, cropped, mask = fgmask)
		    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 
		    chans = cv2.split(finalframe)

		    for y in range(0, 30, height):
		        for x in range(0, 30, width):
		            testimg = finalframe[y:y+80,x:x+140,:]
		            testchans = cv2.split(testimg)
		            for (chan, color) in zip(chans, colors):
		                test_hist, test_bins = np.histogram(chan, bins = 100, range = [0,256])
		            intersection = histogram_intersection(gt_hist, test_hist)
		            print("Intersection Value: "+str(intersection))
		            if (intersection >= 0.7):
		                isFish = True
		                cv2.imshow("Fish Found", testimg)
		                print("Fish found!")

		    cv2.imshow("Frame", finalframe) 
    
			# Draw the histogram 
			plt.clf()
			if cv2.waitKey(10) & 0xFF == ord('q'):
            	break 

            	cap.release()
            	cv2.destroyAllWindows()

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e) 

def main(args):
	ic = image_converter()
	rospy.init_node('fish_image_converter', anonymous = True) 
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down") 
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main(sys.argv) 