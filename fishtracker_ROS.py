#!/usr/bin/env python
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
from sensor_msgs.msg import Image , CompressedImage
from cv_bridge import CvBridge, CvBridgeError 
import rospkg

class image_converter:

    def __init__(self):
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path('ingrid_fishtracker')

        self.image_pub = rospy.Publisher("fishes", Image, queue_size=1)
        self.archfish = cv2.imread(self.package_path+'/nodes/fish3.png')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback) 

                # x and y coords where archerfish is in image 
        tempx = 690
        tempy = 370 

        # crop just the archerfish 
        croparchfish = self.archfish[tempy:tempy+80, tempx:tempx+140, :]

        # split ground truth image into channels 
        ogchans = cv2.split(croparchfish)
        self.colors = ('b', 'g', 'r') 
        # Loop over each of  the channels in the image 
        # For each channel compute a histogram 
        for (chan, color) in zip(ogchans, self.colors):
            # oghist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            self.gt_hist, self.gt_bins = np.histogram(chan, bins = 100, range = [0,256])
            # plt.plot(gt_hist, color = color, linewidth = 2.0) 
            # plt.xlim([0, 256]) 
        
        # Background subtractor 
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.isFish = False

    def histogram_intersection(self,hist_1, hist_2):
        self.minima = np.minimum(hist_1,hist_2)
        self.intersection = np.true_divide(np.sum(self.minima), np.sum(hist_2))
        return self.intersection 
        
        # plt.figure()
        # for the active histogram of small rectangle in webcam frame 
        # self.fig2 = plt.figure()
        # plt.ion()
        '''
        plt.title("'Flattened' Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels") 
		'''

    def callback(self, data):
        # archfish = self.archfish
        # try/catch to catch conversion errors 
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            height,width,depth = frame.shape
            print("Decode to cv2 image") 
        except CvBridgeError as e:
            print(e) 

        # Run once 
        if(True):
            # redraw the canvas 
            # self.fig2.canvas.draw()

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
            fgmask = self.fgbg.apply(cropped) 
            finalframe = cv2.bitwise_and(cropped, cropped, mask = fgmask)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 
            chans = cv2.split(finalframe)

            for y in range(0, 30, height):
                for x in range(0, 30, width):
                    testimg = finalframe[y:y+80,x:x+140,:]
                    testchans = cv2.split(testimg)
                    for (chan, color) in zip(chans, self.colors):
                        test_hist, test_bins = np.histogram(chan, bins = 100, range = [0,256])
                    intersection = self.histogram_intersection(self.gt_hist, test_hist)
                    #print("Intersection Value: "+str(intersection))
                    if (intersection >= 0.7):
                        self.isFish = True
                        #cv2.imshow("Fish Found", testimg)
                        print("Fish found!")
                        cv2.rectangle(finalframe,(y,x),(y+80,x+140),(0,255,0),3)
            # Don't use imshow - publish image below 
            cv2.imshow("Frame", finalframe) 
    
            # Draw the histogram 
            # plt.clf()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                return 

                cap.release()
                cv2.destroyAllWindows()

        try:
            # Put overlay image that shows where the fishes are here
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(finalframe, "bgr8"))
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