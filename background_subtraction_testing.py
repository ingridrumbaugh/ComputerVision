import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

'''
Some code taken from: 
https://stackoverflow.com/questions/43099734/combining-cv2-imshow-with-matplotlib-plt-show-in-real-time
As well as:
https://github.com/mpatacchiola/deepgaze/blob/master/examples/ex_color_classification_images/ex_histogram_intersection.py
'''

# Import frame with archerfish 
archfish = cv2.imread('tinyfish.png') 

# x and y coords where archerfish is in image 
tempx = 690
tempy = 370 

# crop just the archerfish 
croparchfish = archfish

# split ground truth image into channels 
ogchans = cv2.split(croparchfish)
colors = ('b', 'g', 'r') 

# Don't print out the histogram - we just need the data from it

plt.figure()
plt.title("Ground Truth Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")


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
    # Show the original image 
    cv2.imshow("Original ArcherFish", archfish)

    # redraw the canvas 
    fig2.canvas.draw()
    # convert canvas to image 
    img = np.fromstring(fig2.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
    img = img.reshape(fig2.canvas.get_width_height()[::-1]+(3,))
    # img is rgb, conver to opencv bgr 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Plot",img) 

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
    
    cv2.imshow("thresh", thresh)
    chans = cv2.split(finalframe)

    # sweep rectangle over image 
    # don't print the rectangle! 

    for y in range(0, 30, height):
        for x in range(0, 30, width):
            testimg = finalframe[y:y+80,x:x+140,:]
            testchans = cv2.split(testimg)
            for (chan, color) in zip(chans, colors):
                test_hist, test_bins = np.histogram(chan, bins = 100, range = [0,256])
            intersection = histogram_intersection(gt_hist, test_hist)
            #print("Intersection Value: "+str(intersection))
            if (intersection >= 0.7):
                isFish = True
                cv2.imshow("Fish Found", testimg)
                print("Fish found!")

    # show the image 
    cv2.imshow("Frame", finalframe) 
    
    # Draw the histogram 
    plt.clf()
    
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan],[0], None, [256], [2,245])
        plt.plot(hist, color = color, linewidth = 2.0) 
        plt.xlim([0,256])
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break 

            cap.release()
            cv2.destroyAllWindows()



