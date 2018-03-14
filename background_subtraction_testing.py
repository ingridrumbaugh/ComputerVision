import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 
'''
Some code taken from: 
https://stackoverflow.com/questions/43099734/combining-cv2-imshow-with-matplotlib-plt-show-in-real-time
'''

fig = plt.figure()
plt.ion()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
# channels for the histogram 
colors = ('b', 'g', 'r') 

matrix = np.zeros((75, 75), np.float32) 
green = (0, 255, 0) 

#                    start x/y    end x/y

isFish = True

while(True):
   
    # redraw the canvas 
    fig.canvas.draw()
    # convert canvas to image 
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
    img = img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
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
    
    cv2.imshow("Thresh", thresh) 
    # break up channels for the histogram 
    chans = cv2.split(finalframe) 
    
    for y in range(0, height):
        for x in range(0, width):
            pt1 = (int(x), int(y))
            pt2 = (int(x+50), int(y+50))
            cv2.rectangle(finalframe, pt1, pt2, green)
        
    # show the image 
    cv2.imshow("Frame", finalframe) 
    
    # Draw the histogram 
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan],[0], None, [256], [2,245])
        plt.plot(hist, color = color, linewidth = 2.0) 
        plt.xlim([0,256])
       
 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()



