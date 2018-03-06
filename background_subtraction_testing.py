from matplotlib import pyplot as plt 
import numpy as np
import argparse
import cv2 

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args()) 
# image = cv2.imread(args["image"]) 

# Interactive mode on 
plt.ion()

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
colors = ('b', 'g', 'r') 
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels") 

while(True):

    ret, frame = cap.read() 
    height, width, channels = frame.shape
    # starty:endy   startx:endx
    cropped = frame[height/4:(3*height)/4, 0, width] 
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) 
    # blur the image first 
    gray = cv2.GaussianBlur(gray, (5, 5), 0) 
    
    # black out everything except the fish 
    fgmask = fgbg.apply(cropped) 
    finalframe = cv2.bitwise_and(cropped, cropped, mask = fgmask)
    chans = cv2.split(finalframe) 
    
    # Loop over each of  the channels in the image 
    # For each channel compute a histogram 
    '''
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color = color) 
        plt.xlim([0, 256]) 
    '''
    cv2.imshow("Mask", finalframe) 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break 
    # plt.clf() 
    
cap.release()
cv2.destroyAllWindows()

# gray = cv2.bitwise_and(gray,fgmask)



