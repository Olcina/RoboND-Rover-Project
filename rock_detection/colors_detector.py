import cv2
import numpy as np
import matplotlib.image as mpimg
def nothing(x):
    pass

# img needed to pupulate the named window 'image' and the TrackBars crash otherwise
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow("image")
# create trackbars for upper treshold
cv2.createTrackbar('R_upp','image',0,255,nothing)
cv2.createTrackbar('G_upp','image',0,255,nothing)
cv2.createTrackbar('B_upp','image',0,255,nothing)
# create trackbars for lowertresholds
cv2.createTrackbar('R_low','image',0,255,nothing)
cv2.createTrackbar('G_low','image',0,255,nothing)
cv2.createTrackbar('B_low','image',0,255,nothing)

#define an example rock image
example_rock = 'example_rock2.jpg'
rock_img = mpimg.imread(example_rock)


cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    R_upp = cv2.getTrackbarPos('R_upp','image')
    G_upp = cv2.getTrackbarPos('G_upp','image')
    B_upp = cv2.getTrackbarPos('B_upp','image')
    R_low = cv2.getTrackbarPos('R_low','image')
    G_low = cv2.getTrackbarPos('G_low','image')
    B_low = cv2.getTrackbarPos('B_low','image')
    # Convert BGR to HSV
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    lower_yellow = np.array([B_low,G_low,R_low])
    upper_yellow = np.array([B_upp,G_upp,R_upp])
    #lower_yellow = np.array([0,82,0])
    #upper_yellow = np.array([255,255,50])
    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(frame, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow("image",img)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
