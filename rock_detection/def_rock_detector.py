import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def rock_detector(img):
    #Isolate yellow pixel's from the colored rock_images

    #Value selection for the upper and lower yellow can be found in rock_detector.py
    lower_yellow = np.array([0,82,0])
    upper_yellow = np.array([255,255,50])
    # Threshold the BRG image to get only blue colors
    mask = cv2.inRange(img, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,rock_img, mask= mask)
    #apply color_select to our new res image
    color_select = color_thresh(res,(0,82,0))

    return color_select

#Try with an example_rock and matplotlib

#define an example rock image
example_rock = 'example_rock2.jpg'
rock_img = mpimg.imread(example_rock)

color_select = rock_detector(rock_img)


plt.subplot(121)
plt.imshow(rock_img)
plt.subplot(122)
plt.imshow(color_select)

plt.savefig('rock_detection.png')
plt.show()
