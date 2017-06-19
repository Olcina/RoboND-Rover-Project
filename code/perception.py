import numpy as np
import cv2

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

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


def ground_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
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
    res = cv2.bitwise_and(img,img, mask= mask)
    #apply color_select to our new res image
    color_select = color_thresh(res,(0,82,0))

    return color_select


def navigable_terrain(img):
    #a new container for the image wiht RGB channels
    nav_terr = np.zeros_like(img)
    binary_nav = color_thresh(img)
    #all red
    nav_terr[:,:,0] = 255
    ypos, xpos = binary_nav.nonzero()
    nav_terr[ypos,xpos, 0] = 0
    #ypos,xpos = binary_nav.where(binary_nav>0)
    nav_terr[ypos,xpos , 2] = 255
    return nav_terr


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped

def check_if_capture_valid_for_mapping(roll_angle,pith_angle,angle_max_deviation):
    #we should only map when the angles of roll and pith are below some value
    if roll_angle > 360 - angle_max_deviation or roll_angle < angle_max_deviation:
        if pith_angle > 360 - angle_max_deviation or pith_angle < angle_max_deviation:
            return True
        else:
            return False
    else:
        return False

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    image = Rover.img
    dst_size = 5
    scale = 10
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable_world = color_thresh(warped)
    bool_arr = np.array(navigable_world, dtype=np.bool)
    obstacle_world = np.logical_not(bool_arr)
    #Rock detection ---
    rocks_world = rock_detector(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    #nav_terr = navigable_terrain(warped)
    navigable_coords = navigable_world.nonzero()
    obstacle_coords  = obstacle_world.nonzero()
    rocks_coords = rocks_world.nonzero()
    Rover.vision_image[:,:,:] = 0
    Rover.vision_image[navigable_coords[0],navigable_coords[1],0] = 255
    Rover.vision_image[rocks_coords[0],rocks_coords[1],1] = 255
    Rover.vision_image[obstacle_coords[0],obstacle_coords[1],2] = 255


    # 5) Convert map image pixel values to rover-centric coords
    navigable_rover_coords = rover_coords(navigable_world)
    obstacle_rover_coords = rover_coords(obstacle_world)
    rocks_rover_coords = rover_coords(rocks_world)
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    navigable_world_coords = pix_to_world(navigable_rover_coords[0],navigable_rover_coords[1],
                                            Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0], scale)
    obstacle_world_coords = pix_to_world(obstacle_rover_coords[0],obstacle_rover_coords[1],
                                            Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0], scale)
    rocks_world_coords = pix_to_world(rocks_rover_coords[0],rocks_rover_coords[1],
                                            Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0], scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    #Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    """
    if abs(Rover.pitch) < 0.15 and abs(Rover.roll) <0.15:
        Rover.worldmap[navigable_world_coords[1], navigable_world_coords[0], 2] += 1
        Rover.worldmap[rocks_world_coords[1], rocks_world_coords[0], 1] += 1
        Rover.worldmap[obstacle_world_coords[1], obstacle_world_coords[0], 0] += 1
    """
    if check_if_capture_valid_for_mapping(Rover.roll,Rover.pitch,1):
        Rover.worldmap[navigable_world_coords[1], navigable_world_coords[0], 2] += 1
        Rover.worldmap[rocks_world_coords[1], rocks_world_coords[0], 1] += 1
        Rover.worldmap[obstacle_world_coords[1], obstacle_world_coords[0], 0] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    rover_centric_pixel_distances, rover_centric_angles= to_polar_coords(navigable_rover_coords[0],navigable_rover_coords[1])
    rock_centric_pixel_distance, rock_centric_pixel_angles = to_polar_coords(rocks_rover_coords[0], rocks_rover_coords[1])
    # Update Rover pixel distances and angles
    Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_angles = rover_centric_angles
    #if there is a rock calculate the distante and angle toward the rock
    if np.mean(rocks_world) > 0.0:
        Rover.sample_in_view = True
        #Calculate dist and angle toward the rock
        Rover.dist_to_rock = rock_centric_pixel_distance
        Rover.angle_to_rock = rock_centric_pixel_angles
    else:
        Rover.sample_in_view = False
        Rover.dist_to_rock  = 0
        Rover.angle_to_rock = 0

    return Rover
