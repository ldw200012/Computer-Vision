import math

import numpy as np
from PIL import Image
from skimage import color, io
import cv2


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    out = io.imread(image_path)

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    
    
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    out = image1.copy()

    for row in range(0, image.shape[0]):
        for column in range(0, image.shape[1]):
            out[row][column] = 0.5*pow(out[row][column], 2)
    
    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    out = color.rgb2gray(image)

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    k = None
    out = image.copy()
    
    if channel == "R":
        k = 0
    elif channel == "G":
        k = 1
    elif channel == "B":
        k = 2
        
    for i in range(0,300):
        for j in range(0, 300):
            out[i][j][k] = 0

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    L, A, B = cv2.split(lab)

    if channel == "L":
        out = L
    elif channel == "A":
        out = A
    elif channel == "B":
        out = B

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None
    
    H, S, V = cv2.split(hsv)

    if channel == "H":
        out = H
    elif channel == "S":
        out = S
    elif channel == "V":
        out = V

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    
    image1 = rgb_exclusion(image1, channel1)
    image2 = rgb_exclusion(image2, channel2)
    
    out = image1.copy()
    
    if channel2 == 'R':
        k2 = 0
    elif channel2 == 'G':
        k2 = 1
    elif channel2 == 'B':
        k2 = 2
        
    for i in range(0, image2.shape[0]):
        for j in range(int(image2.shape[1]/2), image2.shape[1]):
            if(k2 == 0):
                out[i][j][0] = 0
                out[i][j][1] = image2[i][j][1]
                out[i][j][2] = image2[i][j][2]
            elif(k2 == 1):
                out[i][j][0] = image2[i][j][0]
                out[i][j][1] = 0
                out[i][j][2] = image2[i][j][2]
            elif(k2 == 2):
                out[i][j][0] = image2[i][j][0]
                out[i][j][1] = image2[i][j][1]
                out[i][j][2] = 0

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
