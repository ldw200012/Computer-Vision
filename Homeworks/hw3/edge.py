import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    n_kernel = np.flip(kernel)
    
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = (n_kernel*padded[i:i+Hk,j:j+Wk]).sum()
     #out = np.sum(np.multiply(padded,n_kernel))
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    
    k = (size-1)//2
    
    for i in range(size):
        for j in range(size):
            kernel[i,j] = 1.0/(2*np.pi*np.power(sigma, 2))*np.exp(-float(np.power(i-k, 2) + np.power(j-k, 2))/(2*np.power(sigma, 2)))
    
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    
    x = np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]])
    out = conv(img,x)
    
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    
    y = np.array([[0,0.5,0],[0,0,0],[0,-0.5,0]])
    out = conv(img,y)
    
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    
    G_x = partial_x(img)
    G_y = partial_y(img)
    
    G = np.sqrt(np.power(G_x, 2)+np.power(G_y, 2))
    theta = np.arctan2(G_y, G_x)
    
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if theta[i, j] < 0:
                theta[i, j] += 360
            elif theta[i, j] >360:
                theta[i, j] -= 360
    
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    pad_width = ((1,1),(1,1))
    img = np.pad(G,pad_width)
    
    for i in range(H):
        for j in range(W):
            if theta[i,j] == 0 or theta[i,j] == 180:
                if img[i+1,j+1] >= img[i+1,j] and img[i+1,j+1] >= img[i+1,j+2]:
                out[i,j] = img[i+1,j+1] 
            elif theta[i,j] == 45 or theta[i,j] == 225:
                if img[i+1,j+1] >= img[i+2,j+2] and img[i+1,j+1] >= img[i,j]:
                    out[i,j] = img[i+1,j+1]
            elif theta[i,j] == 90 or theta[i,j] == 270:
                if img[i+1,j+1] >= img[i+2,j+1] and img[i+1,j+1] >= img[i,j+1]:
                    out[i,j] = img[i+1,j+1]
            else:
                if img[i+1,j+1] >= img[i+2,j] and img[i+1,j+1] >= img[i,j+2]:
                    out[i,j] = img[i+1,j+1
    
    ### END YOUR CODE
    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    
    H, W = img.shape
    
    for i in range(H):
        for j in range(W):
            if img[i,j] > high:
                strong_edges[i,j] = 1
            elif img[i,j] >= low:
                weak_edges[i,j] = 1        
    
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    
    for i in range(H):
        for j in range(W):
            if strong_edges[i,j] == 1:
                neighbors = get_neighbors(i,j, H, W)
                for a,b in neighbors:
                    if weak_edges[a,b] == 1:
                        edges[a,b] = 1
            else:
                continue
                    
    
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    
    kernel = gaussian_kernel(kernel_size,sigma)
    n_img = conv(img,kernel)
    G,theta = gradient(n_img)
    out = non_maximum_suppression(G,theta)
    strong_edges, weak_edges = double_thresholding(out,high,low)
    edge = link_edges(strong_edges,weak_edges)
                          
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
                                   
    size = len(ys)
    for i in range(size):
        a = xs[i]
        b = ys[i]
        for j in range(num_thetas):
            rho = (a*cos_t[j]+b*sin_t[j]) + diag_len
            
            
    
    ### END YOUR CODE

    return accumulator, rhos, thetas
