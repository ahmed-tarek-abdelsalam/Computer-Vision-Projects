# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
import copy
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from scipy.signal import convolve2d, correlate2d

def create_gaussian_filter(side_length, sigma):
    '''
    This function is used to create a guassian filter
    Input: 
    Sigma, the required size
    Output :
    Kernel 
    '''

    ax = np.linspace(-(side_length - 1) / 2., (side_length- 1) / 2., side_length)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)

def _isKernelOdd(filter: np.ndarray):
    '''
    This function checks if either of the kernel dimenision is odd
    Input: 
    - 2D nparray filter
    Output: 
    - Has no output, But raise error if either dimension of the filter
      is even
    '''
    if filter.shape[0]%2==0:
        raise NameError('Number of rows of the kernel should be odd') 
    elif filter.shape[1]%2 == 0:
        raise NameError('Number of columns of the kernel should be odd')

def _2Dconvolution(image:np.ndarray,kernel:np.ndarray) ->np.ndarray:
    '''
    This function makes 2D convolution, by flipping the kernel and correlating to the image
    Input:
    - Image 2D np.array
    - Kernel 2D np.array with odd dimensions
    Output: 
    - returns convolved version of the image maintaining the orignal size of the image
    '''
    
    # First let's pad the image in order to avoid Cirular convolution
    # and subconsequently avoid losing info
    
    imageLen= image.shape[0]
    imageWidth= image.shape[1]
    kernelLen,kernelWidth= kernel.shape
    
    
    paddedImage= np.zeros((imageLen+kernelLen-1,imageWidth+kernelWidth-1))
    paddedImage[0:imageLen,0:imageWidth]=copy.deepcopy(image)
    # deep copy is used to avoid any manipulation that could happen in the orignal image
    
    # the image that we are going to return after updating its values
    filteredImage= np.zeros((imageLen+kernelLen-1,imageWidth+kernelWidth-1))
    
    for i in  range(paddedImage.shape[0]):
        for j in range(paddedImage.shape[1]):
            for k in range(kernel.shape[0]):
                for l in  range(kernel.shape[1]):
                    # checking that we have a valid array indexing
                    if i-k>=0 and  j-k>=0:                        
                        # making sure that the calculated value doesn'et exceeds the pixels range
                        value=np.clip(filteredImage[i,j]+ kernel[k,l]*paddedImage[i-k,j-k],0,255)
                        filteredImage[i,j]= value
    # now we want to take restore the image's orignial shape
    # Our kernel's anchor is the left-up cornered square Hence our 
    # convolved image is at the enter. As a result we want to take this part without all the borderds
    # and since our kernel is odd then we an use the bellow equation
    return filteredImage[kernelLen//2:imageLen+kernelLen//2,kernelWidth//2:imageWidth+kernelWidth//2]

def _2Dcorrelation(paddedImage,kernel) -> np.ndarray:
    '''
    This Funtion correlates 2D matrices with each other by passing the inverted kernel to 2D convolution function. 
    Input:
    - Image 2D np.array
    - Kernel 2D np.array with odd dimensions
    Output: 
    - returns Image after correlating it with the kernel,It also maintains the orignal size of the image.
    '''
    return _2Dconvolution(paddedImage,np.flip(kernel))

def my_imfilter(image: np.ndarray, filter: np.ndarray):
  """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  filtered_image = np.asarray([0])

  ##################
  # Your code here #
  
  # First let's check that the kernel is not odd dim 
  # if any dim is even an error message would occur
  _isKernelOdd(filter)
    
  # second let's check if we have a gray or multi-channel Image
  if image.ndim==2: 
    # means it is a gray image
    filtered_image= _2Dcorrelation(image,filter)
  else: 
    # means it is a multi channel image 
    filtered_image= np.zeros(image.shape)
    for c in range(image.shape[2]):
      filtered_image[:,:,c]= _2Dcorrelation(image[:,:,c],filter)                    


  return filtered_image
  ##################

  return filtered_image

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
  kernel = None
  
  # Your code here:
  low_frequencies = None # Replace with your implementation

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
  high_frequencies = None # Replace with your implementation

  # (3) Combine the high frequencies and low frequencies
  # Your code here #
  hybrid_image = None # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!

  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect')
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
