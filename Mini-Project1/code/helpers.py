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
  raise NotImplementedError('my_imfilter function in helpers.py needs to be implemented')
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

  # Making the low pass filter image by
   
  kernel = create_gaussian_filter((cutoff_frequency*2)+1,cutoff_frequency)
  
 
  low_frequencies = my_imfilter(image1,kernel) #

  # Making the high pass filter image by creating a low pass filter version the substracting the original image from the low pass filter version

  image2_blurred = my_imfilter(image2,kernel) 

  high_frequencies = image2 - image2_blurred  

  # Making the hybrid images by combining the low pass filter and high pass filter images
  
  hybrid_image = high_frequencies + low_frequencies 


  # Clipping values above 1 to 1 and values under zero to zero
  low_frequencies = np.clip(low_frequencies, a_min = 0, a_max = 1)
  high_frequencies = np.clip(high_frequencies, a_min = 0, a_max = 1)
  hybrid_image = np.clip(hybrid_image, a_min = 0, a_max = 1)

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
