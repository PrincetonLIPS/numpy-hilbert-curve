import numpy as np

def right_shift(binary, k=1, axis=-1):
  ''' Right shift an array of binary values.

  Parameters:
  -----------
   binary: An ndarray of binary values.

   k: The number of bits to shift. Default 1.

   axis: The axis along which to shift.  Default -1.

  Returns:
  --------
   Returns an ndarray with zero prepended and the ends truncated, along
   whatever axis was specified.
'''

  # If we're shifting the whole thing, just return zeros.
  if binary.shape[axis] <= k:
    return np.zeros_like(binary)

  # Determine the padding pattern.
  padding = [(0,0)] * len(binary.shape)
  padding[axis] = (k,0)

  # Determine the slicing pattern to eliminate just the last one.
  slicing = [slice(None)] * len(binary.shape)
  slicing[axis] = slice(None, -k)

  shifted = np.pad(binary[tuple(slicing)], padding,
                   mode='constant', constant_values=0)

  return shifted


def binary2gray(binary, axis=-1):
  ''' Convert an array of binary values into Gray codes.

  This uses the classic X ^ (X >> 1) trick to compute the Gray code.

  Parameters:
  -----------
   binary: An ndarray of binary values.

   axis: The axis along which to compute the gray code. Default=-1.

  Returns:
  --------
   Returns an ndarray of Gray codes.
  '''
  shifted = right_shift(binary, axis=axis)

  # Do the X ^ (X >> 1) trick.
  gray = np.logical_xor(binary, shifted)

  return gray

def gray2binary(gray, axis=-1):
  ''' Convert an array of Gray codes back into binary values.

  Parameters:
  -----------
   gray: An ndarray of gray codes.

   axis: The axis along which to perform Gray decoding. Default=-1.

  Returns:
  --------
   Returns an ndarray of binary values.
  '''

  # Loop the log2(bits) number of times necessary, with shift and xor.
  shift = 2**(int(np.ceil(np.log2(gray.shape[axis])))-1)
  while shift > 0:
    gray = np.logical_xor( gray, right_shift(gray, shift) )
    shift //= 2

  return gray
