import numpy as np

from .gray import binary2gray

def decode(hilberts, num_dims, num_bits):
  ''' Decode an array of Hilbert integers into locations in a hypercube.

  This is a vectorized-ish version of the Hilbert curve implementation by John
  Skilling as described in:

  Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
    Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

  Params:
  -------
   hilberts - An ndarray of Hilbert integers.  Must be an integer dtype and
              cannot have fewer bits than num_dims * num_bits.

   num_dims - The dimensionality of the hypercube. Integer.

   num_bits - The number of bits for each dimension. Integer.

  Returns:
  --------
   The output is an ndarray of unsigned integers with the same shape as hilberts
   but with an additional dimension of size num_dims.
  '''

  if num_dims*num_bits > 64:
    raise ValueError(
      '''
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      ''' % (num_dims, num_bits)
    )

  # Handle the case where we got handed a naked integer.
  hilberts = np.atleast_1d(hilberts)

  # Keep around the shape for later.
  orig_shape = hilberts.shape

  # Treat each of the hilberts as a sequence of eight uint8.
  # This treats all of the inputs as uint64 and makes things uniform.
  hh_uint8 = np.reshape(hilberts.ravel().astype('>u8').view(np.uint8), (-1, 8))

  # Turn these lists of uints into lists of bits and then truncate to the size
  # we actually need for using Skilling's procedure.
  hh_bits = np.unpackbits(hh_uint8, axis=1)[:,-num_dims*num_bits:]

  # Take the sequence of bits and Gray-code it.
  gray = binary2gray(hh_bits)

  # There has got to be a better way to do this.
  # I could index them differently, but the eventual packbits likes it this way.
  gray = np.swapaxes(
    np.reshape(gray, (-1, num_bits, num_dims)),
    axis1=1, axis2=2,
  )

  # Iterate backwards through the bits.
  for bit in range(num_bits-1, -1, -1):

    # Iterate backwards through the dimensions.
    for dim in range(num_dims-1, -1, -1):

      # Identify which ones have this bit active.
      mask = gray[:,dim,bit]

      # Where this bit is on, invert the 0 dimension for lower bits.
      gray[:,0,bit+1:] = np.logical_xor(gray[:,0,bit+1:], mask[:,np.newaxis])

      # Where the bit is off, exchange the lower bits with the 0 dimension.
      to_flip = np.logical_and(
        np.logical_not(mask[:,np.newaxis]),
        np.logical_xor(gray[:,0,bit+1:], gray[:,dim,bit+1:])
      )
      gray[:,dim,bit+1:] = np.logical_xor(gray[:,dim,bit+1:], to_flip)
      gray[:,0,bit+1:] = np.logical_xor(gray[:,0,bit+1:], to_flip)

  # Pad back out to 64 bits.
  extra_dims = 64 - num_bits
  padded = np.pad(gray, ((0,0), (0,0), (extra_dims,0)),
                  mode='constant', constant_values=0)

  # Now chop these up into blocks of 8.
  locs_chopped = np.reshape(padded[:,:,::-1], (-1, num_dims, 8, 8))

  # Take those blocks and turn them unto uint8s.
  locs_uint8 = np.squeeze(np.packbits(locs_chopped, bitorder='little', axis=3))

  # Finally, treat these as uint64s.
  flat_locs = locs_uint8.view(np.uint64)

  # Return them in the expected shape.
  return np.reshape(flat_locs, (*orig_shape, num_dims))

def show_square(num_bits):
  num_dims = 2
  max_h = 2**(num_dims*num_bits)
  hh = np.arange(max_h)
  locs = decode(hh, num_dims, num_bits)

  import matplotlib.pyplot as plt
  plt.figure(figsize=(12,12))
  plt.plot(locs[:,0], locs[:,1], '.-')
  plt.show()

def show_cube(num_bits):
  num_dims = 3
  max_h = 2**(num_dims*num_bits)
  hh = np.arange(max_h)
  locs = decode(hh, num_dims, num_bits)

  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure(figsize=(12,12))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(locs[:,0], locs[:,1], locs[:,2], '.-')
  plt.show()
