import numpy as np

from .gray import gray2binary

def encode(locs, num_dims, num_bits):
  ''' Decode an array of locations in a hypercube into a Hilbert integer.

  This is a vectorized-ish version of the Hilbert curve implementation by John
  Skilling as described in:

  Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
    Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

  Params:
  -------
   locs - An ndarray of locations in a hypercube of num_dims dimensions, in
          which each dimension runs from 0 to 2**num_bits-1.  The shape can
          be arbitrary, as long as the last dimension of the same has size
          num_dims.

   num_dims - The dimensionality of the hypercube. Integer.

   num_bits - The number of bits for each dimension. Integer.

  Returns:
  --------
   The output is an ndarray of uint64 integers with the same shape as the
   input, excluding the last dimension, which needs to be num_dims.
  '''

  # Keep around the original shape for later.
  orig_shape = locs.shape

  if orig_shape[-1] != num_dims:
    raise ValueError(
      '''
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      ''' % (orig_shape[-1], num_dims)
    )

  if num_dims*num_bits > 64:
    raise ValueError(
      '''
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      ''' % (num_dims, num_bits)
    )

  # TODO: check that the locations are valid.

  # Treat the location integers as 64-bit unsigned and then split them up into
  # a sequence of uint8s.  Preserve the association by dimension.
  locs_uint8 = np.reshape(locs.astype('>u8').view(np.uint8), (-1, num_dims, 8))

  # Now turn these into bits and truncate to num_bits.
  gray = np.unpackbits(locs_uint8, axis=-1)[...,-num_bits:]

  # Run the decoding process the other way.
  # Iterate forwards through the bits.
  for bit in range(0, num_bits):

    # Iterate forwards through the dimensions.
    for dim in range(0, num_dims):

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

  # Now flatten out.
  gray = np.reshape(
    np.swapaxes(gray, axis1=1, axis2=2),
    (-1, num_bits*num_dims)
  )

  # Convert Gray back to binary.
  hh_bin = gray2binary(gray)

  # Pad back out to 64 bits.
  extra_dims = 64 - num_bits*num_dims
  padded = np.pad(hh_bin, ((0, 0), (extra_dims, 0)),
                  mode='constant', constant_values=0)

  # Convert binary values into uint8s.
  hh_uint8 = np.squeeze(np.packbits(np.reshape(padded[:,::-1], (-1, 8, 8)),
                                    bitorder='little', axis=2))

  # Convert uint8s into uint64s.
  hh_uint64 = np.squeeze(hh_uint8.view(np.uint64))

  return hh_uint64
