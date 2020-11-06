import numpy             as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from hilbert import decode

num_dims = 3

def draw_curve(ax, num_bits):

  # The maximum Hilbert integer.
  max_h = 2**(num_bits*num_dims)

  # Generate a sequence of Hilbert integers.
  hilberts = np.arange(max_h)

  # Compute the 2-dimensional locations.
  locs = decode(hilberts, num_dims, num_bits)

  # Draw
  ax.plot(locs[:,0], locs[:,1], locs[:,2], '.-')
  # ax.set_aspect('equal')
  ax.set_title('%d bits per dimension' % (num_bits))
  ax.set_xlabel('dim 1')
  ax.set_ylabel('dim 2')
  ax.set_zlabel('dim 3')


fig = plt.figure(figsize=(16,4))
for ii, num_bits in enumerate([1, 2, 3, 4]):
  ax = fig.add_subplot(1,4,ii+1, projection='3d')
  draw_curve(ax, num_bits)
plt.savefig('example_3d.png', bbox_inches='tight')
