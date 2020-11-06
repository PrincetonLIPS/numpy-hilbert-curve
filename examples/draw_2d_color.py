import numpy             as np
import matplotlib.pyplot as plt
import matplotlib

from hilbert import decode

num_dims = 2

def draw_curve(ax, num_bits):

  # The maximum Hilbert integer.
  max_h = 2**(num_bits*num_dims)

  # Generate a sequence of Hilbert integers.
  hilberts = np.arange(max_h)

  # Compute the 2-dimensional locations.
  locs = decode(hilberts, num_dims, num_bits)

  # Choose pretty colors.
  cmap = matplotlib.cm.get_cmap('copper')

  # Draw. This may be a little slow.
  for ii in range(max_h-1):
    ax.plot([locs[ii,0], locs[ii+1,0]],
            [locs[ii,1], locs[ii+1,1]],
            '-', color=cmap(ii/max_h))
  ax.set_aspect('equal')
  ax.set_title('%d bits per dimension' % (num_bits))
  ax.set_xlabel('dim 1')
  ax.set_ylabel('dim 2')


fig = plt.figure(figsize=(16,4))
for ii, num_bits in enumerate([4, 5, 6, 7]):
  ax = fig.add_subplot(1,4,ii+1)
  draw_curve(ax, num_bits)
plt.savefig('example_2d_color.png', bbox_inches='tight')
