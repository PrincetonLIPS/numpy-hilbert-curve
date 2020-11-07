import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.animation as anim
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

from hilbert import decode

num_bits = 5
num_dims = 3

# The maximum Hilbert integer.
max_h = 2**(num_bits*num_dims)

# Generate a sequence of Hilbert integers.
hilberts = np.arange(max_h)

# Compute the 3-dimensional locations.
locs = decode(hilberts, num_dims, num_bits)

# Draw
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Choose pretty colors.
cmap = matplotlib.cm.get_cmap('copper')

# Draw. This may be a little slow.
for ii in range(max_h-1):
  print(ii, max_h)
  ax.plot([locs[ii,0], locs[ii+1,0]],
          [locs[ii,1], locs[ii+1,1]],
          [locs[ii,2], locs[ii+1,2]],
          '-', color=cmap(ii/max_h))

# ax.set_aspect('equal')
ax.set_title('%d bits per dimension' % (num_bits))
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')

def rotate(angle):
  ax.view_init(azim=angle)

rot_anim = anim.FuncAnimation(fig,
                              rotate,
                              frames=np.arange(0, 362, 2),
                              interval=100)

rot_anim.save('rotate_3d.gif') #, dpi=80, writer='imagemagick')
