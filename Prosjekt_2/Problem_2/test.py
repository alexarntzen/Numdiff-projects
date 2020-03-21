"""
=================
An animated image
=================

This example demonstrates how to animate an image.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2) = plt.subplots(1,2)

def f(x, y):
    return x/2 +y

x,y = np.ogrid[0:10,1:10]
im1 = ax1.imshow(f(x, y*10 ), origin="lower", interpolation="none")
im2 = ax2.imshow(f(x , y), animated=True, origin="lower", interpolation="none")


fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
ax2.set_xlabel("sdfsf")

#
# artists = []
# for i in range(50):
#     im1 = ax1.imshow(f(x, y*i*0.01), animated=True, origin="lower")
#     im2 = ax2.imshow(f(x*i*0.0, y), animated=True, origin="lower")
#     artists.append([im1,im2])
#
#
#
# ani = animation.ArtistAnimation(fig, artists, interval=50, blit=True)
#
