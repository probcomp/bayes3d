import numpy as np
import matplotlib.pyplot as plt

data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_images = data["depth_images"]


# f, axarr = plt.subplots(1,2)
# axarr[0].imshow(depth_images[-1])
# axarr[1].imshow(rgb_images[-1])

# plt.show()


plt.clf()
plt.imshow(rgb_images[-1])
plt.imshow(depth_images[-1],alpha=0.9)

plt.show()
