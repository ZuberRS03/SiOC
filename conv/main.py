import numpy as np
from skimage import io
from scipy.ndimage import convolve

import matplotlib.pyplot as plt

# image = io.imread(r"E:\programowanie\PycharmProjects\SiOC\conv\circle.jpg")
# image.shape
#
# plt.imshow(image)
# plt.show()
#
# laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
# laplace_filter
#
# filtered_image = np.dstack([
#     convolve(image[:, :, channel], laplace_filter, mode="constant", cval=0.0)
#     for channel in range(3)
# ])
#
# filtered_image.shape
#
# filtered_image.min()
#
# plt.imshow(filtered_image)
# plt.show()

image = io.imread(r"E:\programowanie\PycharmProjects\SiOC\conv\obraz.jpg")
image.shape

plt.imshow(image)
plt.show()

mean_filter = np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]]) / 16
mean_filter

mean_filter = np.ones([9, 9]) / (9 ** 2)

filtered_image = np.dstack([
    convolve(image[:, :, channel], mean_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

filtered_image.shape

plt.imshow(filtered_image)
plt.show()

mean_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
mean_filter = mean_filter / mean_filter.sum()
mean_filter

filtered_image = np.dstack([
    convolve(image[:, :, channel], mean_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

filtered_image.shape

plt.imshow(filtered_image)
plt.show()


