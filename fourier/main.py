import sys

sys.path.append("../../src/")

import numpy as np
from matplotlib import pyplot as plt
from skimage import io, restoration


image = io.imread("E:/programowanie/PycharmProjects/SiOC/fourier/obraz.jpg")

plt.imshow(image)
plt.title("Obraz oryginalny")
plt.show()

noised_image = np.random.poisson(image / 64.0) * 64
image.max(), noised_image.max()

plt.imshow(noised_image)
plt.title("Obraz zaszumiony")
plt.show()


def anscombe(x):
    """Computes Anscombe transform"""
    c = 3 / 8
    return 2 * np.sqrt(x + c)


def inv_anscombe(y):
    """Computes inverse Anscombe transform"""
    c = -1 * 3 / 8
    return np.power(0.5 * y, 2) + c


def remove_high_frequencies(image, threshold):
    """Removes high frequencies from the image using Anscombe transform and TV denoising."""
    denoised_channels = []

    for channel in range(image.shape[-1]):
        x = image[:, :, channel] / image[:, :, channel].max()
        x = anscombe(x)
        x = restoration.denoise_tv_chambolle(x, weight=threshold)
        x = inv_anscombe(x)
        denoised_channels.append(x)

    denoised_image = np.stack(denoised_channels, axis=-1)

    return denoised_image

# Set the threshold for TV denoising
threshold_tv = 0.1

# Remove high frequencies to denoise the RGB image using Anscombe transform and TV denoising
denoised_image = remove_high_frequencies(image, threshold_tv)

plt.imshow(denoised_image)
plt.title("Obraz odszumiony")

plt.show()
