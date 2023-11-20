import sys

sys.path.append("../src")

import fractions
import numpy as np
from numpy.typing import NDArray
from skimage import io, transform, color

image = io.imread(r"C:\Users\matiz\PycharmProjects\interpolacja_obrazka\obrazek1.jpg")
image = transform.resize(image, output_shape=(200, 300, 3))
image.shape

def ratio_to_fraction(ratio: float) -> tuple[int, int]:
    """Returns denominator and numerator from any float"""
    frac = fractions.Fraction(ratio).limit_denominator()
    return frac.numerator, frac.denominator

def linear_kernel(t):
    """Linear interpolation kernel"""
    return max(0, 1 - abs(t))

def image_interpolate(image: NDArray, kernel: callable, ratio: int) -> NDArray:
    """Image interpolation for downsampling"""
    h, w = image.shape
    new_h, new_w = h // ratio, w // ratio
    interpolated_image = np.zeros((new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            for m in range(ratio):
                for n in range(ratio):
                    x, y = i * ratio + m, j * ratio + n
                    interpolated_image[i, j] += image[x, y] * kernel(m/ratio) * kernel(n/ratio)

    return interpolated_image

def downsample(image: NDArray, kernel_size: int) -> NDArray:
    """Downsample the image"""
    h, w = image.shape
    new_h, new_w = h // kernel_size, w // kernel_size
    downsampled_image = np.zeros((new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            downsampled_image[i, j] = image[i * kernel_size, j * kernel_size]

    return downsampled_image

def rgb_image_interpolate(image: NDArray, kernel: callable, ratio: int) -> NDArray:
    """Image interpolation for downsampling (RGB image)"""
    h, w, c = image.shape
    new_h, new_w = h // ratio, w // ratio
    interpolated_image = np.zeros((new_h, new_w, c))

    for k in range(c):
        for i in range(new_h):
            for j in range(new_w):
                for m in range(ratio):
                    for n in range(ratio):
                        x, y = i * ratio + m, j * ratio + n
                        interpolated_image[i, j, k] += image[x, y, k] * kernel(m/ratio) * kernel(n/ratio)

    return interpolated_image

def rgb_downsample(image: NDArray, kernel_size: int) -> NDArray:
    """Downsample the image (RGB image)"""
    h, w, c = image.shape
    new_h, new_w = h // kernel_size, w // kernel_size
    downsampled_image = np.zeros((new_h, new_w, c))

    for k in range(c):
        for i in range(new_h):
            for j in range(new_w):
                downsampled_image[i, j, k] = image[i * kernel_size, j * kernel_size, k]

    return downsampled_image

def scale(image: NDArray, ratio: float, kernel: callable) -> NDArray:
    """Scales given image with non-integer ratio and given interpolation kernel"""
    upscale, downscale = ratio_to_fraction(ratio)  # upscale and downscale sizes
    color = True if image.ndim == 3 else False

    if color:
        image = rgb_image_interpolate(image, kernel=kernel, ratio=upscale)
        image = rgb_downsample(image, kernel_size=downscale)
    else:
        image = image_interpolate(image, kernel=kernel, ratio=upscale)
        image = downsample(image, kernel_size=downscale)

    return image

# Zmniejsz obrazek
downscaled_image = scale(image, ratio=0.5, kernel=linear_kernel)

# Powiększ obrazek dwukrotnie
upscaled_image = scale(downscaled_image, ratio=2, kernel=linear_kernel)

# Wyświetl oryginalny, zmniejszony i powiększony obrazek
io.imshow(image)
io.show()
io.imshow(downscaled_image)
io.show()
io.imshow(upscaled_image)
io.show()