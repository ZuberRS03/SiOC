import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
from skimage import color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def add_noise(img, noise_level):
    # Dodanie szumu gaussowskiego do obrazu
    noisy_img = img + noise_level * np.random.normal(size=img.shape)
    return np.clip(noisy_img, 0, 1)

def dft2(image):
    # Dyskretna transformata Fouriera 2D
    return fftshift(fft2(image))

def idft2(fourier_image):
    # Odwrotna dyskretna transformata Fouriera 2D
    return ifft2(ifftshift(fourier_image))

def apply_filter(fourier_image, low_pass_radius):
    # Stworzenie maski filtru dolnoprzepustowego
    n, m = fourier_image.shape[0:2]
    y, x = np.ogrid[:n, :m]
    center = (n / 2, m / 2)
    mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= low_pass_radius ** 2
    filtered_image = np.zeros_like(fourier_image)
    for i in range(3):  # Przejście przez każdy kanał kolorów
        filtered_image[:, :, i] = fourier_image[:, :, i] * mask
    return filtered_image

# Wczytanie obrazu (przykład: obraz RGB)
image = mpimg.imread('E:/programowanie/PycharmProjects/SiOC/fourier/obraz.jpg')
if image.dtype == np.uint8:  # Sprawdzenie, czy wartości pikseli są w skali 0-255
    image = image / 255.0  # Normalizacja do skali 0-1

# Dodanie szumu do obrazu
noise_level = 0.2  # Poziom szumu
noisy_image = add_noise(image, noise_level)

# Transformacja Fouriera dla każdego kanału
fourier_image = np.zeros_like(noisy_image, dtype=complex)
for i in range(3):
    fourier_image[:, :, i] = dft2(noisy_image[:, :, i])

# Filtrowanie (usuwanie wysokich częstotliwości) dla każdego kanału
low_pass_radius = 45  # Promień filtra dolnoprzepustowego
filtered_fourier_image = apply_filter(fourier_image, low_pass_radius)

# Transformacja odwrotna Fouriera dla każdego kanału
denoised_image = np.zeros_like(noisy_image)
for i in range(3):
    denoised_image[:, :, i] = idft2(filtered_fourier_image[:, :, i]).real

# Wyświetlenie wyników
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image), plt.title('Oryginalny obraz')
plt.subplot(132), plt.imshow(noisy_image), plt.title('Zaszumiony obraz')
plt.subplot(133), plt.imshow(np.clip(denoised_image, 0, 1)), plt.title('odszumiony obraz')
plt.show()