import numpy as np
import pywt
import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
from sklearn import metrics
from numpy.typing import NDArray
from typing import Optional

# Definicje klas i funkcji z transforms.py

class CompressionTransform:
    """
    Interface for compression transforms.
    """

    def forward(self, variables: NDArray) -> NDArray:
        pass

    def backward(self, variables: NDArray) -> NDArray:
        pass


class FourierTransform2D(CompressionTransform):
    """
    2D Fourier transform used for compression.
    Inverse transform uses absolute value by default.
    """

    def forward(self, variables: NDArray) -> NDArray:
        return np.fft.fft2(variables)

    def backward(self, variables: NDArray) -> NDArray:
        return np.abs(np.fft.ifft2(variables))


class WaveletTransform2D(CompressionTransform):
    """
    2D wavelet transform used for compression.
    """

    def __init__(self, wavelet_name: str, level: int):
        self.wavelet_name = wavelet_name
        self.level = level
        self.slices: Optional[NDArray] = None

    def forward(self, variables: NDArray) -> NDArray:
        transformed = pywt.wavedec2(variables, self.wavelet_name, level=self.level)
        coefficients, slices = pywt.coeffs_to_array(transformed)
        self.slices = slices

        return coefficients

    def backward(self, variables: NDArray) -> NDArray:
        if self.slices is None:
            raise ValueError("Cannot perform inverse transform without first performing forward transform!")

        variables = pywt.array_to_coeffs(variables, self.slices, output_format="wavedec2")
        return pywt.waverec2(variables, self.wavelet_name)


def compress_and_decompress(image: NDArray, transform: CompressionTransform, compression: float) -> NDArray:
    """
    Compresses and decompresses an image using the Fourier transform.
    This function can be used to see compression and decompression effects.

    :param image: greyscale image
    :param transform: transform to use, using CompressionTransform interface
    :param compression: ratio of coefficients to remove

    :return: image after compression and decompression
    """
    transformed = transform.forward(image)
    coefficients = np.sort(np.abs(transformed.reshape(-1)))  # sort by magnitude

    threshold = coefficients[int(compression * len(coefficients))]
    indices = np.abs(transformed) > threshold

    decompressed = transformed * indices
    return transform.backward(decompressed)

# Definicja funkcji z utils.py

def apply_rgb(func: callable, image: NDArray, *args, **kwargs) -> NDArray:
    """
    Applies a function to each color channel of an image.

    :param func: function to apply to each color channel
    :param image: image to apply function to

    :return: image after function has been applied to each color channel
    """
    return np.dstack([func(image[:, :, channel], *args, **kwargs) for channel in range(3)])

# Główny skrypt

# Wczytanie obrazu
image_path = "E:\\programowanie\\PycharmProjects\\SiOC\\kompresja\\obraz.jpg"
image = io.imread(image_path)

# Przygotowanie podsumowania
summary = []

# Kompresja, dekompresja i obliczanie błędu dla różnych stopni kompresji
for compression in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.999):
    compressed_image = apply_rgb(compress_and_decompress, image, transform=FourierTransform2D(), compression=compression)
    compressed_image = np.clip(compressed_image.astype(int), 0, 255)
    error = metrics.mean_absolute_error(y_true=image.flatten(), y_pred=compressed_image.flatten()) / image.max()
    summary.append({"compression": compression, "error": error})

# Konwersja podsumowania do DataFrame
summary_df = pd.DataFrame.from_dict(summary)

# Wyświetlenie pierwszych 10 wyników
print(summary_df.head(10))

# Wyświetlenie oryginalnego i skompresowanego obrazu dla przykładowego stopnia kompresji (np. 0.9)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Oryginalny obraz')
plt.axis('off')

plt.subplot(1, 2, 2)
sample_compressed_image = apply_rgb(compress_and_decompress, image, transform=FourierTransform2D(), compression=0.99)
sample_compressed_image = np.clip(sample_compressed_image.astype(int), 0, 255)
plt.imshow(sample_compressed_image)
plt.title('Skompresowany obraz (kompresja: 0.99)')
plt.axis('off')

plt.show()