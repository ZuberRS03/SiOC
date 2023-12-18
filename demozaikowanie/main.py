import numpy as np
from skimage import io
from matplotlib import pyplot as plt

image = io.imread(r"E:\programowanie\PycharmProjects\SiOC\demozaikowanie\CFA_BIN\CFA\Bayer\namib.jpg")
image.shape

image[:4, :4, 1]

image.dtype

plt.imshow(image)
plt.show()

green = image[:, :, 1]
red = image[:, :, 0]
blue = image[:, :, 2]

plt.imshow(image)
plt.show()

np.arange(10)[1::2]

to_interp = green
to_interp.shape

np.sum(green)

np.sum(to_interp)

to_interp = green[::2,:]
to_interp = to_interp[:, ::2]

to_interp.shape

plt.imshow(to_interp)
plt.show()

to_interp.shape

def linear_kernel(x, offset: float, width: float):
    """Linear interpolation kernel"""
    return (1 - np.abs((x - offset) / width)) * (np.abs((x - offset) / width) < 1)

x = np.linspace(-3, 3, 1000)
y = linear_kernel(x, offset=0.0, width=1)

plt.plot(x, y)
plt.show()

to_interp[0].shape


def interpolate_row(row):
    kernels = []
    space = np.linspace(0, 1, 2 * len(row))

    for x, y in zip(space.tolist(), row.tolist()):
        kernel = linear_kernel(space, offset=2 * x, width=1 / len(row))
        # print(kernel.shape)
        kernels.append(y * kernel)

    return space, np.sum(np.asarray(kernels), axis=0)


iterpolated = []

for row in to_interp:
    _, i = interpolate_row(row)
    iterpolated.append(i)

iterpolated = np.asarray(iterpolated)

plt.imshow(iterpolated)
plt.show()

iterpolated2 = []

for column in iterpolated.T:
    _, i = interpolate_row(column)
    iterpolated2.append(i)

iterpolated2 = np.asarray(iterpolated2).T

plt.imshow(iterpolated2)
plt.show()

result_green = iterpolated2

red[::2].shape

red_row_inter = []
for row in red[::2]:
    _, i = interpolate_row(row[1::2])
    red_row_inter.append(i)

red_row_inter = np.asarray(red_row_inter)
red_row_inter.shape

red_col_inter = []
for col in red_row_inter.T:
    _, i = interpolate_row(col)
    red_col_inter.append(i)

red_col_inter = np.asarray(red_col_inter)
red_col_inter.T.shape

result_red = red_col_inter

blue_row_inter = []
for row in blue[1::2]:
    _, i = interpolate_row(row[::2])
    blue_row_inter.append(i)

blue_row_inter = np.asarray(blue_row_inter)
blue_row_inter.shape

blue_col_inter = []
for col in blue_row_inter.T:
    _, i = interpolate_row(col)
    blue_col_inter.append(i)

blue_col_inter = np.asarray(blue_col_inter)
blue_col_inter.T.shape

result_blue = blue_col_inter

result_red.shape, result_green.T.shape, result_blue.shape

result_image = np.dstack([result_red.T, result_green, result_blue.T])

plt.imshow(result_image)
plt.show()