
import sys

sys.path.append("../src")

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn import metrics

sns.set()

def simple(x):
    return np.sin(x)


def inverted_sin(x):
    return np.sin(np.power(x, -1))


def sign(x):
    return np.sign(np.sin(8 * x))

#_________________Wypisanie samych funkcji___________________

x = np.linspace(1e-6, 2 * np.pi, 10_000)

y_simple = simple(x)
y_sin_func = inverted_sin(x)
y_sgn_func = sign(x)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x, y_simple)
_ = plt.plot(x, y_sin_func)
_ = plt.plot(x, y_sgn_func)

plt.show()

#______________Interpolacja sinusa________________

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2 * np.pi, n_samples)
y = simple(x)

x_interp = np.linspace(1e-6, 2 * np.pi, n_predictions)
y_interp = np.interp(x_interp, x, y)
y_true = simple(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#_____________________Interpolacja sinusa do -1_________________

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2 * np.pi, n_samples)
y = inverted_sin(x)

x_interp = np.linspace(1e-6, 2 * np.pi, n_predictions)
y_interp = np.interp(x_interp, x, y)
y_true = inverted_sin(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#________________________Interpolacja sign________________

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2 * np.pi, n_samples)
y = sign(x)

x_interp = np.linspace(1e-6, 2 * np.pi, n_predictions)
y_interp = np.interp(x_interp, x, y)
y_true = sign(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#______________________Interpolacja sin Spline_____________

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2 * np.pi, n_samples)
y = simple(x)

spline = interpolate.CubicSpline(x, y)
x_interp = np.linspace(1e-6, 2 * np.pi, n_predictions)
y_interp = spline(x_interp)
y_true = simple(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#______________________Interpolacja sinusa do -1 Spline____

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2 * np.pi, n_samples)
y = inverted_sin(x)

spline = interpolate.CubicSpline(x, y)
x_interp = np.linspace(1e-6, 2*np.pi, n_predictions)
y_interp = spline(x_interp)
y_true = inverted_sin(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#______________________Interpolacja sign Spline____________

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2*np.pi, n_samples)
y = sign(x)

spline = interpolate.CubicSpline(x, y)
x_interp = np.linspace(1e-6, 2*np.pi, n_predictions)
y_interp = spline(x_interp)
y_true = sign(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#______________________Interpolacja sin BSpline____________

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2 * np.pi, n_samples)
y = simple(x)

t, c, k = interpolate.splrep(x, y)
spline = interpolate.BSpline(t, c, k)

x_interp = np.linspace(1e-6, 2*np.pi, n_predictions)
y_interp = spline(x_interp)
y_true = simple(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#_______________Interpolacja sinusa do -1 BSpline____

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2 * np.pi, n_samples)
y = inverted_sin(x)

t, c, k = interpolate.splrep(x, y)  # get b-spline representation of a set of points
spline = interpolate.BSpline(t, c, k)

x_interp = np.linspace(1e-6, 2*np.pi, n_predictions)
y_interp = spline(x_interp)
y_true = inverted_sin(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

#_______________Interpolacja sign BSpline____________

n_samples = 100
n_predictions = 10_000

x = np.linspace(1e-6, 2*np.pi, n_samples)
y = sign(x)

t, c, k = interpolate.splrep(x, y)  # get b-spline representation of a set of points
spline = interpolate.BSpline(t, c, k)

x_interp = np.linspace(1e-6, 2*np.pi, n_predictions)
y_interp = spline(x_interp)
y_true = sign(x_interp)

_ = plt.figure(figsize=[12, 8])

_ = plt.plot(x_interp, y_true, label='Real function')
_ = plt.plot(x_interp, y_interp, label='Interpolation', linestyle='dashed')
_ = plt.scatter(x, y, color='red', s=10, label='Measurements')

plt.legend()
plt.show()

