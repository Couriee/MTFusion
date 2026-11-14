"""NumPy implementations of the five image fusion metrics used in Table 1."""

import numpy as np


def EN_function(image_array):
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 255))
    histogram = histogram / float(np.sum(histogram))
    return -np.sum(histogram * np.log2(histogram + 1e-7))


def SF_function(image):
    image_array = np.array(image)
    row_frequency = np.sqrt(np.mean(np.diff(image_array, axis=0) ** 2))
    column_frequency = np.sqrt(np.mean(np.diff(image_array, axis=1) ** 2))
    return np.sqrt(row_frequency ** 2 + column_frequency ** 2)


def SD_function(image_array):
    height, width = image_array.shape
    mean = np.mean(image_array)
    return np.sqrt(np.sum((image_array - mean) ** 2) / (height * width))


def AG_function(image):
    height, width = image.shape
    grad_y, grad_x = np.gradient(image)
    magnitude = np.sqrt((np.square(grad_x) + np.square(grad_y)) / 2)
    return np.sum(magnitude) / ((height - 1) * (width - 1))


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


def SCD_function(A, B, F):
    return corr2(F - B, A) + corr2(F - A, B)
