import math


def radial_based_gaussian_function(inputValue, centroid, radius):
    return math.exp(-(norm(inputValue, centroid)**2) / radius ** 2)


def norm(vector1, vector2):
    pass
