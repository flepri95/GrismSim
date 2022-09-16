import numpy as np
import math
import numba


@numba.jit(nopython=True)
def make_reverse_lookup(arr):
    """ """
    return {arr[i]: i for i in range(len(arr))}


@numba.jit(
    "float64[:,:](float64[:],float64[:],float64,float64[:],float64[:])",
    nopython=True
)
def histogram2d(y, x, weight, bins_y, bins_x):
    """Histogram 2D"""
    n = len(x)

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    hist = np.zeros((ny, nx), dtype=numba.float64)

    for i in range(n):
        ind_y = math.floor((y[i] - bins_y[0]) / step_y)
        if ind_y < 0:
            continue
        if ind_y >= ny:
            continue
        ind_x = math.floor((x[i] - bins_x[0]) / step_x)
        if ind_x < 0:
            continue
        if ind_x >= nx:
            continue
        hist[ind_y, ind_x] += weight

    return hist


@numba.jit(
    nopython=True
)
def histogram2d_accumulate(y, x, weight, bins_y, bins_x, hist, mask_dict):
    """Histogram 2D

    mask -> list pixels (100,101,102,103)

    """
    n = len(x)

    if n == 0:
        return

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    for i in range(n):
        ind_y = math.floor((y[i] - bins_y[0]) / step_y)
        if ind_y < 0:
            continue
        if ind_y >= ny:
            continue
        ind_x = math.floor((x[i] - bins_x[0]) / step_x)
        if ind_x < 0:
            continue
        if ind_x >= nx:
            continue

        if mask_dict is not None:
            j = mask_dict[ind_y, ind_x]
            if j > -1:
                hist.flat[j] += weight
        else:
            hist[ind_y, ind_x] += weight

    return
