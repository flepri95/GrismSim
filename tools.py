import numpy as np

from astropy.table import Table

import tqdm

    
    
def simulate_source(x, y, sigma_size, flux, wavelength_obs, nphot=1000, nphot_max=10000):
    """ """
    # generate a Gaussian blob by shooting photons
    # a limit is put on the number of photons to save time and memory
    n = int(nphot)
    weight = 1
    if n > nphot_max:
        weight = n / nphot_max
        n = nphot_max

    # generate the image cutout on the detector grid
    sub_y = 100
    sub_x = 100
    cx = int(x) - sub_x//2
    cy = int(y) - sub_y//2
        
    # generate gaussian blob
    xx, yy = np.random.normal(0, sigma_size, (2, n))
    xx += x
    yy += y

    pixel_grid = np.arange(0, sub_y, 1), np.arange(0, sub_x, 1)
    
    h, ey, ex = np.histogram2d(yy-cy, xx-cx, bins=pixel_grid)
    h *= weight

    corner = cx, cy
    
    return h, corner
    
    
def insert_subim(im, subim, corner):
    """ """
    cx, cy = corner
    im_shape = im.shape
    sub_shape = subim.shape

    if cx < 0:
        subim = subim[:, -cx:]
        cx = 0
        sub_shape = subim.shape

    if cy < 0:
        subim = subim[-cy:, :]
        cy = 0
        sub_shape = subim.shape
        
    if cx + sub_shape[1] > im_shape[1]:
        d = cx + sub_shape[1] - im_shape[1]
        subim = subim[:,:d]
        sub_shape = subim.shape

    if cy + sub_shape[0] > im_shape[0]:
        d = cy + sub_shape[0] - im_shape[0]
        subim = subim[:d,:]
        sub_shape = subim.shape

    im[cy:cy+sub_shape[0], cx:cx+sub_shape[1]] += subim
    
    
def cut_subim(im, corner, shape):
    """ """
    cx, cy = corner
    im_shape = im.shape
    
    subim = np.zeros(shape, dtype=im.dtype)

    dx = cx + shape[1]
    dy = cy + shape[0]

    ax = 0
    ay = 0
    corner_flag = 0
    if cx < 0:
        corner_flag=1
        ax = -cx
        cx = 0

    if cy < 0:
        ay = -cy
        cy = 0
        
    
    if dx > im_shape[1]:
        dx = im_shape[1]
    
    if dy > im_shape[0]:
        dy = im_shape[0]
        
    by = ay + dy - cy
    bx = ax + dx - cx

#     print(ay,by, ax, bx)
#     print(cy,dy, cx,dx)
    try:
        subim[ay:by,ax:bx] = im[cy:dy,cx:dx]
    except ValueError:
        print(f" corner {corner}")
        print(f"corner flag {corner_flag}")
        print(subim.shape, subim[ay:by,ax:bx].shape)
        print(im.shape, im[cy:dy,cx:dx].shape)
        print(ay, by, by-ay)
        print(ax, bx, bx-ax)
        print(cy, dy, dy-cy)
        print(cx, dx, dx-cx)
        raise
#     subim = im[cy:dy,cx:dx]
    return subim
    
    
def ensurearray(x):
    try:
        len(x)
    except TypeError:
        return np.array([x])

    return x

    
def make_image(cat, wcs, grism_orientation, nphot_max=10000, noise=True):
    """ """
    im = np.zeros((det_height, det_width))
    bins = np.arange(0, det_height+1, 1)
    pixel_grid = (bins, bins)
    
    wavelength_obs = (1+cat['Z']) * rest_wavelength_halpha

    nphot = flux_to_electron_count(cat['FLUX_HALPHA'], wavelength_obs)

    sigma_size = cat['RADIUS'] / pixel_scale * 2 / 2.355
    
    x, y = wcs.wcs_world2pix(cat['RA'], cat['DEC'], 0)    
    dx, dy = wavelength_to_pix(wavelength_obs, grism_orientation)

    x += dx
    y += dy
    
    x = ensurearray(x)
    y = ensurearray(y)
    sigma_size = ensurearray(sigma_size)
    nphot = ensurearray(nphot)
    wavelength_obs = ensurearray(wavelength_obs)
    
    for obj_i in range(len(x)):        
        if x[obj_i] < 0:
            continue
            
        if y[obj_i] < 0:
            continue
            
        if x[obj_i] > im.shape[1]:
            continue
        
        if y[obj_i] > im.shape[0]:
            continue

        
        subim, corner = simulate_source(
            x[obj_i], 
            y[obj_i], 
            sigma_size[obj_i],
            ensurearray(cat['FLUX_HALPHA'])[obj_i],
            wavelength_obs[obj_i]
        )
        
        insert_subim(im, subim, corner)
                
    if noise:
        # add detector noise background
        im += np.random.normal(0, np.sqrt(sigma2_det * t_int), im.shape)
    return im

