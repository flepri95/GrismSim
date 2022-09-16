import numpy as np
from scipy import interpolate, ndimage
from astropy.io import fits
from tqdm.auto import tqdm

import utils
import sample_dist
import consts
import histogram

from numba import int64
from numba.typed import Dict

class SpectralImager:

    params = {
        'wavelength_range': (12000, 19000),
        'wavelength_step': 1.,
        'nphot_max': 1000000,
    }

    def __init__(self, optics, **kwargs):
        """ """
        self.optics = optics

        self.params.update(kwargs)

        self.wave = np.arange(
            self.params['wavelength_range'][0],
            self.params['wavelength_range'][1],
            self.params['wavelength_step']
        )

        self.sigma = np.sqrt(self.optics.params['sigma2_det'] * self.optics.params['exptime'])

        self.init_image()


    def init_image(self):
        """ """
        bin_y = np.arange(0, self.optics.params['det_height']+1, 1, dtype='d')
        bin_x = np.arange(0, self.optics.params['det_width']+1, 1, dtype='d')
        self.pixel_grid = (bin_y, bin_x)

    def sample_spectrum_cat(self, galaxy, spectrum, wave_grid):

        flux_spectrum = spectrum
        #flux_spectrum[flux_spectrum<0] = 0

        counts_spectrum = self.optics.flux_to_counts(flux_spectrum, wave_grid)

        func = sample_dist.SampleDistribution(wave_grid, counts_spectrum)

        counts = np.random.poisson(np.sum(counts_spectrum))
        if counts == 0:
            return np.array([]), 0

        weight = 1
        if counts > self.params['nphot_max']:
            weight = counts / self.params['nphot_max']
            counts = self.params['nphot_max']
            print(f"alert {weight}")

        try:
            samples = func.sample(counts)
        except ValueError:
            return np.array([]), 0

        return samples, weight

    def sample_cat(self, galaxy, spectrum, wave_grid):
        """ """
        wavelength, weight = self.sample_spectrum_cat(galaxy, spectrum, wave_grid)

        if len(wavelength) == 0:
            return np.array([]), np.array([]), 0

        x, y = self.optics.wavelength_to_pix(wavelength)

        ra, dec = galaxy.sample_image(
            len(x)        )

        xg, yg = self.optics.radec_to_pixel(ra, dec)

        xpsf, ypsf = self.optics.psf.sample(len(x))

        return x + xg + xpsf, y + yg + ypsf, weight

    def make_image_cat(self, galaxy_list, spectra, wave_grid, noise=True):
        """Builds image and variance image"""
        image = np.zeros((self.optics.params['det_height'], self.optics.params['det_width']), dtype='d')

        for gal_i, g in enumerate(tqdm(galaxy_list)):

            x, y, weight = self.sample_cat(g, spectra[gal_i], wave_grid)
            im = histogram.histogram2d_accumulate(
                y,
                x,
                weight,
                bins_y=self.pixel_grid[0],
                bins_x=self.pixel_grid[1],
                hist=image,
                mask_dict=None
            )

        # poisson variance is equal to mean
        var_image = image + self.sigma**2

        if noise:
            # add detector noise background
            image += np.random.normal(0, self.sigma, image.shape)

        return image, var_image


    def sample_spectrum(self, galaxy):
        """Generate samples of wavelength"""

        flux_spectrum = galaxy.sed(self.wave)# + galaxy.emline(self.wave)
        flux_spectrum[flux_spectrum<0] = 0

#         print(flux_spectrum)
        counts_spectrum = self.optics.flux_to_counts(flux_spectrum, self.wave)

        func = sample_dist.SampleDistribution(self.wave, counts_spectrum)

        counts = np.random.poisson(np.sum(counts_spectrum))

        if counts == 0:
            return np.array([]), 0
#         print(f"counts {counts}")

        weight = 1
        if counts > self.params['nphot_max']:
            weight = counts / self.params['nphot_max']
            counts = self.params['nphot_max']
            print(f"alert {weight}")

        try:
            samples = func.sample(counts)
        except ValueError:
            return np.array([]), 0

        return samples, weight

    def sample_emline(self, galaxy, line):

        redshift = galaxy.params['redshift']
        wavelength_obs = (1 + redshift) * consts.lines[line]

       # else:
        sigma_size = galaxy.params['velocity_disp']/consts.c*wavelength_obs

        if wavelength_obs<galaxy.params['obs_wavelength_range'][0] or wavelength_obs>galaxy.params['obs_wavelength_range'][1]:
            counts_emline = 0
        else:
            counts_emline = self.optics.lineflux_to_counts(galaxy.params['fluxes_emlines'][line], wavelength_obs)
            counts_emline = np.random.poisson(counts_emline)

        weight = 1
        if counts_emline > self.params['nphot_max']:
            weight = counts_emline / self.params['nphot_max']
            counts_emline = self.params['nphot_max']
            print(f"alert {weight}")

        samples_emline = np.random.normal(wavelength_obs, sigma_size, counts_emline)

        return samples_emline, weight

    def sample(self, galaxy):
        """ """
        wavelength, weight = self.sample_spectrum(galaxy)

        if len(wavelength) == 0:
            return np.array([]), np.array([]), 0

        x, y = self.optics.wavelength_to_pix(wavelength)

        ra, dec = galaxy.sample_image(
            len(x)
        )

        xg, yg = self.optics.radec_to_pixel(ra, dec)

        xpsf, ypsf = self.optics.psf.sample(len(x))

        #return x + xg, y + yg, weight
        return x + xg + xpsf, y + yg + ypsf, weight

    def sample_line(self, galaxy, line):
        """ """
        wavelength1, weight1 = self.sample_emline(galaxy, line)

        if len(wavelength1) == 0:
            return np.array([]), np.array([]), 0

        x1, y1 = self.optics.wavelength_to_pix(wavelength1)

        ra, dec = galaxy.sample_image(
            len(x1)
        )

        xg, yg = self.optics.radec_to_pixel(ra, dec)

        xpsf, ypsf = self.optics.psf.sample(len(x1))
        
        #return x1 + xg, y1 + yg, weight1
        return x1 + xg + xpsf, y1 + yg + ypsf, weight1

    def make_image(self, galaxy_list, mask=None, noise=True, return_var=True):
        """Builds image and variance image"""

        if mask is not None:
            sel, = np.where(mask.flat > -1)
            # mask_lookup = histogram.make_reverse_lookup(sel)
            image = np.zeros((1, len(sel)), dtype='d')
        else:
            image = np.zeros((self.optics.params['det_height'], self.optics.params['det_width']), dtype='d')


        for g in galaxy_list:

            x, y, weight = self.sample(g)
            histogram.histogram2d_accumulate(
                y,
                x,
                weight,
                bins_y=self.pixel_grid[0],
                bins_x=self.pixel_grid[1],
                hist=image,
                mask_dict=mask
            )

            for l in range(g.params['nlines']):
                x, y, weight = self.sample_line(g, l)
                histogram.histogram2d_accumulate(
                    y,
                    x,
                    weight,
                    bins_y=self.pixel_grid[0],
                    bins_x=self.pixel_grid[1],
                    hist=image,
                    mask_dict=mask
                )

        # remove axes with length 1
        image = np.squeeze(image)

        if return_var:
            # poisson variance is equal to mean
            var_image = image + self.sigma**2

        if noise:
            # add detector noise background
            image += np.random.normal(0, self.sigma, image.shape)

        if return_var:
            return image, var_image
        else:
            return image

    def make_mask(self, galaxy_list, input_mask=None, width=3, iterations_min=3):
        """ """
        galaxy_list = utils.ensurelist(galaxy_list)

        mask_image = np.zeros(
            (
                self.optics.params['det_height'],
                self.optics.params['det_width']
            ),
            dtype=np.bool
        )

        for gal_i in range(len(galaxy_list)):
            x0, y0 = self.optics.radec_to_pixel(
                galaxy_list[gal_i].params['ra'], galaxy_list[gal_i].params['dec'])

            x, y = self.optics.wavelength_to_pix(self.wave)
            x += x0
            y += y0

            weight=1

            im = histogram.histogram2d(y, x, weight, bins_y=self.pixel_grid[0], bins_x=self.pixel_grid[1])

            radius = self.optics.arcsec_to_pixel(
                galaxy_list[gal_i].halflight_radius
            )

            iterations = max(iterations_min, int(np.round(width*radius)))

            im = ndimage.binary_dilation(
                im,
                structure=ndimage.generate_binary_structure(2, 2),
                iterations=iterations
            )

            mask_image += im

        if input_mask is not None:
            mask_image *= input_mask

        # values outside of mask are -1
        index_mask = np.zeros(mask_image.shape, dtype=int) - 1
        # values inside of mask are set to an index 0,1,2,3...
        sel = mask_image > 0
        index_mask[sel] = np.arange(np.sum(sel))

        return index_mask

    # def write(self, filename, **kwargs):
    #     """Write image to a FITS file"""
    #     hdu = fits.PrimaryHDU()
    #     image_hdu = fits.ImageHDU(data=self.image, header=self.optics.wcs.to_header())
    #     hdul = fits.HDUList([hdu, image_hdu])
    #     hdul.writeto(filename, **kwargs)
