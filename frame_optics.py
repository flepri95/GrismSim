import numpy as np
from scipy import interpolate
from astropy.io import fits
from astropy.wcs import WCS

import consts
import psf


class FrameOptics:
    _default_params = {
        'pointing_center': (0, 0),
        'telescope_area_cm2': 1e4,
        'exptime': 550,
        'pixel_scale': 0.3,
        'det_width': 2048,
        'det_height': 2048,
        'dispersion': 13.4,
        'x_0': 0,
        'y_0': 0,
        'wavelength_0': 15200,
        'sigma2_det': 2.33,
        'dispersion_angle': 4,
        'transmission_file': "data/grism_1_transmission.txt",
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        self.params.update(kwargs)

        self.sens = self.params['telescope_area_cm2'] * self.params['exptime']
        self.sens /= consts.planck * consts.c

        self.load_transmission()

        self.psf = psf.PSF()
        self.distortion = None
        
    def arcsec_to_pixel(self, x):
        """Convert values in arcsec to pixel units"""
        return x / self.params['pixel_scale']

    def pixel_to_arcsec(self, x):
        """Convert values in pixels to arcsec units"""
        return x * self.params['pixel_scale']

    def load_from_fits(self, filename, i=1, j=1):
        """ """
        with fits.open(filename) as hdul:
            header = hdul[0].header
            # GWA_POS = 'RGS000  '
            # GWA_TILT
            grism_name = header['GWA_POS']
            grism_list = {
                'RGS000': 0.,
                'RGS180': 180.,
            }
            grism_angle = grism_list[grism_name]
            tilt = float(header['GWA_TILT'])

            self.params['dispersion_angle'] = grism_angle - tilt

            hdu = hdul[f"DET{i}{j}.SCI"]
            self._wcs = WCS(hdu)


    @property
    def wcs(self):
        try:
            return self._wcs
        except AttributeError:
            self._wcs = WCS(naxis=2)
            self._wcs.wcs.crpix = (
                self.params['det_width']/2,
                self.params['det_height']/2
            )
            self._wcs.wcs.crval = self.params['pointing_center']
            self._wcs.wcs.cdelt = np.array(
                [self.params['pixel_scale']/3600, self.params['pixel_scale']/3600]
            )
            self._wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return self._wcs

    #def radec_to_pixel(self, ra, dec):
    #    """ """
    #    return self.wcs.wcs_world2pix(ra, dec, 0)   
    
    def radec_to_pixel(self, ra, dec):
        """ """
        x, y = self.wcs.wcs_world2pix(ra, dec, 1)
        x -= 0.5
        y -= 0.5
        if self.distortion is None:
            return x, y
        else:
            return x, y

    def lineflux_to_counts(self, flux, wavelength):
        """Convert flux to photon counts."""
        t = self.transmission(wavelength)
        return flux * t * self.sens * wavelength

    def flux_to_counts(self, flux, wavelength):
        """Convert flux to photon counts."""
        step = wavelength[1] - wavelength[0]
        t = self.transmission(wavelength)
        return flux * t * step * self.sens * wavelength

    def counts_to_flux(self, counts, wavelength):
        """ """
        step = wavelength[1] - wavelength[0]
        t = self.transmission(wavelength)
        return counts / (t * step * self.sens * wavelength)

    def sensitivity(self, wavelength):
        """ """
        step = wavelength[1] - wavelength[0]
        t = self.transmission(wavelength)
        return  t * step * self.sens * wavelength

    def load_transmission(self):
        """ """
        x, y = np.loadtxt(self.params['transmission_file'], unpack=True)
        self.transmission = interpolate.interp1d(x, y, bounds_error=False, fill_value=0)

    def wavelength_to_pix(self, wavelength):
        """Compute the pixel coordinate offset from the wavelength"""
        d = (wavelength - self.params['wavelength_0']) / self.params['dispersion']

        # rotate by the dispersion angle
        theta = np.pi/180 * self.params['dispersion_angle']
        dx = d * np.cos(theta)
        dy = -d * np.sin(theta)

        return dx + self.params['x_0'], dy + self.params['y_0']

    def footprint(self):
        return self.wcs.calc_footprint(axes=(self.params['det_width'], self.params['det_height']))