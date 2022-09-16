from xml.sax.handler import property_declaration_handler
import numpy as np
from scipy import interpolate

import sample_dist
import profile_sampler



class Galaxy:
    _default_params = {
        'ra': 0,
        'dec': 0,
        'redshift': 1.0,
        'fwhm_arcsec': 1.,
        'profile': 'gaussian',
        'disk_r50': 1.,
        'bulge_r50': 1.,
        'bulge_fraction': 0.2,
        'pa': 0,
        'axis_ratio': 1,
        'obs_wavelength_step': 300,
        'obs_wavelength_range': (12000., 19000.),
        'continuum_params': (15000, -1e-5, -18),
        'fluxes_emlines' : np.zeros(11),
        'velocity_disp': 1.2e15, #angstrom/s
        'nlines': 11,
        'ID': 0,
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown galaxy parameter {key}")

        self.wavelength = np.arange(
            self.params['obs_wavelength_range'][0],
            self.params['obs_wavelength_range'][1],
            self.params['obs_wavelength_step']
        )

        self.init_sed()

    def copy(self):
        params = self.params.copy()
        return type(self)(**params)


 #   def init_emline(self):

 #       redshift = self.params['redshift']
 #       wavelength_obs = (1 + self.params['redshift']) * consts.rest_wavelength_ha
 #       sigma_size = self.params['velocity_disp']/consts.c*wavelength_obs
 #       ampl = self.params['fha']/np.sqrt(2 * np.pi * sigma_size**2)

 #       emline_range=(wavelength_obs-5*sigma_size, wavelength_obs+5*sigma_size)
 #       self.emline_wave = np.arange(
 #           *emline_range,
 #           sigma_size/10.)

 #       n=500

 #       self.emline = ampl * np.random.normal(wavelength_obs, sigma_size, n)
        #self.emline = ampl * np.exp(-(self.emline_wave - wavelength_obs)**2/(2 * sigma_size**2))

    def init_sed(self):
        """Take care of units"""
        x, a, b = self.params['continuum_params']
        self.sed = 10**((self.wavelength - x) * a + b)

    @property
    def profile_sampler(self):
        try:
            return self._profile_sampler
        except AttributeError:
            if self.params['profile'][0].lower() == 'g': # profile name starts with g for gaussian
                # profile is gaussian
                # print("initalizing gaussian profile")
                self._profile_sampler = self._sample_gaussian
            else:
                # profile is bulgydisk
                # print("initalizing bulgy disk profile")
                self._profile_sampler = self._sample_bulgy_disk
        return self._profile_sampler

    @property
    def sed(self):
        return self._sed

    @sed.setter
    def sed(self, y):
        self.sed_params = y
        self._sed = sample_dist.SampleDistribution(self.wavelength, y)

    @property
    def halflight_radius(self):
        """Return the half-light radius in arcsec"""
        try:
            return self._halflight_radius
        except AttributeError:
            pass
        # sample the profile to compute the half light radius
        coord_xy = self.profile_sampler(1e4)
        r = np.sqrt(coord_xy[:, 0]**2 + coord_xy[:, 1]**2)
        self._halflight_radius = np.median(r)

        return self._halflight_radius

 #   @property
 #   def emline(self):
 #       return self._emline

 #   @emline.setter
 #   def emline(self,y):

 #       redshift = self.params['redshift']
 #       wavelength_obs = (1 + self.params['redshift']) * consts.rest_wavelength_ha
 #       sigma_size = self.params['velocity_disp']/consts.c*wavelength_obs
 #       ampl = self.params['fha']/np.sqrt(2 * np.pi * sigma_size**2)

 #       emline_range=(wavelength_obs-5*sigma_size, wavelength_obs+5*sigma_size)
 #       self.emline_wave = np.arange(
 #           *emline_range,
 #           sigma_size/10.)

 #       n=500

 #       self.emline = ampl * np.random.normal(wavelength_obs, sigma_size, n)
 #       self._emline = interpolate.interp1d(self.emline_wave,y)

    def _sample_gaussian(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.fwhm_gaussian_sampler(n) * self.params['fwhm_arcsec']

    def _sample_bulge(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.bulge_sampler(n) * self.params['bulge_r50']

    def _sample_disk(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.disk_sampler(n) * self.params['disk_r50']

    def _sample_bulgy_disk(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))

        n_bulge = int(np.round(self.params['bulge_fraction'] * n))
        n_disk = n - n_bulge
        samples = []
        if n_bulge > 0:
            samples.append(self._sample_bulge(n_bulge))
        if n_disk > 0:
            samples.append(self._sample_disk(n_disk))

        return np.vstack(samples)

    def sample_image(self, n):
        """Draw samples from the galaxy image"""
        coord_xy = self.profile_sampler(n)

        root_axis_ratio = np.sqrt(self.params['axis_ratio'])
        coord_xy *= np.array([root_axis_ratio, 1./root_axis_ratio])

        # rotate to position angle
        cosangle = np.cos(np.deg2rad(self.params['pa']))
        sinangle = np.sin(np.deg2rad(self.params['pa']))
        rotation_mat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])
        coord_xy = coord_xy @ rotation_mat

        coord_xy /= 3600 # convert arcsec to deg

        x, y = coord_xy.transpose()

        # rotate to RA, Dec on sky
        x = self.params['ra'] + x / np.cos(np.deg2rad(self.params['dec']))
        y = self.params['dec'] + y

        return x, y