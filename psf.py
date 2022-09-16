"""
File: psf.py

Copyright (C) 2012-2020 Euclid Science Ground Segment

This file is part of LE3_VMSP_ID.

This library is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with this library.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

class PSF:
    """Gaussian PSF.
    This class contains methods for the PSF.

    """

    _default_params = {
        'psf_amp': 0.781749,
        'psf_scale1': 0.84454,  # pixel coordinates
        'psf_scale2': 3.64980,  # pixel coordinates
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        self.params.update(kwargs)

    def sample(self, count):
        """Sample.
        This function performs sampling.

        """

        coord = np.random.normal(0, 1, (2, count))

        n1 = int(np.round(count * self.params['psf_amp']))

        coord[:, :n1] *= self.params['psf_scale1']
        coord[:, n1:] *= self.params['psf_scale2']

        return coord
