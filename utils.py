"""
File: utils.py

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
from scipy import interpolate




def ensurelist(x):
    """
    Parameters
    ----------
    x : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    if isinstance(x, str):
        return [x]

    try:
        len(x)
    except TypeError:
        return [x]

    return x

def is_number(s):
    """
    Parameters
    ----------
    s : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    try:
        float(s)
        return True
    except TypeError:
        return False
    except ValueError:
        return False
