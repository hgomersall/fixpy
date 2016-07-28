
from .fixed_point import FixedPointArray, SignedFixedPointArray
import numpy as np

def fixed_point_convolve(a, v, mode='full'):
    '''Works like ``numpy.convolve`` but with ``a`` and ``v`` as fixed point
    arrays, returning a properly computed fixed point array.
    '''
    output_int = np.convolve(a.data, v.data, mode)

    output_fractional_bits = a.fractional_bits + v.fractional_bits

    if (isinstance(a, SignedFixedPointArray)
        or isinstance(v, SignedFixedPointArray)):

        output_type = SignedFixedPointArray

    else:
        output_type = FixedPointArray

    output = output_type(
        output_int, output_fractional_bits, data_already_scaled=True)

    return output
