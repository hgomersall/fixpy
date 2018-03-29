
from .fixed_point import FixedPointArray, SignedFixedPointArray
import numpy as np


def fixed_point_convolve(a, v, mode='full'):
    '''Works like ``numpy.convolve`` but with ``a`` and ``v`` as fixed point
    arrays, returning a properly computed fixed point array.
    '''
    output_int = np.convolve(a.data, v.data, mode)

    output_fractional_bits = a.fractional_bits + v.fractional_bits

    if (
        isinstance(a, SignedFixedPointArray) or
        isinstance(v, SignedFixedPointArray)
    ):
        output_type = SignedFixedPointArray
    else:
        output_type = FixedPointArray

    output = output_type(
            output_int,
            output_fractional_bits, data_already_scaled=True
        )

    return output


def log(fp_array, **kwargs):
    '''
    Wrapper function that returns log of the fixed point array
    '''
    return fp_array.log(**kwargs)


def exp(fp_array, **kwargs):
    '''
    Wrapper function that returns exp of the fixed point array
    '''
    return fp_array.exp(**kwargs)


def square(fp_array, **kwargs):
    '''
    Wrapper function that returns square of the fixed point array
    '''
    return fp_array.square(**kwargs)


def sqrt(fp_array, **kwargs):
    '''
    Wrapper function that returns sqrt of the fixed point array
    '''
    return fp_array.sqrt(**kwargs)


def sin(fp_array, **kwargs):
    '''
    Wrapper function that returns sin of the fixed point array
    '''
    return fp_array.sin(**kwargs)


def cos(fp_array, **kwargs):
    '''
    Wrapper function that returns cos of the fixed point array
    '''
    return fp_array.cos(**kwargs)


def tan(fp_array, **kwargs):
    '''
    Wrapper function that returns tan of the fixed point array
    '''
    return fp_array.tan(**kwargs)


def sum(fp_array, **kwargs):
    '''
    Wrapper function that returns sum of array elements over a given axis.
    '''
    return fp_array.sum(**kwargs)


def resize(fp_array, shape):
    '''
    Wrapper function that returns the resized array
    '''
    return FixedPointArray(
        np.resize(fp_array.as_floating_point(), shape),
        fractional_bits=fp_array.fractional_bits
    )


def any(fp_array, **kwargs):
    '''
    Wrapper function that returns whether any array elements evaluate to True
    along a given axis
    '''
    return fp_array.any(**kwargs)


def all(fp_array, **kwargs):
    '''
    Wrapper function that returns whether all array elements evaluate to True
    along a given axis
    '''
    return fp_array.all(**kwargs)


def zeros_like(fp_array, **kwargs):
    '''
    Wrapper function that returns an array of zeros like the input array
    '''
    return fp_array.zeros_like()


def ones_like(fp_array, **kwargs):
    '''
    Wrapper function that returns an array of ones like the input array
    '''
    return fp_array.ones_like()


def append(fp_array, other_array, **kwargs):
    '''
    Wrapper function that appends the two arrays
    '''
    return fp_array.append(other_array, **kwargs)


def where(condition, x=None, y=None, **kwargs):
    '''
    Defers to numpy where
    '''
    fractional_bits = kwargs.get('fractional_bits', 0)
    if x is not None:
        if isinstance(x, FixedPointArray):
            x = x.as_floating_point()
        if isinstance(y, FixedPointArray):
            y = y.as_floating_point()

        if fractional_bits:
            return FixedPointArray(np.where(condition, x, y), fractional_bits)
        else:
            return np.where(condition, x, y)
    else:
        return np.where(condition)


def squeeze(fp_array, axis=None):
    '''
    Wrapper for FixedPointArray squeeze
    '''
    return fp_array.squeeze(axis)


def vstack(tup):
    '''
    Stack arrays in sequence vertically (row wise).
    '''
    fractional_bits = max(x.fractional_bits for x in tup)
    tup = (x.as_floating_point() for x in tup)
    return FixedPointArray(np.vstack(tup), fractional_bits)


def hstack(tup):
    '''
    Stack arrays in sequence horizontally (col wise).
    '''
    fractional_bits = max(x.fractional_bits for x in tup)
    tup = (x.as_floating_point() for x in tup)
    return FixedPointArray(np.hstack(tup), fractional_bits)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    '''
    Defers to numpy apply along axis
    '''
    fractional_bits = kwargs.get('fractional_bits', arr.fractional_bits)
    return FixedPointArray(
        np.apply_along_axis(func1d, axis, arr.as_floating_point(),
                            *args, **kwargs),
        fractional_bits
    )


def power(x1, x2, **kwargs):
    if not isinstance(x1, FixedPointArray):
        raise TypeError("x1 must be of type FixedPointArray")
    if not isinstance(x2, FixedPointArray):
        x2 = FixedPointArray(x2, fractional_bits=x1.fractional_bits)
    output_frac_bits = max(x1.fractional_bits, x2.fractional_bits)
    return FixedPointArray(
        np.power(x1.as_floating_point(), x2.as_floating_point(), **kwargs),
        x1.fractional_bits
    )


def abs(fd_array):
    return FixedPointArray(
        fd_array.as_floating_point(),
        fractional_bits=fd_array.fractional_bits
    )
