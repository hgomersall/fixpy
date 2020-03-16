
import numpy as np

class MockSparse(object):
    '''A class that can be used for missing SciPy. It always returns false
    which means the sparse portions of the code are bypassed.
    '''
    def issparse(self, *args, **kwargs):
        return False

try:
    from scipy import sparse
except ImportError:
    sparse = MockSparse()

import math

f_round = np.round

class FixedPointArray(object):

    # Tell numpy that we're happy to process an array.
    __array_priority__ = 1.0

    def __init__(self, data, fractional_bits, data_already_scaled=False):
        '''A ``FixedPointArray`` is created from the data provided in the
        first argument, with the number of bits allocated to the fractional
        part of the array dictated by ``fractional_bits``.

        By default, the array should be one that needs to be scaled to build
        the correct internal representation of the data. This can be
        disabled with the ``data_already_scaled`` flag, in which case it is
        assumed that the data passed in has the correct scaling already. This
        is largely useful for constructing arrays from data that has been
        extracted from a pre-existing FixedPointArray and we want to put it
        back into one (it is used extensively inside the class).

        If ``data_already_scaled`` is ``True``, then no copy is made of the
        data array.

        If data_already_scaled is ``True`` and the data contains a fractional
        part, a ``ValueError`` is raised. This is to protect against erroneous
        arguments.
        '''

        if isinstance(data, FixedPointArray):

            if fractional_bits != data.fractional_bits:
                scaling_fractional_bits = (
                    fractional_bits - data.fractional_bits)

                data = f_round(data.data.copy() * 2**scaling_fractional_bits)
            else:
                data = data.data

            data_already_scaled = True

        if not sparse.issparse(data):
            data = np.asarray(data)
            is_complex = np.iscomplexobj(data)
        else:
            is_complex = np.iscomplexobj(data.data)


        if is_complex:
            storage_type = 'complex128'
        else:
            storage_type = 'float64'

        if not data_already_scaled:
            scaling = 2**fractional_bits
            self.data = (
                    f_round(data * scaling).astype(storage_type))
        else:
            # The following returns true for data containing fractional parts.
            # It is robust to complex data and to sparse arrays (not many
            # solutions are!).
            if len(np.atleast_1d(f_round(data) - data).nonzero()[0]) != 0:
                raise ValueError('When data_already_scaled is set to True, '
                                 'the data should have zero fractional part. '
                                 'i.e. it should be all integers (though not '
                                 'necessarily an integer type).')

            self.data = data

        self.fractional_bits = fractional_bits
        self.storage_type = storage_type

    @property
    def max_integer_bits(self):
        '''Returns the maximum number of bits needed to encode the integer
        part of data (based on the largest single value in the array).
        '''
        max_val = max(
                np.abs(self.data.real).max(),
                np.abs(self.data.imag).max())

        max_int = math.floor(max_val * 2**-self.fractional_bits)

        if max_int > 0:
            max_integer_bits = int(math.floor(math.log(max_int, 2)) + 1)
        else:
            max_integer_bits = 0

        return max_integer_bits

    @property
    def max_bits(self):
        '''Returns the maximum number of bits needed to encode all the digits
        in the data structure with the desired precision, including a sign
        bit if it is necessary.
        '''

        try:
            # For the sparse matrix
            zero_array = (self.data.getnnz() == 0)
        except AttributeError:
            zero_array = np.all(self.data == 0.0)

        def has_negative_part(array):
            real_neg = array.real < 0.0
            imag_neg = array.imag < 0.0

            try:
                return (real_neg.getnnz() + imag_neg.getnnz()) > 0
            except AttributeError:
                return np.any(real_neg) or np.any(imag_neg)

        if zero_array:
            max_bits = self.fractional_bits
        elif has_negative_part(self.data):
            # We have negative values
            max_bits = self.max_integer_bits + self.fractional_bits + 1
        else:
            max_bits = self.max_integer_bits + self.fractional_bits

        return max_bits


    def __getitem__(self, _slice):

        sliced_data = self.data[_slice]
        return self.__class__(sliced_data,
                fractional_bits=self.fractional_bits,
                data_already_scaled=True)

    def __setitem__(self, _slice, value):

        scaling = 2**(self.fractional_bits - value.fractional_bits)

        self.data[_slice] = f_round(value.data * scaling)

    def __eq__(self, other):

        if not isinstance(other, type(self)):
            other = type(self)(other, fractional_bits=self.fractional_bits)

        return self.as_floating_point() == other.as_floating_point()

    def __ne__(self, other):

        if not isinstance(other, type(self)):
            other = type(self)(other, fractional_bits=self.fractional_bits)

        return self.as_floating_point() != other.as_floating_point()

    def __len__(self):
        return self.data.__len__()

    def __mul__(self, other):
        '''Point-wise multiply the elements. The number of fractional bits
        used for the output is the maximum of the fractional bits of the two
        inputs.
        '''
        if not isinstance(other, type(self)):
            other = type(self)(other, fractional_bits=self.fractional_bits)

        output_frac_bits = max(other.fractional_bits, self.fractional_bits)
        scaling_bits = min(other.fractional_bits, self.fractional_bits)

        # The post scaling is determined by the fractional_bits of
        # other. This is so we have the fractional bits of the solution
        # the same as this self.
        post_scaling = 2**-scaling_bits

        if sparse.issparse(self.data):
            result = f_round(self.data.multiply(other.data) *
                             np.float64(post_scaling))
        else:
            result = f_round(self.data * other.data *
                             np.float64(post_scaling))

        output = self.__class__(result,
                fractional_bits=output_frac_bits,
                data_already_scaled=True)

        return output

    def __rmul__(self, other):

        return self*other

    def __neg__(self):

        return self.__class__(-self.data,
                              fractional_bits=self.fractional_bits,
                              data_already_scaled=True)

    def __add__(self, other):

        if not isinstance(other, type(self)):
            other = type(self)(other, fractional_bits=self.fractional_bits)

        output_frac_bits = max(other.fractional_bits, self.fractional_bits)

        # Firstly, shift the numbers to align the decimal point.
        if self.fractional_bits < output_frac_bits:
            self_data = f_round(
                self.data*
                2**(output_frac_bits - self.fractional_bits))

            other_data = other.data

        else:
            other_data = f_round(
                other.data*
                2**(output_frac_bits - other.fractional_bits))

            self_data = self.data

        output = self.__class__(self_data + other_data,
                    fractional_bits=output_frac_bits,
                    data_already_scaled=True)

        return output


    def __sub__(self, other):

        if not isinstance(other, type(self)):
            other = type(self)(other, fractional_bits=self.fractional_bits)

        output_frac_bits = max(other.fractional_bits, self.fractional_bits)

        # Firstly, shift the numbers to align the decimal point.
        if self.fractional_bits < output_frac_bits:
            self_data = f_round(
                self.data*
                2**(output_frac_bits - self.fractional_bits))

            other_data = other.data

        else:
            other_data = f_round(
                other.data*
                2**(output_frac_bits - other.fractional_bits))

            self_data = self.data

        output = self.__class__(self_data - other_data,
                    fractional_bits=output_frac_bits,
                    data_already_scaled=True)

        return output

    def __rsub__(self, other):
        other = type(self)(other, fractional_bits=self.fractional_bits)

        return other - self

    def __truediv__(self, other):
        if not isinstance(other, type(self)):
            other = type(self)(other, fractional_bits=self.fractional_bits)

        fractional_bits = self.fractional_bits
        scaling_bits = other.fractional_bits

        result = f_round((self.data * 2**scaling_bits)/other.data)

        output = self.__class__(result,
                fractional_bits=fractional_bits,
                data_already_scaled=True)

        return output


    def __div__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):

        other = type(self)(other, fractional_bits=self.fractional_bits)

        return other/self


    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __repr__(self):

        repr_string = '<Q%d.%d fixed point array>(\n%s)' % (
            self.max_integer_bits,
            self.fractional_bits,
            self.as_floating_point().__str__())

        return repr_string

    def fix_to_bitwidth(self, bitwidth):
        '''Returns a new fixed point array with the bitwidth fixed to be
        the prescribed value. The truncation process is minimally lossy,
        shifting the data to the left as far as possible before truncation.
        '''
        max_bits = self.max_bits

        int_bitshift = bitwidth - max_bits
        output_fractional_bits = self.fractional_bits + int_bitshift

        int_data = f_round(self.data * 2**int_bitshift)

        output = self.__class__(
            int_data, fractional_bits=output_fractional_bits,
            data_already_scaled=True)

        if output.max_bits > bitwidth:
            # In this case an extra integer bit was created by the
            # rounding
            int_bitshift -= 1
            output_fractional_bits -= 1

            int_data = f_round(self.data * 2**int_bitshift)

            output = self.__class__(
                int_data, fractional_bits=output_fractional_bits,
                data_already_scaled=True)

        return output

    def copy(self):

        return self.__class__(self.data.copy(),
                              fractional_bits=self.fractional_bits,
                              data_already_scaled=True)

    def transpose(self, *axes):

        return self.__class__(self.data.transpose(*axes),
                              fractional_bits=self.fractional_bits,
                              data_already_scaled=True)

    def conj(self):
        '''Return the complex conjugate of the array.
        '''
        return self.conjugate()

    def conjugate(self):
        '''Return the complex conjugate of the array.
        '''
        return self.__class__(self.data.conjugate(),
                fractional_bits=self.fractional_bits,
                data_already_scaled=True)

    def max(self, *args, **kwargs):
        '''Return the maximum value, equivalent to .max on an ndarray.
        '''
        return self.as_floating_point().max(*args, **kwargs)

    def as_floating_point(self):
        '''Return the array as a floating point numpy array.
        '''
        return self.data * 2**(-self.fractional_bits)

    def dot(self, other):
        '''Take the dot product as per numpy.ndarray.dot().

        To prevent overflow, the number of fractional bits used is the maximum
        of the fractional bits of the two inputs.
        '''
        if not isinstance(other, type(self)):
            other = type(self)(other, fractional_bits=self.fractional_bits)

        output_frac_bits = max(other.fractional_bits, self.fractional_bits)
        scaling_bits = min(other.fractional_bits, self.fractional_bits)

        post_scaling = 2**-scaling_bits
        result = f_round(self.data.dot(other.data) * post_scaling)

        output = self.__class__(result,
                fractional_bits=output_frac_bits,
                data_already_scaled=True)

        return output

    def non_truncated_dot(self, other):
        '''Take the dot product as per numpy.ndarray.dot() but without
        truncating the number of fractional bits of the output. That is, the
        number of fractional bits of the output is the sum of the fractional
        bits of the two inputs.
        '''

        if not isinstance(other, FixedPointArray):
            raise ValueError('Invalid array: the argument should be an array'
                             ' of type FixedPointArray.')

        output_frac_bits = other.fractional_bits + self.fractional_bits

        result = self.data.dot(other.data)

        output = self.__class__(result,
                fractional_bits=output_frac_bits,
                data_already_scaled=True)

        return output

    def non_truncated_multiply(self, other):
        '''Multiply this object with the argument but without truncating the
        number of fractional bits of the output. That is, the number of
        fractional bits of the output is the sum of the fractional bits of the
        two inputs.
        '''

        if not isinstance(other, FixedPointArray):
            raise ValueError('Invalid array: the argument should be an array'
                             ' of type FixedPointArray.')

        output_frac_bits = other.fractional_bits + self.fractional_bits

        result = self.data * other.data

        output = self.__class__(result,
                fractional_bits=output_frac_bits,
                data_already_scaled=True)

        return output

    def divide_with_precision(self, other, output_fractional_bits):
        '''Divide this object by the argument, but with the fractional bits
        of the output array set by the output_fractional_bits argument.
        '''
        if not isinstance(other, FixedPointArray):
            raise ValueError('Invalid array: the argument should be an array'
                             ' of type FixedPointArray.')

        scaling_bits = (other.fractional_bits - self.fractional_bits +
                        output_fractional_bits)


        result = f_round((self.data * 2**scaling_bits)/other.data)

        output = self.__class__(result,
                fractional_bits=output_fractional_bits,
                data_already_scaled=True)

        return output

    def reshape(self, shape):
        '''Return the array reshaped to ``shape``.
        '''
        if sparse.issparse(self.data):
            data = np.array(self.data.todense())

        else:
            data = self.data

        return self.__class__(data.reshape(shape),
                fractional_bits=self.fractional_bits,
                data_already_scaled=True)

    def flatten(self):

        if sparse.issparse(self.data):
            data = np.array(self.data.todense())

        else:
            data = self.data

        return self.__class__(data.flatten(),
                fractional_bits=self.fractional_bits,
                data_already_scaled=True)

    def ravel(self):

        if sparse.issparse(self.data):
            data = np.array(self.data.todense())

        else:
            data = self.data

        return self.__class__(data.ravel(),
                fractional_bits=self.fractional_bits,
                data_already_scaled=True)

    def max_abs(self):
        '''Return the maximum of the absolute values of all the values in the
        array.
        '''
        if sparse.issparse(self.data):
            return np.abs(self.as_floating_point()).data.max()

        else:
            return np.max(np.abs(self.as_floating_point()))

    def sum_abs(self):
        '''Return the sum of the absolute values of all the values in the
        array.
        '''
        if sparse.issparse(self.data):
            return np.abs(self.as_floating_point()).data.sum()

        else:
            return np.sum(np.abs(self.as_floating_point()))

    def astype(self, *args, **kwargs):
        '''Returns itself. This is implemented to allow code that depends on
        astype.
        '''
        return self

    @property
    def real(self):
        '''The real part of the array.
        '''
        return  self.__class__(self.data.real,
                fractional_bits=self.fractional_bits,
                data_already_scaled=True)

    @property
    def imag(self):
        '''The imaginary part of the array.
        '''
        return  self.__class__(self.data.imag,
                fractional_bits=self.fractional_bits,
                data_already_scaled=True)

    @property
    def dtype(self):
        '''Return the dtype of the underlying data. This is useful in that
        allows fixed point object to be used in algorithms designed for
        Numpy that expect such a property. Linear operations and algorithms
        should work fine with fixed point arrays.
        '''
        return self.data.dtype

    @property
    def shape(self):
        '''Returns the shape of the array.
        '''
        return self.data.shape


class SignedFixedPointArray(FixedPointArray):

    @property
    def max_integer_bits(self):
        '''Returns the maximum number of bits needed to encode the integer
        part of data (based on the largest single value in the array).
        '''
        max_data_val = max(self.data.real.max(), self.data.imag.max())
        min_data_val = min(self.data.real.min(), self.data.imag.min())

        max_int = math.floor(max_data_val * 2**-self.fractional_bits)

        min_val = min_data_val * 2**-self.fractional_bits
        if min_val > 0:
            min_val = 0

        min_int = math.ceil(min_val)

        if abs(min_int) > abs(max_int):
            if min_int < 0:
                if min_int == min_val:
                    # This captures the special case of -2**n needing only
                    # n bits
                    max_integer_bits = (
                        int(math.floor(math.log(abs(min_int), 2))))
                else:
                    max_integer_bits = int(
                        math.floor(math.log(abs(min_int), 2)) + 1)
            else:
                max_integer_bits = 0

        else:
            if max_int > 0:
                max_integer_bits = int(math.floor(math.log(max_int, 2)) + 1)
            else:
                max_integer_bits = 0

        return max_integer_bits

    @property
    def max_bits(self):
        '''Returns the maximum number of bits needed to encode all the digits
        in the data structure with the desired precision, always including a
        sign bit.
        '''

        try:
            # For the sparse matrix
            zero_array = (self.data.getnnz() == 0)
        except AttributeError:
            zero_array = np.all(self.data == 0.0)

        if zero_array:
            max_bits = self.fractional_bits

        else:
            # always include a sign bit.
            max_bits = self.max_integer_bits + self.fractional_bits + 1

        return max_bits
