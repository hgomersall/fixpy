
import unittest
import numpy as np
from fixpy import FixedPointArray, SignedFixedPointArray
from fixpy.fixed_point import f_round, MockSparse

try:
    from scipy import sparse
except ImportError:
    sparse = MockSparse()

import itertools

class TestFixedPointArray(unittest.TestCase):
    '''We should have a fixed point array object that works much like a
    numpy array, only the data it represents is of a fixed point type, with
    the resultant knock-on consequences on any arithmetic operations it can
    be involved with.

    For all rounding operations, the same standard as floating point round
    (IEEE 754-1985) should be used - that is round to the nearest even
    value.
    '''

    def __init__(self, *args, **kwargs):
        super(TestFixedPointArray, self).__init__(*args, **kwargs)

        self.fp_class = FixedPointArray

    def create_test_array_pairs(self, a_shapes, b_shapes,
                                a_frac_bits, b_frac_bits):
        '''Create two sets of test arrays, a and b, with shapes
        and fractional bits given by the arguments provided, which should
        be iterables with one shape and frac_bits for each output double
        (i.e. a real, a complex - see below).

        The shapes iterables should be the same shape and the frac_bits
        iterables should be the same shape.

        For each set, include a real array, a complex array.

        Returns four lists of matched arrays:
        [a, FixedPointArray(a), b, FixedPointArray(b)]
        '''

        a = []
        b = []
        fixed_point_a = []
        fixed_point_b = []

        par_iter = zip(itertools.product(a_shapes, a_frac_bits),
                       itertools.product(b_shapes, b_frac_bits))

        for (a_shape, each_a_frac_bits), (b_shape, each_b_frac_bits) in (
            par_iter):

            if each_a_frac_bits < 0:
                a_scale = 2**-each_a_frac_bits
            else:
                a_scale = 1.0

            if each_b_frac_bits < 0:
                b_scale = 2**-each_b_frac_bits
            else:
                b_scale = 1.0

            a.append(np.random.randn(*a_shape) * a_scale)
            b.append(np.random.randn(*b_shape) * b_scale)

            a.append(a_scale * (
                np.random.randn(*a_shape) + 1j*np.random.randn(*a_shape)))
            b.append(b_scale * (
                np.random.randn(*b_shape) + 1j*np.random.randn(*b_shape)))

            fixed_point_a.append(self.fp_class(a[-2], each_a_frac_bits))
            fixed_point_a.append(self.fp_class(a[-1], each_a_frac_bits))

            fixed_point_b.append(self.fp_class(b[-2], each_b_frac_bits))
            fixed_point_b.append(self.fp_class(b[-1], each_b_frac_bits))

        return a, fixed_point_a, b, fixed_point_b


    def test_real_array(self):
        '''We should store real numbers with sufficient precision.
        '''
        for frac_bits in (0, 4, 15, 24):
            for shape in ((4, 200), (1009,), (5, 65, 70)):
                test_array = np.random.randn(*shape)

                fp_array = self.fp_class(test_array, frac_bits)

                abs_error = np.abs(fp_array.as_floating_point() - test_array)

                self.assertTrue(np.all(abs_error < 2**(-frac_bits-1)))

    def test_data_already_scaled_flag(self):
        '''It should be possible to pass a pre-scaled numpy array.

        This should be done by setting ``True`` a ``data_already_scaled``
        argument. This setting allows for arrays to be processed outside
        of a FixedPointArray (as a numpy array) and then a new
        FixedPointArray can be constructed from the result.
        '''
        a = np.random.randn(3) + 1j*np.random.randn(3)
        fp_a = self.fp_class(a, 10)
        frac_bits = fp_a.fractional_bits

        new_fp_a = self.fp_class(fp_a.data, 10, data_already_scaled=True)

        self.assertTrue(np.all(
            new_fp_a.as_floating_point() == fp_a.as_floating_point()))

        fail_fp_a = self.fp_class(fp_a.data, 10)
        self.assertFalse(np.any(
            fail_fp_a.as_floating_point() == fp_a.as_floating_point()))

    def test_construct_with_fixed_point_array(self):
        '''We should be able to construct one FixedPointArray from another.

        The fractional bits argument should still be required, and if it
        is different from the passed array, the new FixedPointArray has that
        which is provided by the argument.

        If possible, a copy should be avoided, though this can only be the
        case if the fractional bits passed is the same as the fractional bits
        of the existing fixed point array.

        In this case, the ``data_already_scaled argument`` should be ignored.
        '''

        start_frac_bits = 10
        a = np.random.randn(3) + 1j*np.random.randn(3)
        fp_a = self.fp_class(a, start_frac_bits)
        frac_bits = fp_a.fractional_bits

        for frac_bits in (8, 10, 12):
            new_fp_a = self.fp_class(fp_a, frac_bits)

            extra_frac_bits = frac_bits - start_frac_bits

            self.assertTrue(new_fp_a.fractional_bits == frac_bits)

            self.assertTrue(np.all(
                new_fp_a.data == f_round(fp_a.data * 2**extra_frac_bits)))

            out_value = (
                f_round(fp_a.as_floating_point() * 2**frac_bits) *
                2**-frac_bits)

            self.assertTrue(np.all(
                new_fp_a.as_floating_point() == out_value))

    def test_construct_from_fixed_point_array_should_round_to_truncate(self):
        '''Round should be used when constructing from a fixed point array

        When a fixed point array is passed in to construct a new fixed point
        array, if truncation is necessary (because the number of fractional
        bits has been reduced), then IEEE 754-1985 round should be used
        (that is, when the choice is ambiguous e.g. 1.5, round to the nearest
        even integer, so 2).
        '''

        frac_bits = 18

        # Start with an array filled with ones, which would normally round
        # up.
        array_a = ((2**25 - 1) * 2**-frac_bits *
                   (np.ones(10) + 1j*np.ones(10)))

        fp_a = self.fp_class(array_a, fractional_bits=frac_bits)
        truncated_a = self.fp_class(array_a, fractional_bits=frac_bits - 1)

        # Check it rounds up as expected
        self.assertEqual(fp_a.max_integer_bits + 1,
                         truncated_a.max_integer_bits)
        self.assertEqual(fp_a.max_bits, truncated_a.max_bits)

        # Now show the two cases around the round-up/down transition
        mid_point = (2**25 * 2**-frac_bits *
                     (np.ones(10) + 1j*np.ones(10))) - (1 + 1j) * 2**-1

        array_a = mid_point
        array_b = mid_point - (1 + 1j) * 2**-frac_bits # Smallest value below

        fp_a = self.fp_class(array_a, fractional_bits=frac_bits)
        fp_b = self.fp_class(array_b, fractional_bits=frac_bits)
        truncated_a = self.fp_class(array_a, fractional_bits=0)
        truncated_b = self.fp_class(array_b, fractional_bits=0)

        # a should round up
        self.assertEqual(fp_a.max_integer_bits + 1,
                         truncated_a.max_integer_bits)
        self.assertEqual(fp_a.max_bits - frac_bits + 1, truncated_a.max_bits)

        # b should round down
        self.assertEqual(fp_b.max_integer_bits,
                         truncated_b.max_integer_bits)
        self.assertEqual(fp_b.max_bits - frac_bits, truncated_b.max_bits)


    def test_data_is_not_copied(self):
        '''The internal data should be the same as passed in if possible.

        This should be true for data_already_scaled being both True and
        false for that flag being False.
        '''
        a = np.random.randn(3) + 1j*np.random.randn(3)
        fp_a = self.fp_class(a, 10)
        frac_bits = fp_a.fractional_bits

        new_fp_a = self.fp_class(fp_a.data, 10, data_already_scaled=True)

        self.assertIsNot(a, fp_a.data)
        self.assertIs(new_fp_a.data, fp_a.data)

    def test_data_already_scaled_integer_only(self):
        '''Data with fractional parts with data_already_scaled should fail.

        If an array is passed in that contains any fractional parts (i.e.
        is not only integers) and ``data_already_scaled`` is ``True``, a
        ValueError should be raised.
        '''
        a = np.random.randn(3) + 1j*np.random.randn(3)
        fp_a = self.fp_class(a, 10)
        frac_bits = fp_a.fractional_bits

        fp_a_data = fp_a.data.copy()
        fp_a_data += 0.1
        self.assertRaises(ValueError, self.fp_class,
                          fp_a_data, 10, data_already_scaled=True)

        fp_sparse_a = self.fp_class(sparse.csc_matrix(fp_a.data), 10,
                                     data_already_scaled=True)

        self.assertRaises(ValueError, self.fp_class,
                          sparse.csc_matrix(fp_a_data), 10,
                          data_already_scaled=True)


    def test_correct_precision(self):
        '''We should lose precision at the right point.
        '''

        for frac_bits in (-4, 0, 4, 15, 24):
            # make the shape _big_ so the chance is high of getting a number
            # in the necessary region
            shape = (10000,)

            if frac_bits < 0:
                scale = 2**-frac_bits
            else:
                scale = 1.0

            # Use complex numbers as they encompass the real
            test_array = (
                    np.random.randn(*shape) +
                    1j*np.random.randn(*shape)) * scale

            fp_array = self.fp_class(test_array, frac_bits)

            real_fp_array = fp_array.as_floating_point().real
            max_rerror = np.max(np.abs(real_fp_array - test_array.real))

            imag_fp_array = fp_array.as_floating_point().imag
            max_ierror = np.max(np.abs(imag_fp_array - test_array.imag))

            max_error = max(max_rerror, max_ierror)

            if max_error < 2**(-frac_bits-2):
                print ('This test is probabilistic, so no need to worry '
                       'about rare failures.')

            self.assertFalse(max_error < 2**(-frac_bits-2))

    def test_complex_array(self):
        '''We should store complex numbers with sufficient precision.
        '''

        for frac_bits in (-5, 4, 15, 24):
            for shape in ((4, 200), (1009,), (5, 65, 70)):
                test_array = (
                    np.random.randn(*shape) +
                    1j*np.random.randn(*shape))

                fp_array = self.fp_class(test_array, frac_bits)

                real_fp_array = fp_array.as_floating_point().real
                max_rerror = np.max(np.abs(real_fp_array - test_array.real))

                imag_fp_array = fp_array.as_floating_point().imag
                max_ierror = np.max(np.abs(imag_fp_array - test_array.imag))

                max_error = max(max_rerror, max_ierror)
                self.assertTrue(max_error < 2**(-frac_bits-1))

    def test_unary_neg(self):
        '''We should be able to negate the array with -array.
        '''
        shape = (4, 200)
        frac_bits = 10

        test_array = (
            np.random.randn(*shape) + 1j*np.random.randn(*shape))

        rounded_array = f_round(test_array * 2**frac_bits) * 2**-frac_bits

        fp_array = self.fp_class(test_array, frac_bits)

        self.assertTrue(np.alltrue(-fp_array == -rounded_array))


    def test_mul(self):
        '''We should be able to multiply two fixed point arrays.

        The number of fractional bits of the input should be the same as the
        larger of the fractional bits of the two inputs.

        Broadcasting should work as with numpy.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))
        a_frac_bits = (-4, 0, 12, 2, 32)
        b_frac_bits = (-5, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits
            fp_b_frac_bits = each_fp_b.fractional_bits

            output_frac_bits = max(fp_a_frac_bits, fp_b_frac_bits)

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

            ref_result = (f_round(_a*_b * 2**output_frac_bits) *
                           2**-output_frac_bits)

            result = each_fp_a * each_fp_b

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

            self.assertTrue(result.fractional_bits == output_frac_bits)


    def test_sum(self):
        '''We should be able to add two fixed point arrays.

        The number of fractional bits of the input should be the same as the
        larger of the fractional bits of the two inputs, thereby not losing
        precision.

        Broadcasting should work as with numpy.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))
        a_frac_bits = (-4, 0, 12, 2, 32)
        b_frac_bits = (-8, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits
            fp_b_frac_bits = each_fp_b.fractional_bits

            output_frac_bits = max(fp_a_frac_bits, fp_b_frac_bits)

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

            _a = f_round(_a * 2**output_frac_bits) * 2**-output_frac_bits
            _b = f_round(_b * 2**output_frac_bits) * 2**-output_frac_bits

            ref_result = _a + _b

            result = each_fp_a + each_fp_b

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

            self.assertTrue(result.fractional_bits == output_frac_bits)

    def test_diff(self):
        '''We should be able to take the difference of two fixed point arrays.

        The number of fractional bits of the input should be the same as the
        larger of the fractional bits of the two inputs, thereby not losing
        precision

        Broadcasting should work as with numpy.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))
        a_frac_bits = (-5, 0, 12, 2, 32)
        b_frac_bits = (-6, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits
            fp_b_frac_bits = each_fp_b.fractional_bits

            output_frac_bits = max(fp_a_frac_bits, fp_b_frac_bits)

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

            _a = f_round(_a * 2**output_frac_bits) * 2**-output_frac_bits
            _b = f_round(_b * 2**output_frac_bits) * 2**-output_frac_bits

            ref_result = _a - _b

            result = each_fp_a - each_fp_b

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

            self.assertTrue(result.fractional_bits == output_frac_bits)

    def test_non_fixed_point_diff(self):
        '''We should be able to subtract a non-fixed point from a fixed point

        We should be able to subtract from the fixed point object a
        non-fixed point object, as long as it makes sense to
        convert it to a fixed point object - this should include scalars
        and normal numpy arrays. The conversion to the fixed point array
        should use the same number of fractional bits as the original array.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))
        a_frac_bits = (-5, 0, 12, 2, 32)
        b_frac_bits = (-3, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits

            output_frac_bits = fp_a_frac_bits

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits

            _a = f_round(_a * 2**output_frac_bits) * 2**-output_frac_bits
            _b = f_round(_b * 2**output_frac_bits) * 2**-output_frac_bits

            ref_scalar = _b.ravel()[0]
            scalar = each_b.ravel()[0]

            lsub_ref_result = _a - _b
            lsub_scalar_ref_result = _a - ref_scalar

            lsub_result = each_fp_a - each_b
            lsub_scalar_result = each_fp_a - scalar

            self.assertTrue(
                np.alltrue(lsub_result.as_floating_point() ==
                           lsub_ref_result))

            self.assertTrue(
                np.alltrue(lsub_scalar_result.as_floating_point() ==
                           lsub_scalar_ref_result))

    def test_non_fixed_point_right_diff(self):
        '''We should be able to subtract a fixed point from a non-fixed point

        We should be able to subtract from a non-fixed point object a
        fixed point object, as long as it makes sense to
        convert it to a fixed point object - this should include scalars
        and normal numpy arrays. The conversion to the fixed point array
        should use the same number of fractional bits as the original array.

        Broadcasting should work as with numpy.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))
        a_frac_bits = (-4, 0, 12, 2, 32)
        b_frac_bits = (-5, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits

            output_frac_bits = fp_a_frac_bits

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits

            _a = f_round(_a * 2**output_frac_bits) * 2**-output_frac_bits
            _b = f_round(_b * 2**output_frac_bits) * 2**-output_frac_bits

            ref_scalar = _b.ravel()[0]
            scalar = each_b.ravel()[0]

            rsub_ref_result = _b - _a
            rsub_scalar_ref_result = ref_scalar - _a

            rsub_result = each_b - each_fp_a
            rsub_scalar_result = scalar - each_fp_a

            self.assertTrue(
                np.alltrue(rsub_result.as_floating_point() ==
                           rsub_ref_result))

            self.assertTrue(
                np.alltrue(rsub_scalar_result.as_floating_point() ==
                           rsub_scalar_ref_result))

    def test_divide(self):
        '''We should be able to divide two fixed point arrays.

        The number of fractional bits in the result should be the same as the
        in the numerator.

        Broadcasting should work as with numpy.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))
        a_frac_bits = (-4, 16, 12, 2, 32, 32)
        b_frac_bits = (-3, 16, 13, 32, 3, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            # We need to make sure we're not going to divide by zero
            if np.min(np.abs(each_fp_b.as_floating_point())) == 0:
                each_fp_b[each_b > 0] += 2**(-each_fp_b.fractional_bits)
                each_fp_b[each_b < 0] -= 2**(-each_fp_b.fractional_bits)

                each_b[each_b > 0] += 2**(-each_fp_b.fractional_bits)
                each_b[each_b < 0] -= 2**(-each_fp_b.fractional_bits)

            fp_a_frac_bits = each_fp_a.fractional_bits
            fp_b_frac_bits = each_fp_b.fractional_bits

            output_frac_bits = fp_a_frac_bits

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

            ref_result = (f_round((_a/_b) * 2**output_frac_bits) *
                           2**-output_frac_bits)

            result = each_fp_a/each_fp_b

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

    def test_scalar_denominator_divide(self):
        '''We should be able to divide a fixed point array by a python scalar

        The scalar should firstly be converted to a fixed point value with
        the same number of fractional bits as the fixed point array and a
        fixed point divide performed.
        '''

        scalars = [float(each) for each in np.random.randn(3)] + [1, 100, -20]

        a = np.random.randn(20) + 1j*np.random.randn(20)
        fp_a = self.fp_class(a, 10)
        frac_bits = fp_a.fractional_bits

        _a = f_round(a * 2**frac_bits) * 2**-frac_bits

        for scalar in scalars:

            fp_scalar = round(scalar * 2**frac_bits) * 2**-frac_bits

            ref_result = (f_round((_a/fp_scalar) * 2**frac_bits) *
                          2**-frac_bits)

            result = fp_a/scalar

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

    def test_scalar_numerator_divide(self):
        '''We should be able to divide a python scalar by a fixed point array

        The scalar should firstly be converted to a fixed point value with
        the same number of fractional bits as the fixed point array and a
        fixed point divide performed.
        '''

        scalars = list(np.random.randn(3)) + [1, 100, -20]

        a = np.random.randn(3) + 1j*np.random.randn(3)
        fp_a = self.fp_class(a, 10)
        frac_bits = fp_a.fractional_bits

        _a = f_round(a * 2**frac_bits) * 2**-frac_bits

        for scalar in scalars:

            fp_scalar = f_round(scalar * 2**frac_bits) * 2**-frac_bits

            ref_result = (f_round((fp_scalar/_a) * 2**frac_bits) *
                          2**-frac_bits)

            result = scalar/fp_a
            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

    def test_equality_check(self):
        '''We should be able to test equality with a fixed point array

        We should be able to do ``a == b`` and yield a numpy array of boolean
        values where the elements of ``a`` equal the elements of ``b`` and
        false where they do not.

        It should work regardless of the number of fractional bits of ``a``
        and ``b``.

        If the arrays are of different size, then the same behaviour as numpy
        should be used (in 1.19 a warning is raised).
        '''
        a = np.random.randn(10) + 1j*np.random.randn(10)
        b = a.copy()

        fp_a = self.fp_class(a, fractional_bits=10)
        fp_b = self.fp_class(b, fractional_bits=10)

        self.assertTrue(np.alltrue(fp_a == fp_b))

        fp_c = self.fp_class(fp_b.as_floating_point(),
                               fractional_bits = 12)

        self.assertTrue(np.alltrue(fp_a == fp_c))

        d = a.copy()
        d[5:] = -d[5:]
        fp_d = self.fp_class(d, fractional_bits=10)

        self.assertTrue(np.alltrue((fp_d == fp_a)[:5]))
        self.assertTrue(np.alltrue(np.logical_not(fp_d == fp_a)[5:]))

        # this test is removed because the behaviour is deprecated in numpy
        #self.assertFalse(fp_a == fp_a[:5])

    def test_equality_with_scalar(self):
        '''We should be able to test equality with a scalar value

        Given a scalar value, we should be able to check which values of
        the fixed point data are equal to the scalar. The scalar should be
        coerced to be a fixed point of the same precision as the array
        and then a boolean numpy array should be returned where values
        are equal.
        '''
        scalar = complex(np.random.rand(1), np.random.rand(1))
        a = np.ones(10) * scalar

        frac_bits = 4

        fp_a = self.fp_class(a, fractional_bits=frac_bits)

        scaled_scalar = scalar * 2**frac_bits

        self.assertTrue(np.alltrue(fp_a == scalar))

        b = a.copy()
        b[5:] = -b[5:]
        fp_b = self.fp_class(b, fractional_bits=frac_bits)

        self.assertTrue(np.alltrue((fp_b == scalar)[:5]))
        self.assertTrue(np.alltrue(np.logical_not(fp_b == scalar)[5:]))

    def test_equality_with_np_array(self):
        '''We should be able to test equality with a numpy array

        Given a numpy array, we should be able to check which values of
        the fixed point data are equal to the corresponding values of the
        numpy array. The numpy array should be coerced to be a fixed point of
        the same precision as the array and then a boolean numpy array should
        be returned where the values are equal.

        If the arrays are of different size, then the same behaviour as numpy
        should be used (in 1.19 a warning is raised).
        '''
        a = np.random.randn(10) + 1j*np.random.randn(10)
        b = a.copy()

        fp_a = self.fp_class(a, fractional_bits=10)

        self.assertTrue(np.alltrue(fp_a == b))

        d = a.copy()
        d[5:] = -d[5:]

        self.assertTrue(np.alltrue((d == fp_a)[:5]))
        self.assertTrue(np.alltrue(np.logical_not(d == fp_a)[5:]))

        # this test is removed because the behaviour is deprecated in numpy
        #self.assertFalse(d[5:] == fp_a)

    def test_not_equal_with_scalar(self):
        '''We should be able to check we are _not_ equal to a scalar

        As with the check to see if we are equal, we should be able to check
        that we are not equal.
        '''
        scalar = complex(np.random.rand(1), np.random.rand(1))
        a = np.ones(10) * scalar

        frac_bits = 4

        fp_a = self.fp_class(a, fractional_bits=frac_bits)

        scaled_scalar = scalar * 2**frac_bits

        self.assertTrue(np.alltrue(np.logical_not(fp_a != scalar)))

        b = a.copy()
        b[5:] = -b[5:]
        fp_b = self.fp_class(b, fractional_bits=frac_bits)

        self.assertTrue(np.alltrue((fp_b != scalar)[5:]))
        self.assertTrue(np.alltrue(np.logical_not(fp_b != scalar)[:5]))

    def test_not_equal_with_np_array(self):
        '''We should be able to test not equals with a numpy array

        Given a numpy array, we should be able to check which values of
        the fixed point data are not equal to the corresponding values of the
        numpy array. The numpy array should be coerced to be a fixed point of
        the same precision as the array and then a boolean numpy array should
        be returned where the values are equal.

        If the arrays are of different size, then the same behaviour as numpy
        is used (in 1.19 a warning is raised).
        '''
        a = np.random.randn(10) + 1j*np.random.randn(10)
        b = a.copy()

        fp_a = self.fp_class(a, fractional_bits=10)

        self.assertTrue(np.alltrue(np.logical_not(fp_a != b)))

        d = a.copy()
        d[5:] = -d[5:]

        self.assertTrue(np.alltrue((d != fp_a)[5:]))
        self.assertTrue(np.alltrue(np.logical_not(d != fp_a)[:5]))

        # this test is removed because the behaviour is deprecated in numpy
        #self.assertTrue(d[5:] != fp_a)

    def test_notequal_check(self):
        '''We should be able to test fixed point arrays are not equal

        We should be able to do ``a != b`` and yield False if ``a`` and ``b``
        contain the same data and True if they contain different data.

        If the arrays are of different size, then the same behaviour as numpy
        should be used (in 1.19 a warning is raised).
        '''
        a = np.random.randn(10) + 1j*np.random.randn(10)
        b = a.copy()

        fp_a = self.fp_class(a, fractional_bits=10)
        fp_b = self.fp_class(b, fractional_bits=10)

        self.assertTrue(np.alltrue(np.logical_not(fp_a != fp_b)))

        d = a.copy()
        d[5:] = -d[5:]
        fp_d = self.fp_class(d, fractional_bits=10)

        self.assertTrue(np.alltrue((fp_d != fp_a)[5:]))
        self.assertTrue(np.alltrue(np.logical_not(fp_d != fp_a)[:5]))

        # this test is removed because the behaviour is deprecated in numpy
        #self.assertTrue(fp_a != fp_a[:5])

    def test_dot_product(self):
        '''There should be a ``dot`` method.
        The ``dot`` method  should take the dot product of two fixed point
        arrays.

        The number of fractional bits of the output should be the maximum of
        the number of fractional bits of the two inputs.
        '''

        a_shapes = ((3, 4, 5), (120,), (59, 34))
        b_shapes = ((10, 5, 3), (120,), (34, 21))
        a_frac_bits = (-3, 0, 12, 2, 32)
        b_frac_bits = (-5, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits
            fp_b_frac_bits = each_fp_b.fractional_bits

            output_frac_bits = max(fp_a_frac_bits, fp_b_frac_bits)

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

            ref_result = (f_round(_a.dot(_b) * 2**output_frac_bits) *
                           2**-output_frac_bits)

            result = each_fp_a.dot(each_fp_b)

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

    def test_non_fixed_point_array_dot_product(self):
        '''The``dot`` method should handle non fixed point arrays.
        '''

        a_shapes = ((3, 4, 5), (120,), (59, 34))
        b_shapes = ((10, 5, 3), (120,), (34, 21))
        a_frac_bits = (-5, 0, 12, 2, 32)
        b_frac_bits = (-6, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits

            output_frac_bits = fp_a_frac_bits

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits

            ref_result = (f_round(_a.dot(_b) * 2**output_frac_bits) *
                           2**-output_frac_bits)

            result = each_fp_a.dot(each_b)

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

    def test_non_truncated_dot_product(self):
        '''There should be a ``non_truncated_dot`` method.
        The ``non_truncated_dot`` method  should take the dot product of two
        fixed point arrays, but without imposing any truncation on the
        fractional part of the result. That is, the number of output fractional
        bits should be the sum of the number of fractional bits of the two
        inputs.
        '''

        a_shapes = ((3, 4, 5), (120,), (59, 34))
        b_shapes = ((10, 5, 3), (120,), (34, 21))
        a_frac_bits = (-5, 0, 12, 2, 32)
        b_frac_bits = (-7, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits
            fp_b_frac_bits = each_fp_b.fractional_bits

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

            ref_result = _a.dot(_b)

            result = each_fp_a.non_truncated_dot(each_fp_b)

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))

    def test_non_truncated_dot_non_fp_fail(self):
        '''The non_truncated_dot method should only accept a FixedPointArray

        Passing something other than a FixedPointArray to the
        non_truncated_dot method should raise a ValueError exception.
        '''
        a = np.random.randn(20) + 1j*np.random.randn(20)
        fp_a = self.fp_class(a, 10)

        try:
            assert_raises_regex = self.assertRaisesRegex
        except AttributeError:
            assert_raises_regex = self.assertRaisesRegexp

        assert_raises_regex(ValueError, 'Invalid array',
                            fp_a.non_truncated_dot, a)

    def test_non_truncated_multiply(self):
        '''There should be a ``non_truncated_multiply`` method.
        The ``non_truncated_multiply`` method  should multiply two
        fixed point arrays, but without imposing any truncation on the
        fractional part of the result. That is, the number of output fractional
        bits should be the sum of the number of fractional bits of the two
        inputs.

        Broadcasting should work as with numpy.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))

        a_frac_bits = (-4, 0, 12, 2, 32)
        b_frac_bits = (-5, 0, 13, 32, 32)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for each_a, each_fp_a, each_b, each_fp_b in zip(*arrays):

            fp_a_frac_bits = each_fp_a.fractional_bits
            fp_b_frac_bits = each_fp_b.fractional_bits

            output_frac_bits = max(fp_a_frac_bits, fp_b_frac_bits)

            _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
            _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

            ref_result = _a * _b

            result = each_fp_a.non_truncated_multiply(each_fp_b)

            self.assertTrue(
                np.alltrue(result.as_floating_point() == ref_result))


    def test_non_truncated_multiply_non_fp_fail(self):
        '''non_truncated_multiply should only accept a FixedPointArray arg.

        Passing something other than a FixedPointArray to the
        non_truncated_multiply method should raise a ValueError exception.
        '''
        a = np.random.randn(20) + 1j*np.random.randn(20)
        fp_a = self.fp_class(a, 10)

        try:
            assert_raises_regex = self.assertRaisesRegex
        except AttributeError:
            assert_raises_regex = self.assertRaisesRegexp

        assert_raises_regex(ValueError, 'Invalid array',
                            fp_a.non_truncated_multiply, a)

    def test_divide_with_precision(self):
        '''There should be a ``divide_with_precision`` method.
        The ``divide_with_precision`` method  should divide two
        fixed point arrays, but with an additional argument of
        `fractional_bits` that dictates how many fractional bits to use for
        the output array.

        Broadcasting should work as with numpy.
        '''
        a_shapes = ((3, 4, 5), (120,), (59, 34), (3, 4, 5), (59, 34), (1,))
        b_shapes = ((3, 4, 5), (120,), (59, 34), (4, 5), (1,), (33, 15))

        a_frac_bits = (-4, 0, 12, 2, 32)
        b_frac_bits = (-5, 0, 13, 32, 32)

        output_frac_bits_set = (-4, -1, 0, 3, 10)

        arrays = self.create_test_array_pairs(
            a_shapes, b_shapes, a_frac_bits, b_frac_bits)

        for _each_a, each_fp_a, _each_b, each_fp_b in zip(*arrays):

            for output_frac_bits in output_frac_bits_set:

                each_a = _each_a.copy()
                each_b = _each_b.copy()

                fp_a_frac_bits = each_fp_a.fractional_bits
                fp_b_frac_bits = each_fp_b.fractional_bits

                if fp_a_frac_bits < 1:
                    # put the values into a range that is more useful
                    each_a[each_a > 0] = ((each_a[each_a > 0] + 1) *
                                          2**(2*-fp_a_frac_bits))
                    each_a[each_a < 0] = ((each_a[each_a < 0] - 1) *
                                          2**(2*-fp_a_frac_bits))

                if fp_b_frac_bits < 1:
                    each_b[each_b > 0] = ((each_b[each_b > 0] + 1) *
                                          2**(2*-fp_b_frac_bits))
                    each_b[each_b < 0] = ((each_b[each_b < 0] - 1) *
                                          2**(2*-fp_b_frac_bits))


                #output_frac_bits = max(fp_a_frac_bits, fp_b_frac_bits)

                _a = f_round(each_a * 2**fp_a_frac_bits) * 2**-fp_a_frac_bits
                _b = f_round(each_b * 2**fp_b_frac_bits) * 2**-fp_b_frac_bits

                fp_a = self.fp_class(each_a, fractional_bits=fp_a_frac_bits)
                fp_b = self.fp_class(each_b, fractional_bits=fp_b_frac_bits)

                # Make sure we never have zero on the denominator
                _b[_b == 0] += 1
                fp_b[fp_b == 0] += 1

                ref_result = (f_round((_a / _b) * 2**output_frac_bits) *
                              2**-output_frac_bits)

                result = fp_a.divide_with_precision(fp_b,
                                                    output_frac_bits)

                self.assertTrue(
                    np.alltrue(result.as_floating_point() == ref_result))


    def test_divide_with_precision_non_fp_fail(self):
        '''divide_with_precision should only accept a FixedPointArray arg.

        Passing something other than a FixedPointArray to the
        divide_with_precision method should raise a ValueError exception.
        '''
        a = np.random.randn(20) + 1j*np.random.randn(20)
        fp_a = self.fp_class(a, 10)

        try:
            assert_raises_regex = self.assertRaisesRegex
        except AttributeError:
            assert_raises_regex = self.assertRaisesRegexp

        assert_raises_regex(ValueError, 'Invalid array',
                            fp_a.divide_with_precision, a, 0)

    def test_slicing(self):
        '''We should be able to get values with a slice.
        The should get back another ``FixedPointArray`` object.
        '''
        a = np.random.randn(20) + 1j*np.random.randn(20)
        fp_a = self.fp_class(a, 10)

        fp_a_slice = fp_a[:5]

        self.assertTrue(fp_a_slice.fractional_bits == fp_a.fractional_bits)

        fractional_bits = fp_a.fractional_bits

        a_slice = f_round(a[:5] * 2**fractional_bits) * 2**-fractional_bits

        self.assertTrue(isinstance(fp_a_slice, self.fp_class))
        self.assertTrue(np.alltrue(a_slice == fp_a_slice.as_floating_point()))

    def test_slice_setting(self):
        '''We should be able to set values with a slice.
        '''
        a = np.random.randn(20) + 1j*np.random.randn(20)
        b = np.random.randn(20) + 1j*np.random.randn(20)

        fp_a = self.fp_class(a, 10)
        fp_b = self.fp_class(b, 8)

        fp_b[:5] = fp_a[:5]
        b[:5] = (f_round(a[:5] * 2**fp_a.fractional_bits) *
                 2**-fp_a.fractional_bits)

        fractional_bits = fp_b.fractional_bits

        _b = f_round(b * 2**fractional_bits) * 2**-fractional_bits
        self.assertTrue(np.alltrue(_b == fp_b.as_floating_point()))

    def _get_sparse_arrays(self):

        if isinstance(sparse, MockSparse):
            raise unittest.SkipTest(
                'Scipy is missing, so sparse tests are ignored.')

        a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)
        b = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        sparse_a = sparse.csc_matrix(a)
        sparse_b = sparse.csc_matrix(b)

        fp_a = self.fp_class(a, 10)
        fp_b = self.fp_class(b, 8)

        fp_sparse_a = self.fp_class(sparse_a, 10)
        fp_sparse_b = self.fp_class(sparse_b, 8)

        return fp_a, fp_b, fp_sparse_a, fp_sparse_b

    def test_sparse_basic(self):
        '''We should be able to create a basic _sparse_ fixed point array
        The underlying data should be sparse.
        '''
        fp_a, fp_b, fp_sparse_a, fp_sparse_b = self._get_sparse_arrays()

        # basic
        self.assertTrue(np.alltrue(fp_a.as_floating_point() ==
                                   fp_sparse_a.as_floating_point()))

        self.assertTrue(sparse.issparse(fp_sparse_a.data))

    def test_sparse_multiplication(self):
        '''We should be able to multiply sparse arrays.
        The behaviour of multiplying sparse arrays should be the same as with
        non sparse arrays - i.e. not a dot product.
        '''
        fp_a, fp_b, fp_sparse_a, fp_sparse_b = self._get_sparse_arrays()

        # multiplication
        self.assertTrue(np.alltrue(
            (fp_a * fp_b).as_floating_point() ==
            (fp_sparse_a * fp_sparse_b).as_floating_point()))

    def test_sparse_add(self):
        '''We should be able to add sparse arrays properly.
        '''

        fp_a, fp_b, fp_sparse_a, fp_sparse_b = self._get_sparse_arrays()

        # addition
        self.assertTrue(np.alltrue(
            (fp_a + fp_b).as_floating_point() ==
            (fp_sparse_a + fp_sparse_b).as_floating_point()))

    def test_sparse_diff(self):
        '''We should be able to subtract sparse arrays properly.
        '''
        fp_a, fp_b, fp_sparse_a, fp_sparse_b = self._get_sparse_arrays()

        # subtraction
        self.assertTrue(np.alltrue(
            (fp_a - fp_b).as_floating_point() ==
            (fp_sparse_a - fp_sparse_b).as_floating_point()))


    def test_sparse_dot_product(self):
        '''We should be able to use dot product with sparse arrays properly
        '''
        fp_a, fp_b, fp_sparse_a, fp_sparse_b = self._get_sparse_arrays()

        # .dot()
        self.assertTrue(np.alltrue(
            (fp_a - fp_b).as_floating_point() ==
            (fp_sparse_a - fp_sparse_b).as_floating_point()))

    def test_dtype_property(self):
        '''The dtype property should return the dtype of the underlying data
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)
        real_a = cmplx_a.real

        fp_cmplx_a = self.fp_class(cmplx_a, 10)
        fp_real_a = self.fp_class(real_a, 10)

        sparse_fp_cmplx_a = self.fp_class(sparse.csc_matrix(cmplx_a), 10)
        sparse_fp_real_a = self.fp_class(sparse.csc_matrix(real_a), 10)

        self.assertTrue(fp_cmplx_a.dtype == 'complex128')
        self.assertTrue(fp_real_a.dtype == 'float64')
        self.assertTrue(sparse_fp_cmplx_a.dtype == 'complex128')
        self.assertTrue(sparse_fp_real_a.dtype == 'float64')

    def test_shape_property(self):
        '''The shape property should return the shape of the array.
        '''
        shape = (3, 4)
        cmplx_a = np.random.randn(*shape) + 1j*np.random.randn(*shape)

        fp_cmplx_a = self.fp_class(cmplx_a, 10)

        sparse_fp_cmplx_a = self.fp_class(sparse.csc_matrix(cmplx_a), 10)

        self.assertTrue(fp_cmplx_a.shape == shape)
        self.assertTrue(sparse_fp_cmplx_a.shape == shape)

    def test_imag_property(self):
        '''The imag property should return the imaginary part of the array.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        fp_cmplx_a = self.fp_class(cmplx_a, 10)
        sparse_fp_cmplx_a = self.fp_class(sparse.csc_matrix(cmplx_a), 10)

        fp_imag_a = fp_cmplx_a.imag
        sparse_fp_imag_a = sparse_fp_cmplx_a.imag

        self.assertTrue(isinstance(fp_imag_a, self.fp_class))
        self.assertTrue(np.alltrue(
            fp_imag_a.as_floating_point() ==
            fp_cmplx_a.as_floating_point().imag))

        self.assertTrue(isinstance(sparse_fp_imag_a, self.fp_class))
        self.assertTrue(np.alltrue(
            sparse_fp_imag_a.as_floating_point() ==
            fp_cmplx_a.as_floating_point().imag))

    def test_real_property(self):
        '''The real property should return the real part of the array.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        fp_cmplx_a = self.fp_class(cmplx_a, 10)
        sparse_fp_cmplx_a = self.fp_class(sparse.csc_matrix(cmplx_a), 10)

        fp_real_a = fp_cmplx_a.real
        sparse_fp_real_a = sparse_fp_cmplx_a.real

        self.assertTrue(isinstance(fp_real_a, self.fp_class))
        self.assertTrue(np.alltrue(
            fp_real_a.as_floating_point() ==
            fp_cmplx_a.as_floating_point().real))

        self.assertTrue(isinstance(sparse_fp_real_a, self.fp_class))
        self.assertTrue(np.alltrue(
            sparse_fp_real_a.as_floating_point() ==
            fp_cmplx_a.as_floating_point().real))

    def test_as_type(self):
        '''There should be an ``astype`` method.
        The ``astype`` method should simply return itself.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        fp_cmplx_a = self.fp_class(cmplx_a, 10)

        self.assertIs(fp_cmplx_a, fp_cmplx_a.astype('complex64'))

    def test_sum_abs(self):
        '''There should be a ``sum_abs`` method.
        The ``sum_abs`` method should should return the sum of the
        absolute values of the array.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        a = f_round(cmplx_a * 2**frac_bits) * 2**-frac_bits

        test_out = np.sum(np.abs(a))

        self.assertTrue(test_out == fp_cmplx_a.sum_abs())
        self.assertTrue(np.allclose(test_out, sparse_fp_cmplx_a.sum_abs()))

    def test_max_abs(self):
        '''There should be a ``max_abs`` method.
        the ``max_abs`` method should return the maximum value of the
        absolute values of the array.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        a = f_round(cmplx_a * 2**frac_bits) * 2**-frac_bits

        test_out = np.max(np.abs(a))

        self.assertTrue(test_out == fp_cmplx_a.max_abs())
        self.assertTrue(np.allclose(test_out, sparse_fp_cmplx_a.max_abs()))

    def test_max(self):
        '''There should be a ``max`` method.
        The ``max`` method should return the maximum value in the array (as
        defined by numpy when it comes to complex values).
        '''

        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        a = f_round(cmplx_a * 2**frac_bits) * 2**-frac_bits

        test_out = np.max(a)

        self.assertTrue(test_out == fp_cmplx_a.max())
        self.assertTrue(np.allclose(test_out, sparse_fp_cmplx_a.max()))

    def test_ravel(self):
        '''There should be a ``ravel`` method like numpy's equivalant.
        The ``ravel`` method should return a flattened FixedPointArray, only
        copying when necessary, like numpy's equivalent.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        self.assertTrue(isinstance(fp_cmplx_a.ravel(), self.fp_class))

        self.assertTrue(len(fp_cmplx_a.ravel().shape) == 1)
        self.assertTrue(np.prod(fp_cmplx_a.shape) == len(fp_cmplx_a.ravel()))
        self.assertTrue(fp_cmplx_a.data.ctypes.data ==
                        fp_cmplx_a.ravel().data.ctypes.data)

        self.assertTrue(len(sparse_fp_cmplx_a.ravel().shape) == 1)
        self.assertTrue(np.prod(sparse_fp_cmplx_a.shape) ==
                        len(sparse_fp_cmplx_a.ravel()))

    def test_flatten(self):
        '''There should be a ``flatten`` method.
        The ``flatten`` method should return a flattened copy of the fixed
        point array.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        self.assertTrue(len(fp_cmplx_a.flatten().shape) == 1)
        self.assertTrue(np.prod(fp_cmplx_a.shape) ==
                        len(fp_cmplx_a.flatten()))
        self.assertFalse(fp_cmplx_a.data.ctypes.data ==
                        fp_cmplx_a.flatten().data.ctypes.data)

        self.assertTrue(len(sparse_fp_cmplx_a.flatten().shape) == 1)
        self.assertTrue(np.prod(sparse_fp_cmplx_a.shape) ==
                        len(sparse_fp_cmplx_a.flatten()))

    def test_copy(self):
        '''There should be a copy method that returns a copy of the object

        The copy should have the same values in the data, but the underlying
        array should not be the same as the source.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        fp_cmplx_a_copy = fp_cmplx_a.copy()

        self.assertTrue(np.alltrue(fp_cmplx_a == fp_cmplx_a_copy))

        self.assertIsNot(fp_cmplx_a, fp_cmplx_a_copy)
        self.assertIsNot(fp_cmplx_a.data, fp_cmplx_a_copy.data)


    def test_len(self):
        '''len(obj) should return the length of the array.
        '''
        frac_bits = 10

        for shape in ((4, 200), (1009,), (5, 65, 70)):
            test_array = np.random.randn(*shape)

            fp_array = self.fp_class(test_array, frac_bits)

            self.assertTrue(len(fp_array) == len(test_array))


    def test_reshape(self):
        '''There should be a reshape method.
        The reshape method should return a reshaped fixed point array.
        '''
        cmplx_a = np.random.randn(20, 20) + 1j*np.random.randn(20, 20)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        a = f_round(cmplx_a * 2**frac_bits) * 2**-frac_bits

        self.assertTrue(np.all(
            fp_cmplx_a.reshape((10, 40)).as_floating_point() ==
            a.reshape((10, 40))))

        self.assertTrue(np.all(
            sparse_fp_cmplx_a.reshape((10, 40)).as_floating_point() ==
            a.reshape((10, 40))))

    def test_transpose(self):
        '''There should be a ``transpose`` method.
        The ``transpose`` method return the transpose of the dataset, as
        described by :meth:`numpy.transpose`
        '''
        cmplx_a = np.random.randn(20, 10) + 1j*np.random.randn(20, 10)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        a = f_round(cmplx_a * 2**frac_bits) * 2**-frac_bits

        self.assertTrue(np.all(
            fp_cmplx_a.transpose().as_floating_point() ==
            a.transpose()))

        self.assertTrue(np.all(
            sparse_fp_cmplx_a.transpose().as_floating_point() ==
            a.transpose()))

    def test_conjugate(self):
        '''There should be a ``conjugate`` method.
        The ``conjugate`` method should return the complex conjugate of all
        the elements in the array as a new fixed point array.
        '''
        cmplx_a = np.random.randn(20, 10) + 1j*np.random.randn(20, 10)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        a = f_round(cmplx_a * 2**frac_bits) * 2**-frac_bits

        self.assertTrue(np.all(
            fp_cmplx_a.conjugate().as_floating_point() ==
            a.conjugate()))

        self.assertTrue(np.all(
            sparse_fp_cmplx_a.conjugate().as_floating_point() ==
            a.conjugate()))

    def test_conj(self):
        '''There should be a ``conj`` method.
        The ``conj`` method should be equivalent to the ``conjugate`` method.
        '''
        cmplx_a = np.random.randn(20, 10) + 1j*np.random.randn(20, 10)

        frac_bits = 10
        fp_cmplx_a = self.fp_class(cmplx_a, frac_bits)

        sparse_fp_cmplx_a = self.fp_class(
            sparse.csc_matrix(cmplx_a), frac_bits)

        a = f_round(cmplx_a * 2**frac_bits) * 2**-frac_bits

        self.assertTrue(np.all(
            fp_cmplx_a.conj().as_floating_point() ==
            a.conjugate()))

        self.assertTrue(np.all(
            sparse_fp_cmplx_a.conj().as_floating_point() ==
            a.conjugate()))

    def test_max_integer_bits(self):
        '''There should be a ``max_integer_bits`` property.
        The ``max_integer_bits`` property should be the maximum number of
        integer bits needed in order to represent the integer part of every
        value in the array, ignoring the number of fractional bits (even
        if the number of fractional bits is negative).
        '''

        # test set is (min val, max val, fractional bits, max integer bits)
        test_set = (
            (-5, 5, 1, 3),
            (-4, 5, 1, 3),
            (-4, 4, 1, 3),
            (-4, 3, 1, 2), # -4 needs 2 bits + sign bit
            (-5, 3, 1, 3),
            (-0.6, 0, 0, 0), # The wierd case in which -1 needs no integer bits
            (-0.6, 0.6, 0, 1), # But the positive integer does (0.6 rounded)
            (0, 0, 0, 0),
            (0, 2, 0, 2),
            (1.8, 1.8, 0, 2),
            (-1.8, 1.8, 0, 2),
            (-3, 3, 1, 2),
            (-5, 5, -1, 3),
            (-5, 5, -2, 3),
            (-5, 5, -3, 4), # rounds to -8 and 8 (round to even)
            (-4, 0, -3, 0), # rounds to zero
            (0, 4, -3, 0), # rounds to zero
        )

        for min_val, max_val, frac_bits, expected_integer_bits in test_set:
            a = self.fp_class([min_val, max_val], fractional_bits=frac_bits)
            self.assertEqual(expected_integer_bits, a.max_integer_bits)

            a = self.fp_class(
                [min_val*1j, max_val*1j], fractional_bits=frac_bits)
            self.assertEqual(expected_integer_bits, a.max_integer_bits)

            a = self.fp_class(
                [min_val*(1+1j), max_val*(1+1j)], fractional_bits=frac_bits)
            self.assertEqual(expected_integer_bits, a.max_integer_bits)

    def test_max_bits(self):
        '''There should be a ``max_bits`` property.
        The ``max_bits`` property should be the number of bits required to
        store the largest value in the array up to the predefined fractional
        precision. It should be the sum of the fractional bits and the
        maximum integer bits.

        If a sign bit is needed, that should also be included in the bit
        count.

        A zero array should have the number of fractional bits defined by
        the fractional bits. This is because zero is always to a certain
        precision.
        '''

        # test set is (min val, max val, fractional bits, max bits)
        test_set = (
            (-5, 5, 1, 5),
            (-4, 5, 1, 5),
            (-4, 4, 1, 5),
            (-4, 3, 1, 4), # -4 needs 2 bits + sign bit
            (-5, 3, 1, 5),
            (-0.6, 0, 0, 1), # The wierd case in which -1 needs no integer bits
            (-0.6, 0.6, 0, 2), # But the positive integer does (0.6 rounded)
            (0, 0, 0, 0),
            (0, 2, 0, 2),
            (1.8, 1.8, 0, 2),
            (-1.8, 1.8, 0, 3),
            (-3, 3, 1, 4),
            (-5, 5, -1, 3),
            (-5, 5, -2, 2),
            (-5, 5, -3, 2), # rounds to -8 and 8 (round to even)
            (-5, 0, -3, 1), # -8 needs only 1 bit!
            (-4, 0, -3, 0), # rounds to zero
            (0, 4, -3, 0), # rounds to zero
        )

        for min_val, max_val, frac_bits, expected_max_bits in test_set:
            a = self.fp_class([min_val, max_val], fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, a.max_bits)

            a = self.fp_class(
                [min_val*1j, max_val*1j], fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, a.max_bits)

            a = self.fp_class(
                [min_val*(1+1j), max_val*(1+1j)], fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, a.max_bits)

    def test_sparse_max_bits(self):
        '''The max bits property should work for sparse arrays.
        '''

        # test set is (min val, max val, fractional bits, max bits)
        test_set = (
            (-5, 5, 1, 5),
            (-4, 5, 1, 5),
            (-4, 4, 1, 5),
            (-4, 3, 1, 4), # -4 needs 2 bits + sign bit
            (-5, 3, 1, 5),
            (-0.6, 0, 0, 1), # The wierd case in which -1 needs no integer bits
            (-0.6, 0.6, 0, 2), # But the positive integer does (0.6 rounded)
            (0, 0, 0, 0),
            (0, 2, 0, 2),
            (1.8, 1.8, 0, 2),
            (-1.8, 1.8, 0, 3),
            (-3, 3, 1, 4),
            (-5, 5, -1, 3),
            (-5, 5, -2, 2),
            (-5, 5, -3, 2), # rounds to -8 and 8 (round to even)
            (-5, 0, -3, 1), # -8 needs only 1 bit!
            (-4, 0, -3, 0), # rounds to zero
            (0, 4, -3, 0), # rounds to zero
        )

        for min_val, max_val, frac_bits, expected_max_bits in test_set:
            a = sparse.csc_matrix(np.array([min_val, max_val]))
            fp_a = self.fp_class(a, fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, fp_a.max_bits)

            a = sparse.csc_matrix(np.array([min_val*1j, max_val*1j]))
            fp_a = self.fp_class(a, fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, fp_a.max_bits)

            a = sparse.csc_matrix(np.array([min_val*(1+1j), max_val*(1+1j)]))
            fp_a = self.fp_class(a, fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, fp_a.max_bits)

    def test_fix_to_bitwidth(self):
        '''There should be a method to fix an array to some bitwidth

        The bitwidth should be provided by an argument.

        Values that are longer than the supplied bitwidth are truncated to
        the defined bitwidth.

        Values that are shorter are zero extended at the low end to fill
        the desired bitwidth (that is, the number of fractional bits is
        increased).
        '''
        cmplx_a = np.random.randn(5) + 1j*np.random.randn(5)

        max_bits = 20

        for scale_bits in (0, 5, 10, 20):
            for frac_bits in (-5, -3, -1, 0, 4, 7, 10, 15):
                test_a = cmplx_a * 2**scale_bits
                fp_cmplx_a = self.fp_class(test_a, frac_bits)

                fixed_fp_cmplx_a = fp_cmplx_a.fix_to_bitwidth(max_bits)

                ref_int_a = f_round(test_a * 2**frac_bits)

                max_val = max((np.max(np.abs(ref_int_a.real)),
                               np.max(np.abs(ref_int_a.imag))))

                if max_val == 0.0:
                    ref_int_a_bits = 0
                else:
                    ref_int_a_bits = np.floor(np.log2(max_val)) + 1

                # The -1 is because of the sign bit
                extra_bits = max_bits - ref_int_a_bits - 1

                fixed_ref_int_a = f_round(ref_int_a * 2**extra_bits)
                new_max_val = max((np.max(np.abs(fixed_ref_int_a.real)),
                                   np.max(np.abs(fixed_ref_int_a.imag))))

                if new_max_val == 0.0:
                    max_int_val_bits = 0
                else:
                    max_int_val_bits = np.floor(np.log2(new_max_val)) + 1

                if max_int_val_bits > max_bits:
                    # The rounding caused an extra bit
                    extra_bits -= 1
                    fixed_ref_int_a = f_round(ref_int_a * 2**extra_bits)

                fixed_ref_fp_a = (fixed_ref_int_a *
                                  2**(-(frac_bits + extra_bits)))

                self.assertEqual(fixed_fp_cmplx_a.max_bits, max_bits)

                self.assertTrue(np.all(
                    fixed_ref_fp_a == fixed_fp_cmplx_a.as_floating_point()))

if __name__ == "__main__":
    unittest.main()

class TestSignedFixedPointArray(TestFixedPointArray):
    '''There should be an equivalent of the FixedPointArray but in which the
    array (and by extension, the outputs from any operations that use the
    array) *always* include a sign bit.
    '''

    def __init__(self, *args, **kwargs):
        super(TestFixedPointArray, self).__init__(*args, **kwargs)

        self.fp_class = SignedFixedPointArray

    def test_max_bits(self):
        '''There should be a ``max_bits`` property.
        The ``max_bits`` property should be the number of bits required to
        store the largest value in the array up to the predefined fractional
        precision. It should be the sum of the fractional bits and the
        maximum integer bits.

        Different to the equivalent test case in TestFixedPointArray,
        the sign bit should always be included.

        A zero array should have the number of fractional bits defined by
        the fractional bits (and, of course, the sign bit). This is because
        zero is always to a certain precision.
        '''

        # test set is (min val, max val, fractional bits, max bits)
        test_set = (
            (-5, 5, 1, 5),
            (-4, 5, 1, 5),
            (-4, 4, 1, 5),
            (-4, 3, 1, 4), # -4 needs 2 bits + sign bit
            (-5, 3, 1, 5),
            (-0.6, 0, 0, 1), # The wierd case in which -1 needs no integer bits
            (-0.6, 0.6, 0, 2), # But the positive integer does (0.6 rounded)
            (0, 0, 0, 1),
            (0, 2, 0, 3),
            (1.8, 1.8, 0, 3),
            (-1.8, 1.8, 0, 3),
            (-3, 3, 1, 4),
            (-5, 5, -1, 3),
            (-5, 5, -2, 2),
            (-5, 5, -3, 2), # rounds to -8 and 8 (round to even)
            (-5, 0, -3, 1), # -8 needs only 1 bit!
            (-4, 0, -3, 1), # rounds to zero
            (0, 4, -3, 1), # rounds to zero
        )

        for min_val, max_val, frac_bits, expected_max_bits in test_set:
            a = self.fp_class([min_val, max_val], fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, a.max_bits)

            a = self.fp_class(
                [min_val*1j, max_val*1j], fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, a.max_bits)

            a = self.fp_class(
                [min_val*(1+1j), max_val*(1+1j)], fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, a.max_bits)

    def test_sparse_max_bits(self):
        '''The max bits property should work for sparse arrays.
        '''

        # test set is (min val, max val, fractional bits, max bits)
        test_set = (
            (-5, 5, 1, 5),
            (-4, 5, 1, 5),
            (-4, 4, 1, 5),
            (-4, 3, 1, 4), # -4 needs 2 bits + sign bit
            (-5, 3, 1, 5),
            (-0.6, 0, 0, 1), # The wierd case in which -1 needs no integer bits
            (-0.6, 0.6, 0, 2), # But the positive integer does (0.6 rounded)
            (0, 0, 0, 1),
            (0, 2, 0, 3),
            (1.8, 1.8, 0, 3),
            (-1.8, 1.8, 0, 3),
            (-3, 3, 1, 4),
            (-5, 5, -1, 3),
            (-5, 5, -2, 2),
            (-5, 5, -3, 2), # rounds to -8 and 8 (round to even)
            (-5, 0, -3, 1), # -8 needs only 1 bit!
            (-4, 0, -3, 1), # rounds to zero
            (0, 4, -3, 1), # rounds to zero
        )

        for min_val, max_val, frac_bits, expected_max_bits in test_set:
            a = sparse.csc_matrix(np.array([min_val, max_val]))
            fp_a = self.fp_class(a, fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, fp_a.max_bits)

            a = sparse.csc_matrix(np.array([min_val*1j, max_val*1j]))
            fp_a = self.fp_class(a, fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, fp_a.max_bits)

            a = sparse.csc_matrix(np.array([min_val*(1+1j), max_val*(1+1j)]))
            fp_a = self.fp_class(a, fractional_bits=frac_bits)
            self.assertEqual(expected_max_bits, fp_a.max_bits)

