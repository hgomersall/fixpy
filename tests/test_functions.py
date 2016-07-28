import unittest

from fixpy import (
    FixedPointArray, SignedFixedPointArray, fixed_point_convolve)
import numpy as np
import random

class TestFixedPointConvolve(unittest.TestCase):
    '''There should be a function equivalent to numpy.convolve that acts on
    signed or unsigned fixed point arrays, returning a fixed point array.
    '''

    def test_fixed_point_convolve(self):
        '''The fixed point convolve function should be equivalent to numpy's
        convolve function, but acting on fixed point arrays with fixed point
        computation.
        '''

        for kind in ('full', 'valid', 'same'):
            a = np.random.randn(random.randrange(50, 500))
            b = np.random.randn(random.randrange(50, 500))

            fractional_bits_a = random.randrange(3, 20)
            fractional_bits_b = random.randrange(3, 20)

            fp_a = FixedPointArray(a, fractional_bits_a)
            fp_b = FixedPointArray(b, fractional_bits_b)

            test_out = fixed_point_convolve(fp_a, fp_b, kind)

            ref_out_data = np.convolve(
                fp_a.as_floating_point(), fp_b.as_floating_point(), kind)

            self.assertTrue(
                np.all(ref_out_data == test_out.as_floating_point()))

    def test_fixed_point_convolve_with_Signed(self):
        '''The fixed point convolve function should work with
        SignedFixedPoint arrays.
        '''

        for kind in ('full', 'valid', 'same'):
            a = np.random.randn(random.randrange(50, 500))
            b = np.random.randn(random.randrange(50, 500))

            fractional_bits_a = random.randrange(3, 20)
            fractional_bits_b = random.randrange(3, 20)

            fp_a = FixedPointArray(a, fractional_bits_a)
            sfp_a = SignedFixedPointArray(a, fractional_bits_a)
            fp_b = FixedPointArray(b, fractional_bits_b)
            sfp_b = SignedFixedPointArray(b, fractional_bits_b)

            test_out = fixed_point_convolve(fp_a, fp_b, kind)

            out_fpa_fpb = fixed_point_convolve(fp_a, fp_b, kind)
            out_fpa_sfpb = fixed_point_convolve(fp_a, sfp_b, kind)
            out_sfpa_fpb = fixed_point_convolve(sfp_a, fp_b, kind)
            out_sfpa_sfpb = fixed_point_convolve(sfp_a, sfp_b, kind)

            self.assertTrue(np.all(out_fpa_fpb.data == out_fpa_sfpb.data))
            self.assertTrue(np.all(out_fpa_fpb.data == out_sfpa_fpb.data))
            self.assertTrue(np.all(out_fpa_fpb.data == out_sfpa_sfpb.data))

            self.assertEqual(out_fpa_fpb.fractional_bits,
                             out_fpa_sfpb.fractional_bits)
            self.assertEqual(out_fpa_fpb.fractional_bits,
                             out_sfpa_fpb.fractional_bits)
            self.assertEqual(out_fpa_fpb.fractional_bits,
                             out_sfpa_sfpb.fractional_bits)

    def test_either_input_Signed_returns_SignedFixedPointArray(self):
        '''If either of the inputs is a SignedFixedPointArray then so is the
        output.
        '''
        for kind in ('full', 'valid', 'same'):
            a = np.random.randn(random.randrange(50, 500))
            b = np.random.randn(random.randrange(50, 500))

            fractional_bits_a = random.randrange(3, 20)
            fractional_bits_b = random.randrange(3, 20)

            fp_a = FixedPointArray(b, fractional_bits_b)
            sfp_a = SignedFixedPointArray(a, fractional_bits_a)
            fp_b = FixedPointArray(b, fractional_bits_b)
            sfp_b = SignedFixedPointArray(b, fractional_bits_b)

            out_fpa_fpb = fixed_point_convolve(fp_a, fp_b, kind)
            out_fpa_sfpb = fixed_point_convolve(fp_a, sfp_b, kind)
            out_sfpa_fpb = fixed_point_convolve(sfp_a, fp_b, kind)
            out_sfpa_sfpb = fixed_point_convolve(sfp_a, sfp_b, kind)

            self.assertTrue(isinstance(out_fpa_fpb, FixedPointArray))
            self.assertTrue(isinstance(out_fpa_sfpb, SignedFixedPointArray))
            self.assertTrue(isinstance(out_sfpa_fpb, SignedFixedPointArray))
            self.assertTrue(isinstance(out_sfpa_sfpb, SignedFixedPointArray))
