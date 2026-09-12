#!/usr/bin/python3

import unittest
from unittest.mock import patch

from src.mystery_data import mystery_data


class TestMysteryData(unittest.TestCase):

    def test_result(self):
        coefficients = mystery_data()
        self.assertAlmostEqual(
            -11,
            sum(coefficients),
            msg="Sum of the fitted regression coefficients should be "
            "about -11, got %r." % (sum(coefficients),),
        )

    def test_calls(self):
        with patch("src.mystery_data.LinearRegression") as linreg:
            mystery_data()
            if len(linreg.call_args[0]) > 0:
                self.assertFalse(
                    linreg.call_args[0][0],
                    msg="LinearRegression must be constructed with "
                    "fit_intercept=False - do not fit the intercept.",
                )
            else:
                linreg.assert_called_once_with(fit_intercept=False)
            a, b = linreg().fit.call_args[0]
            shape = (1000, 5)
            self.assertEqual(
                a.shape,
                shape,
                msg="fit's first argument (the X matrix) should have "
                "shape %r, got %r." % (shape, a.shape),
            )
            self.assertEqual(
                len(b),
                1000,
                msg="fit's second argument (the y values) should have "
                "length 1000, got %r." % (len(b),),
            )


if __name__ == '__main__':
    unittest.main()
