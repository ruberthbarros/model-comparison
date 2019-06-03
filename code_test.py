"""Test code via pytest."""


import pytest

import numpy as np

from utils import read_input
from comparison import McNemarTest, CochranQTest


def create_mcnemartest_instance(correction=True):
    y_true, y_models = read_input('Classifiers_Results.csv')

    mcnemar = McNemarTest(y_true, y_models[:, :2], correction=correction)

    return mcnemar.evaluate()


def create_cochranqtest_instance(use_mcnemar=False):
    y_true, y_models = read_input('Classifiers_Results.csv')

    cochran = CochranQTest(y_true, y_models, use_mcnemar=use_mcnemar)

    return cochran.evaluate()


class TestModelComparison:

    @pytest.mark.parametrize('classinfo, expected', [
        (create_mcnemartest_instance(), np.array([[83, 9], [3, 5]]))
    ])
    def test_mcnemar_table_attribute(self, classinfo, expected):
        # If all elements are equal, the sum should be 4.
        assert np.sum(np.uint8(expected == classinfo.tables_['M0-M1'])) == 4

    @pytest.mark.parametrize('classinfo, expected', [
        (create_mcnemartest_instance(), 2.083)
    ])
    def test_mcnemar_coefficient_attribute(self, classinfo, expected):
        assert round(classinfo.coefficient_['M0-M1'], 3) == expected

    @pytest.mark.parametrize('classinfo, expected', [
        (create_mcnemartest_instance(), 0.149)
    ])
    def test_mcnemar_p_values_attribute(self, classinfo, expected):
        assert round(classinfo.p_value_['M0-M1'], 3) == expected

    @pytest.mark.parametrize('classinfo, expected', [
        (create_mcnemartest_instance(), False)
    ])
    def test_mcnemar_test_result_attribute(self, classinfo, expected):
        assert classinfo.test_result_['M0-M1'] == expected

    @pytest.mark.parametrize('classinfo, expected', [
        (create_cochranqtest_instance(), False)
    ])
    def test_cochran_test_result_attribute(self, classinfo, expected):
        assert classinfo.test_result_['all'] == expected
