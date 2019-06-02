import numpy as np
from scipy import stats


def _mcnemar_table(y_true, y_clf1, y_clf2):
    """ Creates the (2, 2) MacNemar table in the following format:
    |A|B|
    |C|D|
    where:
    A: # of correcly classified items by both models.
    B: # of correcly classified items by y_clf2 that y_clf1 got wrong.
    C: # of correctly classified items by y_clf1 that y_clf2 got wrong.
    D: # of wrongly classified items by both models.

    Reference: https://arxiv.org/pdf/1811.12808.pdf.

    Parameters
    ----------

    y_true: numpy.array shape (n,).
        The ground truth results.
    y_clf1: numpy.array shape (n,).
        The results of classifier 1.
    y_clf2: numpy.array shape (n,).
        The results of classifier 1.

    Returns
    -------
    table: numpy.array shape(2, 2)
        The McNemar table.
    """

    table = np.zeros(4, dtype=np.int).reshape(2, 2)

    # [0,1] array where y_true equals y_cls1 element-wise
    ytc1_equal = np.uint8((y_true == y_clf1))

    # [0,1] array where y_true equals y_cls2 element-wise
    ytc2_equal = np.uint8((y_true == y_clf2))

    # [0,1] array where y_true differs from y_cls1 element-wise
    ytc1_diff = np.uint8((y_true != y_clf1))

    # [0,1] array where y_true differs from y_cls2 element-wise
    ytc2_diff = np.uint8((y_true != y_clf2))

    table[0, 0] = np.sum(ytc1_equal & ytc2_equal)
    table[0, 1] = np.sum(ytc1_diff & ytc2_equal)
    table[1, 0] = np.sum(ytc1_equal & ytc2_diff)
    table[1, 1] = np.sum(ytc1_diff & ytc2_diff)

    return table


def _multiple_mcnemar_tables(y_true, y_models):
    """ Creates a (2, 2) MacNemar table for each pair of classifiers
    results to be compared in the following format:
    |A|B|
    |C|D|
    where:
    A: # of correcly classified items by both models.
    B: # of correcly classified items by y_clf2 that y_clf1 got wrong.
    C: # of correctly classified items by y_clf1 that y_clf2 got wrong.
    D: # of wrongly classified items by both models.
    Reference: https://arxiv.org/pdf/1811.12808.pdf.
    Parameters
    ----------
    y_true: numpy.array shape (n,).
        The ground truth results.
    y_models: numpy.array shape (n, m).
        Array with all classifiers results.
    Returns
    -------
    table: list of m numpy.arrays of shape(2, 2)
        The list of McNemar tables.
    """

    tables = dict()
    m = y_models.shape[1]

    # Generates each unique combination of m classifiers.
    for iindex in np.arange(m):
        for jindex in np.arange(iindex + 1, m):
            y_clf1 = y_models[:, iindex]
            y_clf2 = y_models[:, jindex]

            combination_key = f'M{iindex}-M{jindex}'
            tables[combination_key] = _mcnemar_table(y_true, y_clf1, y_clf2)

    return tables


def _mcnemar_test(table, correction=True):
    """ McNemar test implemented as following:
    chi2 = ((B - C) ** 2) / B + C, if correction=False.
    chi2 = ((abs(B - C) - 1) ** 2) / B + C, if correction=True

    Reference: https://arxiv.org/pdf/1811.12808.pdf.

    Parameters
    ----------

    table: numpy.array, shape(2, 2).
        The McNemar table.
    correction: Bool, optional, default: False.
        Flag that indicates if the continuous correction by
        L. Edwards should be used.

    Returns
    -------
    chi_sqrd: float.
        The resulting value form the indicated formula.
    p_value: float.
        The p-value from a chi2 distribution with 1 degree of
        freedom given the chi_sqrd resulting value.
    """

    b = table[0, 1]
    c = table[1, 0]
    denominator = b + c

    # Degree of freedom always 1 for McNemar Test
    df = 1

    if not correction:
        numerator = (b - c) ** 2
    else:
        numerator = (np.abs(b - c) - 1) ** 2

    chi_sqrd = numerator / denominator

    p_value = 1 - stats.chi2.cdf(chi_sqrd, df)

    return chi_sqrd, p_value


def _cochran_q_test(y_true, y_models):
    """Cochran's Q Test implemented as following:
    Q = (M - 1) * (M * sum(Gi²) - T²) / M * T - sum(Mj²), where:

    M: # of classifiers C1...Cm.
    Gi: # of correctly classified test samples by classifier C1...Cm.
    Mj: # of classifiers C1...Cm that correctly classified test sample j.
    T: # of correctly classified test samples by all classifiers.

    Reference: https://arxiv.org/pdf/1811.12808.pdf.

    Helper table:

    |C1|C2|C3|...|Cm|# of cases where the given combination occurs|
    | 1| 1| 1|...|Cm|#|
    | 1| 1| 0|...|Cm|#|
    | 1| 0| 0|...|Cm|#|
    | 0| 0| 1|...|Cm|#|
    |..|..|..|...|..|#|

    Parameters
    ----------

    y_true: numpy.array shape (n,).
        The ground truth results.
    y_models: numpy.array shape (n, m).
        The results of all classifiers.

    Returns
    -------
    q: float.
        The Q statistic.
    p_value: float.
        The p-value from a chi2 distribution with M - 1 degrees of
        freedom given the q resulting value.
    """
    n = y_true.shape[0]
    m = y_models.shape[1]

    # M - 1 degrees of freedom
    df = m - 1

    # Gi part of the equation.
    g_array = np.zeros(m)

    # (n, m) matrix to hold all classifiers predictions.
    clfs_matrix = np.zeros(n * m).reshape(n, m)

    for index in np.arange(m):
        # [0,1] array where y_true equals y_model_index element-wise
        yt_mi_equal = np.uint8((y_models[:, index] == y_true))
        g_array[index] = np.sum(yt_mi_equal)

        # Populates each column with 1, if prediction is right, 0 otherwise.
        clfs_matrix[:, index] = yt_mi_equal

    g = sum(g_array ** 2)

    # T part of the equation
    t = np.sum(g_array)

    # Array to hold the step necessary to calculate the Mj² part of
    # the equation.
    mj_sums = np.zeros(n)

    # Calculates the number of classifiers that got test sample i(index)
    # right.
    for index in np.arange(n):
        mj_sums[index] += np.sum(clfs_matrix[index, :])

    # Calculates unique values based on the combinations of the helper table
    n_classifiers_right, counts = np.unique(mj_sums, return_counts=True)
    sum_sqrd_mj = np.sum((n_classifiers_right ** 2) * counts)

    numerator = m * g - (t ** 2)
    denominator = m * t - sum_sqrd_mj

    q = (m - 1) * numerator / denominator

    p_value = 1 - stats.chi2.cdf(q, df)

    return q, p_value


def _check_difference(p_values, alpha):
    test_diff_result = dict()
    for key, value in p_values.items():
        has_diff = value < alpha
        test_diff_result[key] = has_diff

    return test_diff_result


class ModelComparison:
    """Base class for all model comparison classes."""
    def __init__(self, y_true, y_models):
        assert y_true.shape[0] == y_models.shape[0], \
            'Input arrays should have the same (n,) size.'

        assert y_models.shape[1] > 1, \
            'y_models should have at least shape(n, 2).'

        self._y_true = y_true
        self._y_models = y_models
        self.coefficient_ = dict()
        self.p_value_ = dict()
        self.test_result_ = dict()

    def evaluate(self, alpha):
        """Base method for models comparison."""
        raise NotImplementedError


class McNemarTest(ModelComparison):
    """Class for McNemar Test.

    Reference: https://arxiv.org/pdf/1811.12808.pdf.
    """
    def __init__(self, y_true, y_models, correction=True):
        ModelComparison.__init__(self, y_true, y_models)
        self.tables_ = dict()
        self._correction = correction

    def evaluate(self, alpha=0.05):
        self.tables_ = _multiple_mcnemar_tables(self._y_true, self._y_models)

        # Calculates each pair chi2 and p_value for each mcnemar table
        for combination, table in self.tables_.items():
            chi2, p_value = _mcnemar_test(table, correction=self._correction)
            self.coefficient_[combination] = chi2
            self.p_value_[combination] = p_value

        self.test_result_ = _check_difference(self.p_value_, alpha)

        return self


class CochranQTest(ModelComparison):
    """Class for Cochran's Q Test.

    Reference: https://arxiv.org/pdf/1811.12808.pdf.
    """
    def __init__(self, y_true, y_models, use_mcnemar=False):
        ModelComparison.__init__(self, y_true, y_models)
        self._use_mcnemar = use_mcnemar
        if use_mcnemar:
            self.mcnemar_tables_ = None
            self.mcnemar_coefficients_ = None
            self.mcnemar_p_values_ = None
            self.mcnemar_test_result_ = None

    def evaluate(self, alpha=0.05):
        q, p_value = _cochran_q_test(self._y_true, self._y_models)

        self.coefficient_['all'] = q
        self.p_value_['all'] = p_value

        has_difference = p_value < alpha
        self.test_result_['all'] = has_difference

        if not has_difference:
            return self

        if has_difference and not self._use_mcnemar:
            return self

        self.mcnemar_tables_ = _multiple_mcnemar_tables(self._y_true,
                                                        self._y_models)

        self.mcnemar_coefficients_ = dict()
        self.mcnemar_p_values_ = dict()
        for combination, table in self.mcnemar_tables_.items():
            chi2, p_value = _mcnemar_test(table)
            self.mcnemar_coefficients_[combination] = chi2
            self.mcnemar_p_values_[combination] = p_value

        self.mcnemar_test_result_ = \
            _check_difference(self.mcnemar_p_values_, alpha)

        return self
