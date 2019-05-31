import numpy as np
from scipy import stats


class ModelComparison:
    """Static class that implements techniques for model(classifiers)
    comparison.

    Reference: https://arxiv.org/pdf/1811.12808.pdf.
    """
    def __init__(self):
        pass

    @staticmethod
    def mcnemar_table(y_true, y_clf1, y_clf2):
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

    @staticmethod
    def mcnemar_test(table, correction=True):
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

    @staticmethod
    def cochran_q_test(y_true, y_models):
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
        # right. O(n) loop.
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
