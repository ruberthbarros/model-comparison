from comparison import McNemarTest, CochranQTest
import numpy as np
import argparse
import pathlib


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default=None, required=True,
                        help='Path to file with test samples values and '
                             'classifier results.')

    return parser.parse_args()


def read_input(file_path, delimiter=';'):
    """ Read a input csv file with the results of M classifiers.

    Parameters
    ----------

    file_path: str
    delimiter: str, optional, default: ';'

    Returns
    -------

    y_true: numpy.array, shape(n, 1)
        The first column of input csv file that represents the ground truth
        test samples.
    clfs_array: numpy.array, shape(n, m)
        Matrix where each column is the result of a classifier C1...Cm.
    """
    content_array = np.genfromtxt(file_path, delimiter=delimiter,
                                  skip_header=1, dtype=np.uint8)

    y_true = content_array[:, 0]
    clfs_array = content_array[:, 1:]

    return y_true, clfs_array


if __name__ == '__main__':
    args = get_arguments()
    filepath = pathlib.Path(args.file)
    y_true, y_models = read_input(filepath)

    mcnemar = McNemarTest(y_true, y_models)
    mcnemar = mcnemar.evaluate()

    print(mcnemar.test_result_)

    cochran = CochranQTest(y_true, y_models)
    cochran = cochran.evaluate()

    print(cochran.test_result_)

    cochran_mcnemar = CochranQTest(y_true, y_models, use_mcnemar=True)
    cochran_mcnemar = cochran_mcnemar.evaluate()

    print(cochran_mcnemar.mcnemar_test_result_)
