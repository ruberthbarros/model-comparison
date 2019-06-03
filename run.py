"""Runs two methods implemented in the comparison package as example"""

from comparison import McNemarTest, CochranQTest
from utils import get_arguments, read_input

import pathlib


if __name__ == '__main__':
    args = get_arguments()
    filepath = pathlib.Path(args.file)
    y_true, y_models = read_input(filepath)

    # Default McNemarTest - Continuity Correction
    mcnemar = McNemarTest(y_true, y_models)
    mcnemar = mcnemar.evaluate()

    print(mcnemar.tables_)
    print(mcnemar.coefficient_)
    print(mcnemar.p_value_)
    print(mcnemar.test_result_)

    # Default CochranQTest - Without McNemarTest
    cochran = CochranQTest(y_true, y_models)
    cochran = cochran.evaluate()

    print(cochran.test_result_)

    # CochranQTest + McNemarTest
    cochran_mcnemar = CochranQTest(y_true, y_models, use_mcnemar=True)
    cochran_mcnemar = cochran_mcnemar.evaluate()

    print(cochran_mcnemar.mcnemar_test_result_)
