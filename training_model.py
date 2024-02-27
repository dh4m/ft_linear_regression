import sys
import time

import numpy as np
from matplotlib import pyplot as plt

from RegressionModel import RegressionModel

def main():
    """
    main() -> None

    main function, Entrypoint when this module is executed directly.
    """
    model = RegressionModel("data.csv", "train_parameter.json", 1e-5)
    while model.training():
        print(f"\rtheta0: {model.theta[0]}, theta1: {model.theta[1]}", end="")
    print()
    model.write_result_parameter()
    model.view_plot()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"AssertionError: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Exception Occurred!", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
