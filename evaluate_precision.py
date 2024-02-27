import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calc_squared_error(data, theta):
    estimate = theta[0] + (theta[1] * data[0])
    return (data[1] - estimate) ** 2

def calc_absolute_error(data, theta):
    estimate = theta[0] + (theta[1] * data[0])
    return np.abs(data[1] - estimate)

def calc_r_square(dataset, theta):
    data_average = 0
    residual_sum_square = 0
    total_sum_square = 0

    for data in dataset:
        data_average += data[1]
        residual_sum_square += calc_squared_error(data, theta)
    data_average /= dataset.shape[0]

    for data in dataset:
        total_sum_square += (data[1] - data_average) ** 2
    return 1 - (residual_sum_square / total_sum_square)


def calc_rmse(dataset, theta):
    total = 0
    for data in dataset:
        total += calc_squared_error(data, theta)
    total /= dataset.shape[0]
    return np.sqrt(total)


def calc_mae(dataset, theta):
    total = 0
    for data in dataset:
        total += calc_absolute_error(data, theta)
    total /= dataset.shape[0]
    return total


def evaluate_precision():
    dataset = pd.read_csv("data.csv").to_numpy()
    file = Path("train_parameter.json")
    if file.exists():
        with open("train_parameter.json", 'r') as f:
            parameter = json.load(f)
            theta = np.array([parameter['theta0'], parameter['theta1']])
            print(f"R^2 value: {calc_r_square(dataset, theta)}")
            print(f"RMSE value: {calc_rmse(dataset, theta)}")
            print(f"MAE value: {calc_mae(dataset, theta)}")
    else:
        print("Please complete the model training first.")
        print("(There is no train_parameter.json file)")


def main():
    """
    main() -> None

    main function, Entrypoint when this module is executed directly.
    """
    evaluate_precision()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"AssertionError: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Exception Occurred!", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
