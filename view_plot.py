import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def view_plot():
    dataset = pd.read_csv("data.csv").to_numpy()
    file = Path("train_parameter.json")
    if not file.exists():
        print("Please complete the model training first.")
        print("(There is no train_parameter.json file)")
        return
    with open("train_parameter.json", 'r') as f:
        parameter = json.load(f)
        theta = np.array([parameter['theta0'], parameter['theta1']])

    plt.figure(figsize=(16, 9))

    dist = dataset[:, 0]
    price = dataset[:, 1]
    plt.scatter(dist, price, label='DataSet')
    
    x = np.linspace(0, 250000, 100)
    y = theta[0] + theta[1] * x
    plt.plot(x, y, 'r-', label='estimated function')
    
    plt.title('Linear Regression')
    plt.xlabel('distance (mile)')
    plt.ylabel('price (doller)')
    plt.legend()
    plt.show()


def main():
    """
    main() -> None

    main function, Entrypoint when this module is executed directly.
    """
    view_plot()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"AssertionError: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Exception Occurred!", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
