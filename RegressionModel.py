import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class RegressionModel:
    def __init__(self, datapath, outputpath, epsilon):
        self.learnData = pd.read_csv(datapath).to_numpy()
        self.learnDataSize = self.learnData.shape[0]
        self.last_theta = np.array([0, 0])
        self.theta = np.array([0, 0])
        self.epilon = epsilon
        self.outputpath = outputpath

    def training(self) -> bool:
        gradient = self.gradient_err_func(self.theta)
        if np.all(self.theta == 0):
            learningRate = 1e-5
        else:
            last_gradient = self.gradient_err_func(self.last_theta)
            delta_gradient = gradient - last_gradient
            learningRate = (
                np.abs((self.theta - self.last_theta) @ (delta_gradient)) \
                / (delta_gradient @ delta_gradient)
            )
        tmp_theta = self.theta - learningRate * gradient
        if np.array_equal(tmp_theta, self.theta):
            return False
        self.last_theta = self.theta
        self.theta = tmp_theta
        return True

    def estimate(self, mile: float, theta) -> float:
        return theta[0] + (theta[1] * mile)

    def gradient_err_func(self, theta: np.ndarray) -> np.ndarray:
        dtheta0 = 0
        for data in self.learnData:
            dtheta0 += self.estimate(data[0], theta) - data[1]
        dtheta0 /= self.learnDataSize
        dtheta1 = 0
        for data in self.learnData:
            dtheta1 += (self.estimate(data[0], theta) - data[1]) * data[0]
        dtheta1 /= self.learnDataSize
        return np.array([dtheta0, dtheta1])
    
    def write_result_parameter(self):
        parameter = {'theta0': self.theta[0], 'theta1': self.theta[1]}
        with open(self.outputpath, 'w') as f:
            json.dump(parameter, f)

    def calc_r_square(self):
        data_average = 0
        residual_sum_square = 0
        total_sum_square = 0
        for data in self.learnData:
            data_average += data[1]
            residual_sum_square += (data[1] - self.estimate(data[0], self.theta)) ** 2
        data_average /= self.learnDataSize

        for data in self.learnData:
            total_sum_square += (data[1] - data_average) ** 2

        return 1 - (residual_sum_square / total_sum_square)

    def view_plot(self):
        plt.figure(figsize=(16, 9))

        plt.text(230000, 8000, f'R^2 = {self.calc_r_square():.5f}', fontsize=12)

        dist = self.learnData[:, 0]
        price = self.learnData[:, 1]
        plt.scatter(dist, price, label='DataSet')

        x = np.linspace(0, 250000, 100)
        y = self.theta[0] + self.theta[1] * x
        plt.plot(x, y, 'r-', label='estimated function')

        plt.title('Linear Regression')
        plt.xlabel('distance (mile)')
        plt.ylabel('price (doller)')

        plt.legend()
        plt.show()