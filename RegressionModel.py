import numpy as np
import pandas as pd

class RegressionModel:
    def __init__(self, path):
        self.learnData = pd.read_csv(path).to_numpy()
        self.learnDataSize = self.learnData.shape[0]
        self.last_theta = np.array([0, 0])
        self.theta = np.array([0, 0])

    def learn(self) -> bool:
        gradient = self.gradient_err_func(self.theta)
        last_gradient = self.gradient_err_func(self.last_theta)
        delta_gradient = gradient - last_gradient
        learningRate = (
            np.abs((self.theta - self.last_theta) @ (delta_gradient)) \
            / (delta_gradient @ delta_gradient)
        ) * 2
        tmp_theta = self.theta - learningRate * gradient
        if self.err_average(tmp_theta) < self.err_average(self.theta):
            self.last_theta = self.theta
            self.theta = tmp_theta
            return True
        return False
        
    def estimate(self, mile: float) -> float:
        return self.theta[0] + self.theta[1] * mile
    
    def gradient_err_func(self, theta: np.ndarray) -> np.ndarray:
        pass

    def err_average(self, theta: np.ndarray) -> np.ndarray:
        pass
