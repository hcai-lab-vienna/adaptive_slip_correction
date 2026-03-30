import numpy as np

class PageHinkley:
    def __init__(self, delta=0.005, lambda_=50, alpha=0.99):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0
        self.cumulative = 0
        self.minimum = 0
        self.t = 0

    def update(self, value):
        self.t += 1

        # actualización media incremental
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value

        # acumulador PH
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)

        # condición de drift
        if (self.cumulative - self.minimum) > self.lambda_:
            self.reset()
            return True

        return False