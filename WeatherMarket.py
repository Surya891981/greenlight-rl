import numpy as np
import math

class WeatherMarket:
    def __init__(self, episode_length):
        self.T = episode_length
        self._generate()

    def _generate(self):
        t = np.linspace(0, 2 * np.pi, self.T)

        self.outside_temp = 18 + 12 * np.sin(t)
        self.radiation = np.maximum(0, 900 * np.sin(t))
        self.humidity = 55 + 25 * np.sin(t)

        # Dynamic electricity pricing
        self.energy_price = 5 + 3 * np.sin(t + np.pi / 3)

    def get(self, step):
        return (
            self.outside_temp[step],
            self.radiation[step],
            self.humidity[step],
            self.energy_price[step]
        )