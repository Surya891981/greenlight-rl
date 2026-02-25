import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import math

# ------------------------------------------------------------
# 🌡 Greenhouse Simulation Environment
# ------------------------------------------------------------

class GreenLightEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # ---- Observation Space ----
        # temp, humidity, co2, outside_temp, energy_usage
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -10, 0]),
            high=np.array([50, 100, 2000, 50, 100]),
            dtype=np.float32
        )

        # ---- Action Space ----
        # heating, ventilation, co2_injection
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.reset()

    # --------------------------------------------------------
    # 🔄 Reset Environment
    # --------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.temp = 22.0
        self.humidity = 60.0
        self.co2 = 400.0
        self.outside_temp = np.random.uniform(10, 35)

        self.energy_usage = 0.0
        self.timestep = 0

        obs = self._get_obs()
        return obs, {}

    # --------------------------------------------------------
    # 🎯 Step Function
    # --------------------------------------------------------

    def step(self, action):
        heating, ventilation, co2_injection = action

        # ---- Climate Dynamics ----

        # Temperature dynamics
        self.temp += (
            heating * 0.8
            - ventilation * 0.6
            + (self.outside_temp - self.temp) * 0.05
        )

        # Humidity dynamics
        self.humidity += (
            - ventilation * 1.2
            + np.random.normal(0, 0.3)
        )

        # CO₂ dynamics
        self.co2 += (
            co2_injection * 50
            - ventilation * 30
            - 5
        )

        # Bound values
        self.temp = np.clip(self.temp, 0, 50)
        self.humidity = np.clip(self.humidity, 0, 100)
        self.co2 = np.clip(self.co2, 300, 2000)

        # ---- Energy Cost Model ----
        energy_cost = (
            heating * 2.5
            + ventilation * 1.2
            + co2_injection * 0.8
        )

        self.energy_usage += energy_cost

        # ---- Reward Engineering ----

        temp_penalty = -abs(self.temp - 24)
        humidity_penalty = -abs(self.humidity - 65) * 0.1
        co2_penalty = -abs(self.co2 - 800) * 0.01
        energy_penalty = -energy_cost * 0.5

        reward = (
            temp_penalty
            + humidity_penalty
            + co2_penalty
            + energy_penalty
        )

        # ---- Termination ----
        self.timestep += 1
        done = self.timestep >= 200

        obs = self._get_obs()
        info = {}

        return obs, reward, done, False, info

    # --------------------------------------------------------
    # 👁 Observation Builder
    # --------------------------------------------------------

    def _get_obs(self):
        return np.array([
            self.temp,
            self.humidity,
            self.co2,
            self.outside_temp,
            self.energy_usage
        ], dtype=np.float32)


# ------------------------------------------------------------
# 📊 Custom Training Logger Callback
# ------------------------------------------------------------

class TrainingLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        self.rewards.append(self.locals["rewards"])
        return True


# ------------------------------------------------------------
# 🚀 Environment Setup
# ------------------------------------------------------------

def make_env():
    return GreenLightEnv()

env = DummyVecEnv([make_env])

# ------------------------------------------------------------
# 🧠 PPO Model
# ------------------------------------------------------------

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

logger = TrainingLogger()

# ------------------------------------------------------------
# 🎯 Training
# ------------------------------------------------------------

model.learn(
    total_timesteps=200_000,
    callback=logger
)

model.save("greenlight_ppo")

# ------------------------------------------------------------
# 🧪 Evaluation Loop
# ------------------------------------------------------------

eval_env = GreenLightEnv()
obs, _ = eval_env.reset()

total_reward = 0

for step in range(500):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = eval_env.step(action)

    total_reward += reward

    if done:
        obs, _ = eval_env.reset()

print("Total Evaluation Reward:", total_reward)