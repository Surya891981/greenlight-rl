import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import your environment
from greenlight_gym.envs import GreenLightEnv

# Create environment
env = GreenLightEnv()

# Optional sanity check (good practice)
check_env(env)

# Create model
model = PPO(
    policy="MlpPolicy",     # Neural network policy
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

# Train
model.learn(total_timesteps=100_000)

# Save model
model.save("ppo_greenlight")

# Close env
env.close()