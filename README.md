# 🌱 GreenLight-RL  
GreenLight-RL is an advanced reinforcement learning–based greenhouse control system designed to optimize crop growth, energy consumption, and economic profit.

The system simulates realistic greenhouse dynamics including weather conditions, thermal inertia, crop physiology, and dynamic energy pricing.  
It applies Proximal Policy Optimization (PPO) to learn optimal climate control strategies.

---

🚀 Features

Environment Module
- Weather simulation (day–night cycles)
- Solar radiation modeling
- Thermal inertia dynamics
- CO₂ concentration control
- Humidity regulation
- Dynamic energy pricing

Crop Growth Module
- Biomass accumulation model
- Temperature & CO₂ efficiency factors
- Crop stress & health tracking
- Growth-based revenue modeling

Reinforcement Learning Module
- PPO algorithm implementation
- Multi-objective reward function
- Profit-based optimization
- Constraint penalties
- Observation normalization

Evaluation Module
- Performance tracking
- Final biomass measurement
- Energy cost analysis
- Reward monitoring

---

🧠 Reinforcement Learning Algorithm
- PPO (Proximal Policy Optimization)
- Continuous action space control
- Multi-variable optimization

---

🧑‍💻 Tech Stack

Language: Python  
RL Framework: Stable-Baselines3  
Environment: Gymnasium  
Deep Learning: PyTorch  
Numerical Computing: NumPy  

---

📂 Project Structure

greenlight-rl/
│
├── greenlight_env.py        # Greenhouse simulation environment
├── train.py                 # PPO training script
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── saved_models/            # Trained RL models

---

🎯 Objective

To design an intelligent greenhouse control system that:

- Maximizes crop yield
- Minimizes energy consumption
- Optimizes economic profit
- Maintains safe climate conditions
- Learns adaptive control policies using reinforcement learning

---

⚙️ Installation

pip install -r requirements.txt

---

🚀 Run Training

python train.py

---

📊 Future Enhancements

- Real meteorological dataset integration
- SAC / TD3 algorithm comparison
- LSTM-based recurrent policies
- Visualization dashboard
- Cloud deployment

---

📌 Author

Your Name  
Computer Science / AI Project  
