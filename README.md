# RL Trading Bot

This project is a Reinforcement Learning (RL) based trading bot for financial markets. It leverages advanced RL algorithms (such as PPO and LSTM-based agents) to train, evaluate, and visualize trading strategies on historical stock data.

## Features

- Multiple RL agent implementations (PPO, PPO-LSTM, Multi-Asset PPO)
- Training, evaluation, and visualization scripts
- Support for multiple assets (AAPL, MSFT, TSLA)
- TensorBoard integration for monitoring training progress
- Modular environment and agent design

## Project Structure

```
.
├── agents/                # RL agent implementations
├── data/                  # Historical stock data (CSV)
├── env/                   # Trading environment code
├── models/                # Saved model files
├── utils/                 # Utility scripts (indicators, data collection)
├── ppo_*_tensorboard/     # TensorBoard logs for different experiments
├── train_*.py             # Training scripts
├── evaluate_*.py          # Evaluation scripts
├── visualize_*.py         # Visualization scripts
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rl-trading-bot.git
   cd rl-trading-bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data:**
   - Place your historical stock data CSV files in the `data/` directory.

## Usage

### Training

- **Train PPO agent:**
  ```bash
  python train_mlp.py
  ```

- **Train PPO-LSTM agent:**
  ```bash
  python train_lstm.py
  ```

- **Train Multi-Asset PPO agent:**
  ```bash
  python train_multi.py
  ```

### Evaluation

- **Evaluate trained agent:**
  ```bash
  python evaluate.py
  ```

- **Evaluate multi-asset agent:**
  ```bash
  python evaluate_multi.py
  ```

### Visualization

- **Visualize results:**
  ```bash
  python visualize.py
  ```

- **Visualize multi-asset results:**
  ```bash
  python visualize_multi.py
  ```

### TensorBoard

- To monitor training progress:
  ```bash
  tensorboard --logdir=ppo_lstm_tensorboard/
  ```

## Notes

- Model files and logs are ignored by git as specified in `.gitignore`.
- The environments and agents are modular and can be extended for other assets or RL algorithms.

## License

[MIT License](LICENSE) (update as appropriate)
