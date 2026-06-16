# 🎮 Super Mario Bros Deep Reinforcement Learning Agent

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)
![Reinforcement Learning](https://img.shields.io/badge/RL-DQN-green.svg)
![Gym](https://img.shields.io/badge/Environment-OpenAI%20Gym-blue.svg)

An autonomous AI agent trained using Deep Reinforcement Learning to navigate and play Super Mario Bros. The agent learns optimal movement strategies and decision-making strictly from raw pixel inputs and dynamic environmental rewards.

## 🧠 Technical Architecture
The system utilizes a **Deep Q-Network (DQN)** coupled with a **Convolutional Neural Network (CNN)** front-end:
* **Visual Processing (CNN):** Extracts spatial features directly from the game screen, allowing the agent to "see" obstacles, enemies, and platforms.
* **State Representation:** Temporal dependencies are captured by stacking multiple consecutive frames together, giving the agent a sense of motion and velocity.
* **Decision Making (RL):** The DQN algorithm maps the extracted visual states to discrete controller actions (e.g., Move Right, Jump) by maximizing the expected cumulative future rewards.

## 📂 Core Files
* `agent.py`: Defines the CNN architecture, action-selection policies, and memory replay buffer.
* `wrappers.py`: Preprocesses the raw game frames (Grayscale conversion, frame resizing to 84x84, and frame stacking).
* `train.py`: Implements the main training loop, optimization steps, and logs performance metrics.
* `play.py`: Loads the trained model weights to execute and visualize the agent's gameplay in real-time.

## 🚀 Quick Start

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
