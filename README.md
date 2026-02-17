# David Carrilho - Data Science & AI Portfolio

Welcome to my Data Science & AI portfolio. Here I showcase projects that span Machine Learning, Deep Learning, Reinforcement Learning and Data Visualization from university work to independent projects.

---

### Machine Learning  
Supervised & unsupervised learning projects:  
Classification, Regression, Clustering  
Feature engineering  
Model evaluation & interpretability

---

### Deep Learning  
Neural networks using TensorFlow / PyTorch:  
CNNs for image classification  
RNN/GRU/LSTM for sequences  
Autoencoders & VAEs  
Regularization, optimization, hyperparameter tuning

---

### Reinforcement Learning  
AI agents that learn by interacting with environments:  
Custom 2D grid environments  
DQN agents  
Exploration strategies  
Reward shaping

---

## Projects

### `rl_gridworld`
Q-learning agent trained on a custom 2D GridWorld with traps and a goal state.

**How to run**
- From the repo root:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `python rl_gridworld/train.py`

**Optional flags**
- `--episodes 5000`
- `--epsilon 0.1`
- `--epsilon-min 0.05`
- `--epsilon-decay 0.999`
- `--seed 42`
- `--no-plot` (plot is skipped automatically when `--animate` is used)
- `--output-dir artifacts` (saves `rewards_per_episode.csv` and `learning_curve.png`)
- `--no-save-artifacts`
- `--animate`
- `--animate-episodes 5`
- `--animate-delay 0.03`
- `--animate-hold` (keep animation window open)
- `--backend MacOSX` (force GUI backend if window doesn’t show)

## 📌 About Me  
I am a 3rd-year Data Science student at NOVA IMS, passionate about Machine Learning, Deep Learning, and building intelligent systems. I love experimenting, learning new techniques, and creating projects that combine data, AI, and real-world problems.
