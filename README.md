# Reinforcement Learning Trainer

![logo](https://github.com/kayendns/RLTrainer/blob/master/logo.png)

This projects builds upon the [fancy_gym](https://github.com/ALRhub/fancy_gym) fork of OpenAI Gym and aims to provide an easy way for doing Reinforcement Learning from Human Feedback (RLHF) for RL agents.

## Structure

The framework has generally three components:

- **Data Collection** - giving the user a graphical interface to give preferences on trajectories and collecting them into a dataset
- **Reward Function Aggregation** - allowing the user to choose an algorithm for the aggregation of the preference data into a reward function or some other function from which the agent can be trained
- **Training the Agent on the Preferences** - allowing the user to specify how the policy should be extracted from the reward function

The last two steps may be done in one step (e.g. with DPO) or, more generally, follow a different control flow, depending on the chosen learning algorithm!

## Roadmap

### Installation & Setup
- create requirements.txt
- create setup.py
- write installation guide

### Data Collection
- Add Docstring
- Improve the GUI

### Reward Function Aggregation
- Add support for:
    - DPO

### Training the Agent on the Preferences
- Add support for:
    - PPO
