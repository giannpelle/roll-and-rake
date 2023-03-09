# Evaluation of Deep Learning techniques for Roll & Write board games
The code in this repository was developed as part of my thesis for the Master's degree program in Computational Data Science at unibz.
A complete description of the work I performed can be found here.

## Abstract
The majority of game AI research has concentrated on video games and classical board games, ignoring modern board games with innovative mechanics and themes. Additionally, most of the previous studies have been conducted on powerful machines, with training periods of several weeks or months. Consequently, the aim of this study is to evaluate and compare the performance of state-of-the-art reinforcement learning algorithms on modern board games with a training time of less than one week. The game chosen for analysis is the Roll & Write board game Rake&Roll, which was first developed in OpenAI Gym in three variants: the complete version, the deterministic version and the simplified version. After that, DuelingDoubleDeepQNetwork with experience replay (PerD3QN) and Monte Carlo Policy Gradient (REINFORCE) algorithms were implemented in order to train the game AI agents. The REINFORCE agent was unable to learn anything and achieved a similar result as the random agent. Conversely, the PerD3QN agent achieved a significant score, even though it was not optimal. These results were consistent across all game variants introduced. In conclusion, the performance of the PerD3QN agent has demonstrated that, even on commercial machines, it is possible to train AI agents for modern board games within a limited period of time.

## Environment: Rake&Roll
An example of the game can be seen in the screenshot below.

![rollandrake](https://github.com/giannpelle/roll-and-rake/blob/main/img/rake-and-roll.png)

# Techniques developed
The implementation of the Monte Carlo Policy Gradient algorithm (REINFORCE) is available [here](https://github.com/giannpelle/roll-and-rake/blob/main/train_reinforce_v0_zero_two.py).
The implementation of the DoubleDuelingDeepQNetwork with prioritized experience replay (PerD3QN) is available [here](https://github.com/giannpelle/roll-and-rake/blob/main/DuelingPerDoubleDQN_agent_v0_zero_two.py).

## Results
### REINFORCE (gamma 0.2) agent on Rake&Roll-Complete
![results](https://github.com/giannpelle/roll-and-rake/blob/main/img/reinforce_v0_two.png)

### PerD3QN (gamma 0.2) agent on Rake&Roll-Complete
![results](https://github.com/giannpelle/roll-and-rake/blob/main/img/d3qn_v0_two.png)

### Agents comparison on Rake&Roll-Complete
![results](https://github.com/giannpelle/roll-and-rake/blob/main/img/env_v0_run.png)

## Installation

```bash
cd roll-and-rake
conda env create -f environment.yml
conda activate roll-and-rake

python3 play_manual.py
```
