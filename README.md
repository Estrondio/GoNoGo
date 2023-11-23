# 🚦🧠 Go/NoGo Task SIM 3000 🐭🛑
![Banner for the GoNoGo SIM 3000, an anthropomorphic mouse looking staring into a red light](Images/Gonogogithub.png)

Welcome to Go/NoGo Task SIM 3000 (GTS3), a Go/No-Go Task Simulator.

## Overview

GTS3 is a Python-based simulator designed for experimenting with different strategies in a go/no-go task simulation. It allows users to explore the performance of different agents' policies against different environmental contingencies.

## Features

- **Strategy Exploration**: Test and compare various pre-built strategies, from random actions to q-learning agents.
- **Visualization**: Visualize the accumulation of tracking variables over trials. Or visualize how these vary over multiple simulations.
- **Data collection**: Export the results on a trial-by-trial basis into a .csv file. 
- **Flexible Experimentation**: Easily modify the number of trials, sessions, strategy, and track variables.
- **Modularity**: Easily implement your own strategy or learning agent policy/ strategy.

## How does it work?

GTS3 takes instances of a defined environment and an agent and runs the experiment for a selected number of trials:

- **The Environment class**: The environment has two attributes: "Tube state" and "Spout state"; As well as two methods: "Update tube state" and "Set spout state".
   - **Tube state** (tube_state) refers to the Go/NoGo cue. Its states can be "GO", "NOGO", or "CHECK" (absence of cue).
   - **Spout state** (spout_state) refers to the presence of reward. Its states can be "ON" or "OFF".
   - **Update tube state** (update_tube_state) is a method that updates the state of the spout based on the set probabilities.
   - **Set spout state** (set_spout_state) is a method that deterministically sets the state of the spout depending on the state of the tube.
   - ❗**CUSTOMIZE**❗: Modify the "probabilities" list in the "update_tube_state" method of the environment class to set your own environment probabilities.
 
- **The Agent class**: The agent have multiple attributes and methods. Most attributes and methods refer to storing and performing operations with information obtained throughout the session. This information can then be utilized in real-time in the decision-making policy defined in the strategies. Succinctly, depending on the strategy:
   - The agent observes the cue (tube_state). 
   - The agent selects an action based on the strategy (To Lick or to Wait).
   - The agent observes the outcome (spout_state).
   - The agent registers information about the trial.
   -  **Warning** Although the code for the QLearning Agent are methods of the Agent, **the reward structure used by the agent is not defined here**. "Rewards", "Timeouts" and "Null" are simply qualitative tracking variables.
   - ❗**CUSTOMIZE**❗: Modify the parameters learning_rate, discount_factor, epsilon to explore the QLearningAgent strategy.

- **The Experiment class**: 
   
## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/Estrondio/GoNoGo.git
    ```

2. Navigate to the project directory:

    ```bash
    cd GoNoGo
    ```

3. Run the experiment:

    ```bash
    python neuro_simulator.py
    ```

4. Explore the results and enjoy the journey into the world of reinforcement learning!

## Dependencies

- Python 3.x
- Matplotlib
- NumPy

## Contribution

Feel free to contribute by forking the repository and creating pull requests. If you have suggestions, improvements, or find any issues, let's collaborate to make NeuroSimGoNoGo even better!

## License

This project is licensed under the [MIT License](LICENSE).

---

Happy coding and exploring the realms of reinforcement learning!
