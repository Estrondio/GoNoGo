import random
import matplotlib.pyplot as plt
import numpy as np

class Environment:
    def __init__(self):
        self.spout_state = "OFF"
        self.tube_state = "NOGO"

    def update_tube_state(self):
        probabilities = [0.3, 0.5, 0.2]
        self.tube_state = random.choices(["GO", "NOGO", "CHECK"], probabilities)[0]

    def set_spout_state(self):
        self.spout_state = "ON" if self.tube_state == "GO" else "OFF"

class Agent:
    def __init__(self, strategy_fn, action_space=["Lick", "Wait"],
                 learning_rate=0.01, discount_factor=0.4, epsilon=0.1,
                 input_size=1, hidden_size=2, output_size=2,
                 **kwargs):
        self.rewards = 0
        self.timeouts = 0
        self.null = 0
        self.profit = 0
        self.strategy_fn = strategy_fn
        self.trial_count = 0
        self.recent_actions = []
        self.previous_outcomes = []
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.strategy_kwargs = kwargs
        self.q_table = {}

    def get_observation(self, environment):
        return environment.tube_state

    def choose_action(self, environment):
        self.environment = environment
        action = self.strategy_fn(self, environment)
        self.update_recent_actions(action)
        return action

    def update_recent_actions(self, action):
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.strategy_kwargs.get("memory_size", 10):
            self.recent_actions.pop(0)
        self.previous_outcomes.append(self.environment.spout_state)
        if len(self.previous_outcomes) > self.strategy_kwargs.get("memory_size", 10):
            self.previous_outcomes.pop(0)

    def lick(self, environment):
        tube_state = self.get_observation(environment)
        if tube_state == "GO":
            self.rewards += 1
        elif tube_state == "NOGO":
            self.timeouts += 1
        else:
            self.null += 1

    def wait(self):
        self.null += 1

    def calculate_profit(self):
        return self.rewards - self.timeouts

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_q_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            q_values = [self.get_q_value(state, action) for action in self.action_space]
            return self.action_space[np.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        best_next_q = max([self.get_q_value(next_state, a) for a in self.action_space])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * best_next_q)
        self.q_table[(state, action)] = new_q

def random_strategy(agent, environment, lick_probability=0.5):
    return "Lick" if random.uniform(0, 1) < lick_probability else "Wait"

def always_lick_strategy(agent, environment, return_lick=0):
    return "Lick" if return_lick == 0 else "Wait"

def periodic_strategy(agent, environment, consecutive_wait=3, consecutive_lick=2):
    agent.trial_count += 1
    if (agent.trial_count % (consecutive_wait + consecutive_lick)) < consecutive_wait:
        return "Wait"
    else:
        return "Lick"

def plastic_strategy(agent, environment):
    memory_size = agent.strategy_kwargs.get("memory_size")
    if len(agent.recent_actions) == 0:
        return random.choice(["A", "B", "C", "D", "E", "F"])
    elif len(agent.recent_actions) == 2:
        return random.choice(["Lick", "Wait"])
    recentActions = agent.recent_actions
    total_county = sum(agent.recent_actions.count(action) for action in ["Lick", "Wait"])
    lick_probability = agent.recent_actions.count("Lick") / total_county
    return "Lick" if random.uniform(0, 1) > lick_probability else "Wait"

def adaptive_strategy(agent, environment):
    memory_size = agent.strategy_kwargs.get("memory_size", 10)
    if len(agent.previous_outcomes) == 0:
        return random.choice(["A", "B", "C", "D", "E", "F"])
    elif len(agent.recent_actions) == 2:
        return random.choice(["Lick", "Wait"])
    else:
        outcomes = agent.previous_outcomes
        lick_probability = outcomes.count("ON") / len(outcomes)
        return "Lick" if random.uniform(0, 1) < lick_probability else "Wait"

def qlearning_strategy(agent, environment):
    return agent.choose_q_action(environment.tube_state)

class Experiment:
    def __init__(self, num_trials, strategy_fns, values_to_plot, num_simulations, **kwargs):
        self.num_trials = num_trials
        self.strategy_fns = strategy_fns
        self.values_to_plot = values_to_plot
        self.num_simulations = num_simulations
        self.kwargs = kwargs
        self.results = {}

    def run_single_experiment(self, strategy_fn, simulation_index):
        self.environment = Environment()
        self.agent = Agent(strategy_fn, **self.kwargs)
        rewards_over_trials = []
        timeouts_over_trials = []
        null_over_trials = []
        profit_over_trials = []
        self.agent.update_recent_actions(self.agent.choose_action(self.environment))

        for trial in range(1, self.num_trials + 1):
            state = self.environment.tube_state
            action = self.agent.choose_action(self.environment)

            if action == "Lick":
                self.agent.lick(self.environment)
            else:
                self.agent.wait()

            if action == "Lick":
                if self.environment.tube_state == "GO":
                    reward = 1
                elif self.environment.tube_state == "NOGO":
                    reward = -1
                else:
                    reward = 0
            else:
                reward = 0

            self.agent.update_q_value(state, action, reward, self.environment.tube_state)
            self.agent.profit = self.agent.calculate_profit()

            rewards_over_trials.append(self.agent.rewards)
            timeouts_over_trials.append(self.agent.timeouts)
            null_over_trials.append(self.agent.null)
            profit_over_trials.append(self.agent.profit)

            self.environment.update_tube_state()
            self.environment.set_spout_state()

        final_rewards = self.agent.rewards
        final_timeouts = self.agent.timeouts
        final_null = self.agent.null
        final_profit = self.agent.profit

        print(f"Simulation {simulation_index} - Profit: {final_profit}, Rewards: {final_rewards}, Timeouts: {final_timeouts}, Null: {final_null}")

        return {"FinalRewards": final_rewards, "FinalTimeouts": final_timeouts, "FinalNull": final_null, "FinalProfit": final_profit}

    def run_experiment(self):

        strategy_boxplot_data = {}  # Store final profit values for each strategy for boxplot

        for strategy_name, strategy_fn in self.strategy_fns.items():
            print(f"\nRunning Simulations for Strategy: {strategy_name}")
            strategy_results = []
            for simulation_index in range(1, self.num_simulations + 1):
                #print(f"Running Simulation {simulation_index}")
                strategy_result = self.run_single_experiment(strategy_fn, simulation_index)
                strategy_results.append(strategy_result)

            # Extract final profit values for the boxplot
            final_profit_values = [result["FinalProfit"] for result in strategy_results]
            strategy_boxplot_data[strategy_name] = final_profit_values


            final_profit_avg = np.mean([result["FinalProfit"] for result in strategy_results])
            print(f"Final Values average for Strategy {strategy_name}: Profit - {final_profit_avg}")

        # Plot boxplot
        self.plot_boxplot(strategy_boxplot_data)

    def plot_boxplot(self, strategy_boxplot_data):
        plt.figure(figsize=(10, 6))
        boxplot_data = [strategy_boxplot_data[strategy_name] for strategy_name in strategy_boxplot_data.keys()]
        plt.boxplot(boxplot_data, labels=strategy_boxplot_data.keys())
        plt.xlabel("Strategy")
        plt.ylabel("Final Profit Values")
        plt.title("Boxplot of Final Profit Values for Each Strategy")
        plt.show()

# Define the strategies to simulate
strategies_to_simulate = {
    "Always": always_lick_strategy,
    "Random": random_strategy,
    "Plastic": plastic_strategy,
    "Adaptive": adaptive_strategy,
    #"QLearningAgent": qlearning_strategy,
}

values_to_plot = ["Profit"]
num_simulations = 50  # Set the desired number of simulations

experiment = Experiment(num_trials=500, strategy_fns=strategies_to_simulate,
                        values_to_plot=values_to_plot, num_simulations=num_simulations, memory_size=10)
experiment.run_experiment()