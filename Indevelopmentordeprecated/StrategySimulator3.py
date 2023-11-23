import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
class Environment:
    def __init__(self):
        # Initialize spout and tube states
        self.tube_state = "NOGO"
        self.spout_state = "OFF"

    def update_tube_state(self):
        # Update tube state probabilistically
        probabilities = [0.3, 0.5, 0.2]
        self.tube_state = random.choices(["GO", "NOGO", "CHECK"], probabilities)[0]

    def set_spout_state(self):
        # Set spout state based on tube state
        self.spout_state = "ON" if self.tube_state == "GO" else "OFF"


class Agent:
    def __init__(self, strategy_fn, action_space=["Lick", "Wait"],
                 learning_rate=0.01, discount_factor=0.4, epsilon=0.1,
                 **kwargs):
        # Initialize tracking variables and strategy function
        self.rewards = 0
        self.timeouts = 0
        self.null = 0
        self.profit = 0
        self.strategy_fn = strategy_fn
        self.trial_count = 0
        self.recent_actions = []
        self.recent_spout_states = []
        self.recent_tube_states = []
        self.winning_heuristic = []  # Store tube states in "ON" outcomes
        self.losing_heuristic = []  # Store tube state when "OFF" outcomes
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Explore vs exploit
        self.strategy_kwargs = kwargs
        self.q_table = {}  # Use a dictionary to store Q-value

    def tube_observation(self, environment):
        # Agent "sees" the state of the tube
        tube_state = environment.tube_state
        self.update_recent_tube_states(tube_state)
        return tube_state
    def update_recent_tube_states(self, tube_state):
        #Update the list of recent tube states
        self.recent_tube_states.append(tube_state)
        if len(self.recent_tube_states) > self.strategy_kwargs.get("memory_size", 10):
            self.recent_tube_states.pop(0)

    def choose_action(self, environment):
        # Agent chooses action based on the provided strategy function
        #self.environment = environment  # Add this line to store the environment
        #self.update_recent_actions("Wait")  # Ensure previous_outcomes is initialized
        action = self.strategy_fn(self, environment)
        self.update_recent_actions(action)
        return action

    def update_recent_actions(self, action):
        # Update the list of recent outcomes
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.strategy_kwargs.get("memory_size", 10):
            self.recent_actions.pop(0)

    def spout_observation(self, environment):
        # Agent "sees" the state of the tube
        spout_state = environment.spout_state
        self.update_recent_spout_states(spout_state)
        return spout_state

    def update_recent_spout_states(self, spout_state):
        # Update the list of recent spout states
        self.recent_spout_states.append(spout_state)
        if len(self.recent_spout_states) > self.strategy_kwargs.get("memory_size", 10):
            self.recent_spout_states.pop(0)

    def lick(self, environment):
        # Lick outcome selection based on its observation of the tube state
        tube_state = self.tube_observation(environment)
        if tube_state == "GO":
            self.rewards += 1
            self.spout_observation(environment)
        elif tube_state == "NOGO":
            self.timeouts += 1
            self.spout_observation(environment)
        else:
            self.null += 1
            self.spout_observation(environment)

    def wait(self, environment):
        # Agent waits action
        tube_state = self.tube_observation(environment)
        if tube_state == "GO":
            self.null += 1
            self.recent_spout_states.append("MISS")
            if len(self.recent_spout_states) > self.strategy_kwargs.get("memory_size", 10):
                self.recent_spout_states.pop(0)
        elif tube_state == "NOGO":
            self.null += 1
            self.recent_spout_states.append("MISS")
            if len(self.recent_spout_states) > self.strategy_kwargs.get("memory_size", 10):
                self.recent_spout_states.pop(0)
        else:
            self.null += 1
            self.recent_spout_states.append("MISS")
            if len(self.recent_spout_states) > self.strategy_kwargs.get("memory_size", 10):
                self.recent_spout_states.pop(0)

    def calculate_profit(self):
        #Subtracts timeouts from rewards.
        return self.rewards - self.timeouts

    def get_q_value(self, state, action):
        # Get the Q-value for a given state-action pair
        return self.q_table.get((state, action), 0.0)

    def choose_q_action(self, state):
        # Epsilon-greedy strategy for exploration
        if random.uniform(0, 1) < self.epsilon:
            print("Randomhit!")
            return random.choice(self.action_space)
        else:
            # Choose action with the highest Q-value
            q_values = [self.get_q_value(state, action) for action in self.action_space]
            #print("QValues for state", state, "are:", q_values)
            #print("Actual action return", self.action_space[np.argmax(q_values)])
            return self.action_space[np.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state):
        # Q-value update using the Bellman equation
        #print("This section is in agent update q. state:", state, "action", action,
                #"reward", reward, "next state", next_state)
        current_q = self.get_q_value(state, action)
        best_next_q = max([self.get_q_value(next_state, a) for a in self.action_space])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * best_next_q)
        self.q_table[(state, action)] = new_q

# Strategies
def random_strategy(agent, environment, lick_probability=0.5):
    return "Lick" if random.uniform(0, 1) < lick_probability else "Wait"

def qlearning_strategy(agent, environment):
    return agent.choose_q_action(environment.tube_state)

class Experiment:
    def __init__(self, num_trials, strategy_fns, values_to_plot, **kwargs):
        # Initialize experiment parameters and strategy functions
        self.num_trials = num_trials
        self.strategy_fns = strategy_fns
        self.values_to_plot = values_to_plot
        self.kwargs = kwargs
        self.results = {}

    def run_single_experiment(self, strategy_fn):
        # Initialize agent and environment
        self.environment = Environment()
        self.agent = Agent(strategy_fn, **self.kwargs)

        # Initialize lists to store tracking variables over trials
        rewards_over_trials = []
        timeouts_over_trials = []
        null_over_trials = []
        profit_over_trials = []

        # Initialize previous_outcomes outside the loop
        #self.agent.update_recent_actions(self.agent.choose_action(self.environment))


        for trial in range(1, self.num_trials + 1):

            # Store the current state before taking any action
            #state = self.environment.tube_state
            state = self.agent.tube_observation(self.environment)

            # Agent action based on the provided strategy function
            action = self.agent.choose_action(self.environment)
            #print(f"Agent Action: {action}")

            if action == "Lick":
                self.agent.lick(self.environment)
            else:
                self.agent.wait(self.environment)

            # Define the reward structure with punishment for licking in "NOGO" state
            if action == "Lick":
                if self.environment.tube_state == "GO":
                    reward = 1
                elif self.environment.tube_state == "NOGO":
                    reward = -1
                else:
                    reward = 0
            else:
                reward = 0

            # Update Q-value
            self.agent.update_q_value(state, action, reward, self.environment.tube_state)
            # Calculate profit and update agent's profit variable
            self.agent.profit = self.agent.calculate_profit()

            # Print information (line)
            print(f"Trial {trial} - Action: {action}, Reward: {reward},"
                  f"Profit: {self.agent.profit}, Tube State: {self.environment.tube_state}")

            # Append tracking variables to lists for plotting
            rewards_over_trials.append(self.agent.rewards)
            timeouts_over_trials.append(self.agent.timeouts)
            null_over_trials.append(self.agent.null)
            profit_over_trials.append(self.agent.profit)

            #print("Recentactions: ",self.agent.recent_actions)
            #print("Recentspouts", self.agent.recent_spout_states )
            #print( "Recenttubes", self.agent.recent_tube_states )

            # Update environment
            self.environment.update_tube_state()
            self.environment.set_spout_state()




        # Store the final values in new variables
        final_rewards = self.agent.rewards
        final_timeouts = self.agent.timeouts
        final_null = self.agent.null
        final_profit = self.agent.profit

        print(f"\nFinal Values - Rewards: {final_rewards}, Timeouts: {final_timeouts}, Null: {final_null}\n")

        return {"Rewards": rewards_over_trials, "Timeouts": timeouts_over_trials, "Null": null_over_trials, "Profit": profit_over_trials,
                "FinalRewards": final_rewards, "FinalTimeouts": final_timeouts, "FinalNull": final_null, "FinalProfit": final_profit}

    def run_experiment(self):
        # Run experiments for each selected strategy
        for strategy_name, strategy_fn in self.strategy_fns.items():
            print(f"\nRunning Experiment for Strategy: {strategy_name}")
            self.results[strategy_name] = self.run_single_experiment(strategy_fn)

        # Plotting
        self.plot_results()
        self.plot_groupedbarchart()

    from scipy.interpolate import interp1d

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        for value in self.values_to_plot:
            for strategy_name, result in self.results.items():
                plt.plot(range(1, self.num_trials + 1), result[value], label=f"{strategy_name} - {value}")

        plt.xlabel("Trial Number")
        plt.ylabel("Accumulated Count")
        plt.title("Accumulation of Tracking Variables Over Trials")
        plt.legend()

        plt.figure(figsize=(10, 6))

        for strategy_name, result in self.results.items():
            # Create a continuous function from the original data for each strategy
            x_original = range(1, self.num_trials + 1)
            y_original = result[value]
            f_original = interp1d(x_original, y_original, kind='linear', fill_value="extrapolate")

            # Plot the continuous function for each strategy
            x_continuous = np.linspace(min(x_original), max(x_original), 100)
            plt.plot(x_continuous, f_original(x_continuous), label=f"{strategy_name} - Continuous Function")

        plt.xlabel("Trial Number")
        plt.ylabel("Accumulated Count")
        plt.title("Continuous Function of Accumulation Over Trials")
        plt.legend()

        plt.figure(figsize=(10, 6))

        for strategy_name, result in self.results.items():
            # Calculate and plot the derivative of the continuous function for each strategy
            derivative = np.gradient(result[value], x_original)
            plt.plot(x_original[:-1], derivative, label=f"{strategy_name} - Derivative")

        plt.xlabel("Trial Number")
        plt.ylabel("Derivative")
        plt.title("Derivative of Accumulation Over Trials")
        plt.legend()

        plt.figure(figsize=(10, 6))

        for strategy_name, result in self.results.items():
            # Calculate and plot the double derivative of the continuous function for each strategy
            double_derivative = np.gradient(np.gradient(result[value], x_original), x_original[:-1])
            plt.plot(x_original[:-2], double_derivative, label=f"{strategy_name} - Double Derivative")

        plt.xlabel("Trial Number")
        plt.ylabel("Double Derivative")
        plt.title("Double Derivative of Accumulation Over Trials")
        plt.legend()

        plt.show()

    def plot_groupedbarchart(self):
        final_rewards = [result["FinalRewards"] for result in self.results.values()]
        final_timeouts = [result["FinalTimeouts"] for result in self.results.values()]
        final_null = [result["FinalNull"] for result in self.results.values()]
        final_profit = [result["FinalProfit"] for result in self.results.values()]

        # Random X-axis positions for better visualization
        x_positions = np.arange(len(final_rewards))

        # Bar width
        bar_width = 0.2

        plt.figure(figsize=(10, 6))
        plt.bar(x_positions - 1.5 * bar_width, final_rewards, bar_width, alpha=0.7, label="Final Rewards")
        plt.bar(x_positions - 0.5 * bar_width, final_timeouts, bar_width, alpha=0.7, label="Final Timeouts")
        plt.bar(x_positions + 0.5 * bar_width, final_null, bar_width, alpha=0.7, label="Final Null")
        plt.bar(x_positions + 1.5 * bar_width, final_profit, bar_width, alpha=0.7, label="Final Profit")

        plt.xlabel("Strategy")
        plt.ylabel("Count")
        plt.title("Final Values for Each Strategy")
        plt.xticks(x_positions, self.results.keys())  # Set X-axis ticks
        plt.legend()
        plt.show()


# Define the strategies to simulate
strategies_to_simulate = {
    "Random": random_strategy, #Takes a random action according to the probability set
    "QLearningAgent": qlearning_strategy,#A q learning agent that learns through reinforcement learning
}
values_to_plot = ["Profit"] # Profit, Reward, Timeouts, Null
# Run the experiment with x trials for selected strategies and values
experiment = Experiment(num_trials=1000, strategy_fns=strategies_to_simulate,
                        values_to_plot=values_to_plot, memory_size=10)
experiment.run_experiment()
