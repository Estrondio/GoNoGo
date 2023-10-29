import random
import matplotlib.pyplot as plt
import numpy as np


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
                 input_size=1, hidden_size=2, output_size=2,
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
            self.rewards += 0
            self.recent_spout_states.append("MISS")
        elif tube_state == "NOGO":
            self.timeouts += 0
            self.recent_spout_states.append("MISS")
        else:
            self.null += 0
            self.recent_spout_states.append("MISS")


    def calculate_profit(self):
        return self.rewards - self.timeouts

    def get_q_value(self, state, action):
        # Get the Q-value for a given state-action pair
        return self.q_table.get((state, action), 0.0)

    #def store_results(self, trial):
        # Store results for each trial
    #    self.rewards_over_trials.append(self.rewards)
    #    self.timeouts_over_trials.append(self.timeouts)
    #    self.null_over_trials.append(self.null)

    def choose_q_action(self, state):
        # Epsilon-greedy strategy for exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            # Choose action with the highest Q-value
            q_values = [self.get_q_value(state, action) for action in self.action_space]
            return self.action_space[np.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state):
        # Q-value update using the Bellman equation
        current_q = self.get_q_value(state, action)
        best_next_q = max([self.get_q_value(next_state, a) for a in self.action_space])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * best_next_q)
        self.q_table[(state, action)] = new_q

# Strategies
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
    #print("Pastchoice1:", agent.recent_actions)
    if len(agent.recent_actions) == 0:
        #print("Pastchoice2:", agent.recent_actions)
        return random.choice(["A", "B", "C", "D", "E", "F"])
    #This is a bandaid to make it work. The way it is working this method is appending the previous random choice
    #twice to the agent.recent_actions. Those values don't even manage to constitute an action. (They're not returned
    #as actions I believe) It seems it is being called once before the task and appending twice Idk
    elif len(agent.recent_actions) == 2:
        #print("Pastchoice3:", agent.recent_actions)
        return random.choice(["Lick", "Wait"])

    recentActions = agent.recent_actions
    total_county = sum(agent.recent_actions.count(action) for action in ["Lick", "Wait"])
    lick_probability = agent.recent_actions.count("Lick") / total_county
    #print("Pastchoices", recentActions, "Number of recent Licks", agent.recent_actions.count("Lick"), "Total actions",
          #total_county, "PLick",  lick_probability)
    return "Lick" if random.uniform(0, 1) > lick_probability else "Wait"


def adaptive_strategy(agent, environment):
    memory_size = agent.strategy_kwargs.get("memory_size", 10)
    #print("Pastchoice1", agent.previous_outcomes)
    if len(agent.previous_outcomes) == 0:
        #print("Pastchoice2", agent.previous_outcomes)
        return random.choice(["A", "B", "C", "D", "E", "F"])
    # This is a bandaid to make it work. The way it is working this method is appending the previous random choice
    # twice to the agent.recent_actions. Those values don't even manage to constitute an action. (They're not returned
    # as actions I believe) It seems it is being called once before the task and appending twice Idk
    elif len(agent.recent_actions) == 2:
        #print("Pastchoice3:", agent.previous_outcomes)
        return random.choice(["Lick", "Wait"])
    else:
# Use the previous outcomes excluding the current trial's outcome
        outcomes = agent.previous_outcomes
        # Calculate lick probability based on past outcomes
        lick_probability = outcomes.count("ON") / len(outcomes) # if len(outcomes) > 0 else 0.0
        #print("Past outcomes", outcomes , "PLick", lick_probability)
        #print(agent.previous_outcomes)
        return "Lick" if random.uniform(0, 1) < lick_probability else "Wait"

def qlearning_strategy(agent, environment):
    return agent.choose_q_action(environment.tube_state)

def bayesian_strategy(agent, environment):
    memory_size = agent.strategy_kwargs.get("memory_size", 10)

    if len(agent.recent_actions) < memory_size:
        return random.choice(["Lick", "Wait"])

    # Use the previous outcomes and actions
    outcomes = agent.previous_outcomes
    print("Outcomes history", outcomes)
    actions = agent.recent_actions
    print("Actions history", actions)
    tube_states = [environment.tube_state]
    print("Tube states history", tube_states)

    # Create a list of tuples representing the combination of tube state, action, and outcome
    combined_data = list(zip(tube_states, actions, outcomes))
    #print("Combined tuple:", combined_data)

    # Count occurrences of each combination
    occurrences = {}
    for combo in combined_data:
        occurrences[combo] = occurrences.get(combo, 0) + 1
        #print("Ocurrences:", occurrences)

    # Calculate conditional probability of reward based on past outcomes, actions, and tube states
    total_rewards = sum(1 for combo, count in occurrences.items() if combo[1] == "Lick" and combo[2] == "GO")
    total_attempts = sum(count for combo, count in occurrences.items() if combo[1] == "Lick")

    # Avoid division by zero
    conditional_probability = total_rewards / total_attempts if total_attempts > 0 else 0.0

    # Cap the probability at 1.0
    conditional_probability = min(conditional_probability, 1.0)

    # Decide whether to lick based on the calculated probability
    return "Lick" if random.uniform(0, 1) < conditional_probability else "Wait"


class Experiment:
    def __init__(self, num_trials, strategy_fns, values_to_plot, **kwargs):
        # Initialize experiment parameters and strategy functions
        self.num_trials = num_trials
        self.strategy_fns = strategy_fns
        self.values_to_plot = values_to_plot
        self.kwargs = kwargs
        self.results = {}
        #self.simulated_strategies = []
        #def get_simulated_strategies(self):
        #return self.simulated_strategies

    #def print_trial_info(self, trial):
     #   print(f"\nTrial {trial}:")
      #  print(f"Tube State: {self.environment.tube_state}")
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
            state = self.agent.tube_observation

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

            print("Recentactions: ",self.agent.recent_actions)
            print("Recentspouts", self.agent.recent_spout_states )
            print( "Recenttubes", self.agent.recent_tube_states )

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

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        for value in self.values_to_plot:
            for strategy_name, result in self.results.items():
                plt.plot(range(1, self.num_trials + 1), result[value], label=f"{strategy_name} - {value}")

        plt.xlabel("Trial Number")
        plt.ylabel("Accumulated Count")
        plt.title("Accumulation of Tracking Variables Over Trials")
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
    "Random": random_strategy,
    "Always Lick": always_lick_strategy,
    #"Periodic": periodic_strategy,
    #"Plastic": plastic_strategy,
    #"Adaptive": adaptive_strategy,
    #"Bayesian" : bayesian_strategy,
    #"QLearningAgent": qlearning_strategy,
}

values_to_plot = ["Profit", "Timeouts"]

# Run the experiment with 50 trials for selected strategies and values
# Run the experiment with the new neural network strategy
experiment = Experiment(num_trials=10, strategy_fns=strategies_to_simulate,
                        values_to_plot=values_to_plot, memory_size=10)
experiment.run_experiment()