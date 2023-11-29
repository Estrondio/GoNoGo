import random
import math
import matplotlib.pyplot as plt
import numpy as np

class Environment:
    def __init__(self):
        # Initialize spout and tube states
        self.tube_state = "NOGO"
        self.spout_state = "OFF"

    def update_tube_state(self):
        # Update tube state probabilistically
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
            #print("Randomhit!")
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

class Experiment:
    def __init__(self, num_trials, strategy_fns, selected_strategies, values_to_plot, **kwargs):
        # Initialize experiment parameters and strategy functions
        self.num_trials = num_trials
        self.strategy_fns = strategy_fns
        self.values_to_plot = values_to_plot
        self.kwargs = kwargs
        self.selected_strategies = selected_strategies
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

            # Print information (line) - Uncomment to see real-time trial by trial actions
            #print(f"Trial {trial} - Action: {action}, Reward: {reward},"
                  #f"Profit: {self.agent.profit}, Tube State: {self.environment.tube_state}")

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

# Strategies
def random_strategy(agent, environment, lick_probability=0.5):
    """Takes a random action according to a set probability (Default: 50%)"""
    if len(agent.recent_spout_states) == 0:
        return "Wait"
    else:
        #print('Branch1. recent spout states:', agent.recent_spout_states)
        #print("Recent spout states last", agent.recent_tube_states[-1])
        return "Lick" if random.uniform(0, 1) < lick_probability else "Wait"

def always_lick_strategy(agent, environment, return_lick=0):
    """Always perform the same set action (Default: Lick)"""
    return "Lick" if return_lick == 0 else "Wait"

def periodic_strategy(agent, environment, consecutive_wait=3, consecutive_lick=2):
    """Always repeat the same set pattern (Default: 3xWait,2xLick)"""
    agent.trial_count += 1
    if (agent.trial_count % (consecutive_wait + consecutive_lick)) < consecutive_wait:
        return "Wait"
    else:
        return "Lick"

def plastic_strategy(agent, environment):
    """This agent is most likely to take the least frequent recent action"""
    memory_size = agent.strategy_kwargs.get("memory_size")
    if len(agent.recent_actions) < memory_size:
        return random.choice(["Lick", "Wait"])
    else:
        wait_probability = agent.recent_actions.count("Lick") / len(agent.recent_actions)
        return "Wait" if random.uniform(0, 1) < wait_probability else "Lick"

def adaptive_strategy(agent, environment):
    """This agent is more likely to Lick when the recent spout states have been favorable"""
    memory_size = agent.strategy_kwargs.get("memory_size")
    if len(agent.recent_actions) < memory_size:
        return random.choice(["Lick", "Wait"])
    else:
        outcomes = agent.recent_spout_states
        lick_probability = 1 - (outcomes.count("OFF") / len(outcomes))
        #print("Outcomes:", outcomes, "recentspoutstates:", agent.recent_spout_states, " PWait:", lick_probability)
        return "Lick" if random.uniform(0, 1) > lick_probability else "Wait"

def intuitive_strategy(agent, environment):
    """If there was reward in the last x trials they will Lick. (Default: 3)"""
    memory_size = agent.strategy_kwargs.get("memory_size")
    if len(agent.recent_actions) < memory_size:
        return random.choice(["Lick", "Wait"])
    else:
        if any(element == 'ON' for element in agent.recent_spout_states[-3:]):
            return "Lick"
        elif agent.recent_spout_states[-3:].count('OFF') >= 3:
            return "Wait"
        else:
            return random.choice(["Lick", "Wait"])

def stochastic_strategy(agent, environment):
    """Lick probability increases as the trials pass from the last reward"""
    memory_size = agent.strategy_kwargs.get("memory_size")
    if 'ON' in agent.recent_spout_states:
        last_ON = len(agent.recent_spout_states) - 1 - agent.recent_spout_states[::-1].index('ON')
        distance = len(agent.recent_spout_states) - 1 - last_ON
        wait_probability = (10 - distance) / 10
        return "Wait" if random.uniform(0, 1) < wait_probability else "Lick"
    else:
        return random.choice(["Lick", "Wait"])

def jackpot_strategy(agent, environment):
    """After some profit checkpoints, it becomes less likely to lick"""
    if 1 <= agent.profit <=4:
        return "Lick" if random.uniform(0, 1) < (0.5 - (agent.profit / 10)) else "Wait"
    elif 5 <= agent.profit <=99:
        return "Lick" if random.uniform(0, 1) < (0.1 - (agent.profit / 1000)) else "Wait"
    elif agent.profit == 100:
        return "Wait"
    else:
        return random.choice(["Lick", "Wait"])

def exponential_decay_strategy(agent, environment):
    """Less likely to lick as the profit increase"""
    profit = agent.profit
    # Define the base probability of licking
    base_probability = 0.5
    # Define the decay rate
    decay_rate = 0.1
    # Calculate the probability of licking using exponential decay
    lick_probability = base_probability * math.exp(-decay_rate * profit)
    # Make a decision based on the calculated probability
    return "Lick" if random.uniform(0, 1) < lick_probability else "Wait"

def empirical_strategy(agent, environment):
    """Establishes deterministic relations between spout states and tube states"""
    # Start with a random choice for the first trial
    if len(agent.recent_spout_states) == 0:
        return random.choice(["Lick", "Wait"])

    # Update heuristics based on the previous trial's outcome
    if agent.recent_spout_states[-1] == "ON":
        agent.winning_heuristic.append(agent.recent_tube_states[-2])
    elif agent.recent_spout_states[-1] == "OFF":
        agent.losing_heuristic.append(agent.recent_tube_states[-2])

    # Check for action selection based on the current tube state
    if environment.tube_state in agent.winning_heuristic:
        return "Lick"
    elif environment.tube_state in agent.losing_heuristic:
        return "Wait"
    else:
        return random.choice(["Lick", "Wait"])

def qlearning_strategy(agent, environment):
    """A q learning agent that learns through reinforcement learning"""
    return agent.choose_q_action(environment.tube_state)

# Define the strategies to simulate
strategies_to_simulate = {
    "Random": random_strategy, #Takes an random action according to the probability set (Default: 50%)
    "Always Lick": always_lick_strategy, #Always perform the same set action (Default: Lick)
    "Periodic": periodic_strategy,  #Always repeat the same set pattern (Default: 3xWait,2xLick)
    "Plastic": plastic_strategy, #Is most likely to take the least frequent recent action (Memory size)
    "Adaptive": adaptive_strategy, #Is most likely to Lick when the recent states have been favorable (Memory Size)
    "Intuitive": intuitive_strategy, #If there was reward in the last three he will Lick.
    "Stochastic": stochastic_strategy, #Lick probability increases as the time passes from the last reward
    "Jackpot": jackpot_strategy, # After some profit checkpoints becomes more conservative
    "Decay": exponential_decay_strategy, #Less likely to lick as the profit increase
    "Empirical" : empirical_strategy, #Establishes deterministic relations between spout states and tube states
    "QLearning": qlearning_strategy,#A q learning agent that learns through reinforcement learning
}
values_to_plot = ["Profit"] # Profit, Reward, Timeouts, Null
probabilities = [0.3, 0.5, 0.2] #Go, NoGo, Check

def get_user_parameters():
    print("\nWelcome to the Go/NoGo Task SIM 3000! \n")

    choice = input("Enter '1' to set up a new simulation or '2' to run a predefined test simulation:")

    if choice == '2':
        return {
            "num_trials": 100,
            "memory_size": 10,
            "selected_strategies": list(strategies_to_simulate.keys()),  # Use all strategies by default
        }
    elif choice == '1':
        while True:
            # Get user input for tube state probabilities
            check_rate = int(input("Enter the probability (0-100) for a check trial (trials without a stimulus):\n")) / 100
            if check_rate < 0 or check_rate > 1:
                print('Input has to be between 0 and 100!')
                continue
            go_probability = int(input("Enter the probability for a GO cue trial (0-100):\n")) / 100
            if go_probability < 0 or go_probability > 1:
                print('Input has to be between 0 and 100!')
                continue
            nogo_probability = 1 - (check_rate + go_probability)
            probabilities[0] = go_probability
            probabilities[1] = nogo_probability
            probabilities[2] = check_rate
            print("Probabilities selected: \nGO cue:", go_probability*100, "\nNOGO cue:", nogo_probability*100, "\nCHECK (no cue):", check_rate*100)
            # Ask for user confirmation
            confirm = input("Are you ok with these probabilities? (yes/ no): ")
            confirm_lower = confirm.lower()
            if confirm_lower == 'yes':
                break
            elif confirm_lower == 'no':
                continue
            else:
                print("Invalid input. Please enter either 0 or 1.")

        # Display available strategies with comments
        print("Now it is time to select the agent policy. Here is a list of the available strategies:")
        for i, (strategy_name, strategy_fn) in enumerate(strategies_to_simulate.items()):
            comment = strategy_fn.__doc__ if strategy_fn.__doc__ else "No comments available."
            print(f"{i + 1}. {strategy_name} - {comment}")

        while True:
            # Get user-selected strategies
            selected_strategies_indices = input(
                "Enter the numbers of the desired agents/strategies (separated by commas): ")

            try:
                selected_strategies_indices = [int(idx) - 1 for idx in selected_strategies_indices.split(',')]
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                continue

            invalid_indices = [idx for idx in selected_strategies_indices if
                               idx < 0 or idx >= len(strategies_to_simulate)]

            if invalid_indices:
                print("Invalid input")
            else:
                selected_strategies = [list(strategies_to_simulate.keys())[idx] for idx in selected_strategies_indices]
                break

        return {
            "num_trials": int(input("Enter the number of trials: ")),
            "memory_size": int(input("Enter the memory size: ")),
            "selected_strategies": selected_strategies,
        }
    else:
        print("Invalid choice. Please enter either 1 or 2.")
        exit()

# ... (rest of the code)

# Get user parameters
user_parameters = get_user_parameters()

# Create an instance of the Experiment class with the environment
experiment = Experiment(
    strategy_fns={k: v for k, v in strategies_to_simulate.items() if k in user_parameters["selected_strategies"]},
    values_to_plot=values_to_plot,
    **user_parameters
)

# Run the experiment
experiment.run_experiment()