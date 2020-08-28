import gym, random
import numpy as np

# Setup. NOTE with epsilon set to .1, a min of 100,000 training sessions is required or the table won't solve the game
env = gym.make("Taxi-v3").env
env.reset()
train_episodes = 100000
print("Initializing q_table...")

# Hyperparameters
alpha = 0.1 # learning rate
gamma = 0.6 # future reward discount
epsilon = 0.1 # random action rate

# Initialize q_table as 500x6 matrix, could also us numpy to make it simpler. 
q_table = [[0 for i in range(env.action_space.n)] for j in range(env.observation_space.n)]

# Train Brain
print("Table Complete.\n\nTraining " + str(train_episodes) + " times...")
for i in range(1, train_episodes + 1):
    
    # Set up new environment
    state = env.reset()
    steps, penalties, reward = 0, 0, 0
    done = False

    # Solve and update until complete
    while not done:
        
        # Take random action or use learned values
        if random.uniform(0,1) < epsilon: 
            action = env.action_space.sample()
        else:
            action = q_table[state].index(max(q_table[state])) # have to get the index of max value
            
        next_state, reward, done, _ = env.step(action)

        # Update Table
        old_value = q_table[state][action]
        next_max = max(q_table[next_state])

        # Q(state,action) = (1-alpha)Q(state,action) + alpha(reward + gamma(maxQ(next state, all actions)))
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action] = new_value

        if reward == -10:
            penalties += 1

        # Move to new state
        state = next_state
        steps += 1
print("Training complete.\n")

# Test Trained Table
print("Testing trained q-table 100 times...")
total_steps, total_penalties = 0, 0
for i in range(100):
    state = env.reset()
    steps, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = q_table[state].index(max(q_table[state])) # have to get the index of max value
        state, reward, done, _ = env.step(action)

        if reward == -10:
            penalties += 1

        steps += 1

        if steps > 1000:
            print("ERROR: Agent cannot solve game. More training required.")
            break
        
    total_steps += steps
    total_penalties += penalties

print("Testing Complete.\n\nResults after 100 episodes:\n---------------------------")
print("Average steps per playthrough: " + str(total_steps/100))
print("Average penalties per episode: " + str(total_penalties/100))
