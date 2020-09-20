import gym, random

class qtable:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = [[0 for i in range(env.action_space.n)] for j in range(env.observation_space.n)]

    def train(self, episodes):
        # Train for 'episodes' episodes
        print('Training agent over {} episodes...'.format(episodes))
        for i in range(episodes):

            # Run 1 episode
            state = self.env.reset()
            steps, penalties, reward = 0, 0, 0
            done = False
            while not done:
                
                # Choose random action or policy action
                if random.uniform(0,1) < self.epsilon: 
                    action = self.env.action_space.sample()
                else:
                    action = self.qtable[state].index(max(self.qtable[state]))

                # Execute action, observe new state, reward, etc.
                nextstate, reward, done, info = self.env.step(action)
                
                # Update Qtable 
                oldvalue = self.qtable[state][action]
                nextmax = max(self.qtable[nextstate])
                self.qtable[state][action] = (1-self.alpha)*oldvalue + self.alpha*(reward + self.gamma*nextmax)

                # Update steps, penalties, and move to new state
                if reward == -10:
                    penalties += 1
                steps += 1
                state = nextstate
        print('Complete.\n')

    def play(self):
        total_steps, total_penalties = 0, 0
        for i in range(100):
            state = self.env.reset()
            steps, penalties, reward = 0, 0, 0
            done = False

            while not done:
                action = self.qtable[state].index(max(self.qtable[state])) 
                state, reward, done, _ = self.env.step(action)

                if reward == -10:
                    penalties += 1
                steps += 1
                if steps > 1000:
                    print("ERROR: Agent cannot solve game. More training required.")
                    break
            total_steps += steps
            total_penalties += penalties
        print("Results after 100 episodes:\n---------------------------")
        print("Average steps per playthrough: " + str(total_steps/100))
        print("Average penalties per episode: " + str(total_penalties/100))

# Main
env = gym.make('FrozenLake-v0')
alpha = .2
gamma = .9
epsilon = .2 
episodes = 10000

qLearn = qtable(env, alpha, gamma, epsilon)
qLearn.train(episodes)
qLearn.play()
