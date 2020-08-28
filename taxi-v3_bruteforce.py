import gym
env = gym.make("Taxi-v3").env
env.reset()

steps = 0
penalties, reward = 0, 0
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)

    if reward == -10: # for wrong pickup/dropoffs
        penalties += 1

    steps += 1

print("Bruteforcing taxi-v3...")
print("Starting State: " + str(env.s) + "\n")
print("Total Steps to Completion: " + str(steps))
print("Penalties (Incorrect dropoff/pickup): " + str(penalties))
