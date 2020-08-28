# taxi-v3
Simple reinforcement learning solution to [taxi-v3](https://gym.openai.com/envs/Taxi-v3/).

## References
1. [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/), Satwik Kansal and Brendan Martin

## Synopsis
State Space: 500 states  
Action Space: south, north, east, west, pickup, dropoff  
Q-table: simple table mapping 500 states to best action for highest cumulative long-term reward. Updated via q-learning.

Run on Python 3.7 on Windows 10 (ANSI escape code is not recognized so env.render [doesn't work properly](https://stackoverflow.com/questions/51431428/openai-gym-not-rendering-colors-correctly-in-console-environments))
