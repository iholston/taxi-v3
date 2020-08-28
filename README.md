# taxi-v3
Simple reinforcement learning solution to [taxi-v3](https://gym.openai.com/envs/Taxi-v3/).

## References
1. [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/), Satwik Kansal and Brendan Martin

## Synopsis
State Space: 500 states  
Action Space: south, north, east, west, pickup, dropoff  
Q-table: simple table mapping 500 states to best action for highest cumulative long-term reward. Updated via q-learning.
