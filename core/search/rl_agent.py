"""
Docstring for core.search.rl_agent
State: (awg, par, turns, rpm)

Action: ±1 이동

Reward: margin - penalty
"""
# core/search/rl_agent.py

import random
import numpy as np

class DQNAgent:

    def __init__(self, action_space):
        self.action_space = action_space
        self.epsilon = 0.2

    def act(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        # 간단한 heuristic
        awg, par, turns, rpm = state
        return (awg + random.choice([-1,0,1]),
                par + random.choice([-1,0,1]),
                turns + random.choice([-1,0,1]))

    def get_reward(self, margin_pct, fail_prob):
        if fail_prob > 0.05:
            return -10

        return margin_pct * 2.0
