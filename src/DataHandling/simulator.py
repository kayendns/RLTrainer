import math
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
from src.DataHandling.trajectory_pair import Transition, Trajectory

''' A class for simulating gym environments and receiving trajectories.
'''
class Simulator():
    
    def __init__(self, env, frame_rate: int = 50, run_speed_factor: float = 1.0):
        self.env = env
        self.frame_rate = frame_rate
        self.run_speed_factor = run_speed_factor

    def simulate_for_n_seconds(self, seconds: int = 10, seed=42) -> Tuple[List[np.ndarray], Trajectory]:
        frames = []
        transitions = []
        observation, info = self.env.reset(seed = seed)
        initial_condition = {'seed': [seed]}
        rng = np.random.default_rng(seed)

        # floor necessary, as run_speed_factor is float 
        max_frames = math.floor(seconds * self.frame_rate * self.run_speed_factor)

        for _ in range(max_frames):
            action = self.env.action_space.sample()
            frames.append(self.env.render())
            current_obs = observation
            observation, reward, terminated, truncated, info = self.env.step(action)

            transitions.append(Transition(current_obs, action, reward, terminated, truncated, observation))

            if terminated or truncated:
                seed = rng.integers(0, 100)
                self.env.reset(seed = seed)
                initial_condition['seed'].append(seed)

        return frames, Trajectory(initial_condition, transitions)

    # Simulate an Episode. Returns the frames of the episode + transitions of the episode.
    #Note: Capturing frames is there so I can check that I capture initial conditions during recreation
    # properly. This will be removed eventually FIXME
    def simulate_episode(self, seed: int = 42, max_time_steps: int = 1000) -> Tuple[List[np.ndarray], Trajectory]:
        frames = []
        transitions = []
        observation, info = self.env.reset(seed=seed)
        initial_condition = {'seed': [seed]}

        for _ in range(max_time_steps):
            frames.append(self.env.render())
            action = self.env.action_space.sample()
            
            current_obs = observation

            observation, reward, terminated, truncated, _ = self.env.step(action)

            transitions.append(Transition(current_obs, action, reward, terminated, truncated, observation))

            if terminated or truncated:
                break
        
        trajectory = Trajectory(initial_conditions, transitions)

        return frames, trajectory
