import datetime
import math
from typing import Any, List, Tuple

import cv2
import gymnasium as gym
import numpy as np
import yaml
from src.DataHandling.trajectory_pair import Transition, Trajectory

'''A class for generating videos from trajectories of Gym Environments.
'''
class VideoExtractor():

    def __init__(self, env, frame_rate: int = 50, run_speed_factor: float = 1.0):
        '''
        Assumes that the Gym environment is in render_mode 'rgb_array'. 
        '''
        if env.render_mode != 'rgb_array':
            raise ValueError("Environment render mode must be 'rgb_array'")

        self.env = env
        self.frame_rate = frame_rate
        self.run_speed_factor = run_speed_factor

        try:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                self.trajectories_folder = config.get('trajectories_folder', '/this/path')
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading config file: {e}")
            self.trajectories_folder = '/res/trajectories'

    def generate_video(self, frames: List[np.ndarray], file_name: str = "trajectory", add_timestamp: bool = True) -> str:
        if not frames:
            raise ValueError("The frames list is empty. Cannot generate video.")

        height, width, layers = frames[0].shape

        if layers != 3:
            raise ValueError("Frames are not an RGB-array.")

        file_name = f"{self.trajectories_folder}/{file_name}"

        if add_timestamp:
            # Adds timestamp to chosen file name
            file_name = f"{file_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fps = self.frame_rate*self.run_speed_factor
        video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video.release()

        return file_name

    def recreate_frames_from_trajectory(self, trajectory: Trajectory) -> List[np.ndarray]:
        initial_condition, transitions = trajectory
        seeds = initial_condition['seed']
        frames = []
        self.env.reset(seed=seeds[0])
        ctr = 0

        for state, action, _, terminated, truncated, _ in transitions:
            self.env.step(action) 
            frames.append(self.env.render())

            if terminated or truncated:
                ctr += 1
                seed = seeds[ctr]
                self.env.reset(seed=seed)

        return frames

