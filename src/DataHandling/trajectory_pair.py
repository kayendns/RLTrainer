import numpy as np

from collections import namedtuple
from typing import Any, List, NamedTuple, Tuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'terminated', 'truncated', 'next_state'])
Trajectory = namedtuple('Trajectory', ['initial_conditions', 'transitions'])


class TrajectoryPair():
    def __init__(self, trajectory1: Trajectory, trajectory2: Trajectory, preference=None):
        self._trajectory1 = trajectory1
        self._trajectory2 = trajectory2

        self._preference = preference
        self._skipped = False

    def prefer_video1(self):
        self._preference = 0
    
    def prefer_video2(self):
        self._preference = 1
    
    def prefer_no_video(self):
        self._preference = 0.5
    
    def skip(self):
        self._skipped = True
    
    def unskip(self):
        self._skipped = False

    def undo_preference(self):
        self._preference = None

    @property
    def trajectory1(self):
        return self._trajectory1

    @property
    def trajectory2(self):
        return self._trajectory2

    @property
    def preference(self):
        return self._preference

    @property
    def skipped(self):
        return self._skipped

    def __eq__(self, other):
        if not isinstance(other, TrajectoryPair):
            return NotImplemented

        return (self.trajectory1 == other.trajectory1 and 
                self.trajectory2 == other.trajectory2)
    
    def to_bson(self):
        return {
            'trajectory1': trajectory_to_dict(self._trajectory1),
            'trajectory2': trajectory_to_dict(self._trajectory2),
            'preference': self._preference,
            'skipped': self._skipped
        }
    
def convert_to_list(item):
    if isinstance(item, np.ndarray):
        return item.tolist(), str(item.dtype)
    elif isinstance(item, list):
        return [convert_to_list(subitem) for subitem in item], None
    elif isinstance(item, dict):
        return {key: convert_to_list(val)[0] for key, val in item.items()}, None
    return item, None

def transition_to_dict(transition: Transition) -> dict:
    state_list, state_dtype = convert_to_list(transition.state)
    action_list, action_dtype = convert_to_list(transition.action)
    next_state_list, next_state_dtype = convert_to_list(transition.next_state)
    return {
        'state': state_list,
        'state_dtype': state_dtype,
        'action': action_list,
        'action_dtype': action_dtype,
        'reward': transition.reward,
        'terminated': transition.terminated,
        'truncated': transition.truncated,
        'next_state': next_state_list,
        'next_state_dtype': next_state_dtype
    }

def trajectory_to_dict(trajectory: Trajectory) -> dict:
    transitions_dict = [transition_to_dict(t) for t in trajectory.transitions]
    return {
        'initial_conditions': trajectory.initial_conditions,
        'transitions': transitions_dict
    }
