import pytest
import random
import string
import time
from src.DataHandling.buffered_queue_manager import BufferedQueueManager
from src.DataHandling.trajectory_pair import Transition, Trajectory, TrajectoryPair

class MockDBManager:
    def get_next_entry(self):
        test_transition = Transition('test_state', 'test_action', 'test_reward', False, False, 'test_next_state')
        test_trajectory = Trajectory('test_data', [test_transition])
        trajectory_pair = TrajectoryPair(test_trajectory, test_trajectory)
        time.sleep(1)
        return trajectory_pair

class MockVideoExtractor:
    def recreate_frames_from_trajectory(self, trajectory):
        pass  # Implement if needed

    def generate_video(self, frames):
        # Return a random video file path
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

@pytest.fixture
def queue_manager():
    db_manager = MockDBManager()
    video_extractor = MockVideoExtractor()
    queue_manager = BufferedQueueManager(db_manager, video_extractor, n=3, sleep_interval=0, daemon=False)
    time.sleep(4)  # Wait for the queue to potentially fill
    yield queue_manager
    queue_manager.close_routine()

def test_queue_filling_to_max_size(queue_manager):
    assert queue_manager.get_queue_size() == queue_manager.get_queue_max()

def test_data_structure_in_queue(queue_manager):
    trajectory, video1, video2 = queue_manager.get_next_entry()
    assert isinstance(trajectory, TrajectoryPair)
    assert isinstance(video1, str)
    assert isinstance(video2, str)

def test_queue_size_after_retrieval(queue_manager):
    original_size = queue_manager.get_queue_size()
    queue_manager.get_next_entry()
    assert queue_manager.get_queue_size() == original_size - 1

def test_data_content(queue_manager):
    trajectory, video1, video2 = queue_manager.get_next_entry()
    test_transition = Transition('test_state', 'test_action', 'test_reward', False, False, 'test_next_state')
    test_trajectory = Trajectory('test_data', [test_transition])
    trajectory_pair = TrajectoryPair(test_trajectory, test_trajectory)
    assert trajectory == trajectory_pair
