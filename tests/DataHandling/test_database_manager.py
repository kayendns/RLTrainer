import pytest
import os
import shutil
import subprocess

from src.DataHandling.database_manager import DBManager
from src.DataHandling.trajectory_pair import Transition, Trajectory, TrajectoryPair

#TODO: test further queue dynamics & edge-cases

@pytest.fixture(scope="class")
def db_manager():
    # Setup code
    mongod_process = subprocess.Popen(
        ["mongod", "--port", "27017", "--dbpath", "tests/data/db"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    db_manager_instance = DBManager()
    yield db_manager_instance

    # Teardown code
    mongod_process.terminate()
    mongod_process.wait()

    db_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'db')
    for filename in os.listdir(db_directory):
        file_path = os.path.join(db_directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def test_basic_add_video_pair(db_manager):
    test_transition = Transition('test_state', 'test_action', 'test_reward', False, False, 'test_next_state')
    test_trajectory = Trajectory('test_data', [test_transition])
    trajectory_pair = TrajectoryPair(test_trajectory, test_trajectory)

    success_message, error = db_manager.add_entry(trajectory_pair)
    assert success_message is not None
    assert not error

    success_message, error = db_manager.fetch_entry(test_trajectory, test_trajectory)
    assert success_message is not None
    assert not error

def test_set_preference(db_manager):
    test_transition = Transition('test_state', 'test_action', 'test_reward', False, False, 'test_next_state')
    test_trajectory = Trajectory('test_data', [test_transition])
    trajectory_pair = TrajectoryPair(test_trajectory, test_trajectory, preference=None)

    db_manager.add_entry(trajectory_pair)
    db_manager.set_preference(trajectory_pair, preference=0.0)

    reconstructed_trajectory_pair, error = db_manager.fetch_entry(test_trajectory, test_trajectory)
    assert (reconstructed_trajectory_pair.preference == 0.0)
    assert error is None

    db_manager.set_preference(trajectory_pair, preference=1.0)

    reconstructed_trajectory_pair, error = db_manager.fetch_entry(test_trajectory, test_trajectory)
    assert (reconstructed_trajectory_pair.preference == 1.0)
    assert error is None


def test_skip_pair(db_manager):
    test_transition = Transition('test_state', 'test_action', 'test_reward', False, False, 'test_next_state')
    test_trajectory = Trajectory('test_data', [test_transition])
    trajectory_pair = TrajectoryPair(test_trajectory, test_trajectory)

    db_manager.add_entry(trajectory_pair)

    success_message, error_message = db_manager.skip_pair(trajectory_pair)
    assert success_message is not None
    assert error_message is None

    reconstructed_trajectory_pair, _ = db_manager.fetch_entry(test_trajectory, test_trajectory)

    assert reconstructed_trajectory_pair.skipped is True
