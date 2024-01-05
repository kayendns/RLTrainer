from src.DataHandling.trajectory_pair import Trajectory, TrajectoryPair
from src.DataHandling.simulator import Simulator
from src.DataHandling.database_manager import DBManager

class VideoStreamer():

    def __init__(self, db_manager: DBManager, simulator: Simulator, max_entries: int = 3):
        self.db_manager = db_manager
        self.simulator = simulator
        self.max_entries = max_entries
        self.stream_env()
        print("Done Streaming.")
    
    def stream_env(self):
        for _ in range(self.max_entries):
            _, trajectory1 = self.simulator.simulate_for_n_seconds(3)
            _, trajectory2 = self.simulator.simulate_for_n_seconds(3)
            print(self.db_manager.add_entry(TrajectoryPair(trajectory1, trajectory2)))