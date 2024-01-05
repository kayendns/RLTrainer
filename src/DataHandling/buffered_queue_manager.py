import queue
import threading
import time

from typing import Tuple

from src.DataHandling.trajectory_pair import Trajectory, TrajectoryPair
from src.DataHandling.database_manager import DBManager
from src.DataHandling.video_extractor import VideoExtractor

class BufferedQueueManager():
    """
    Manages a buffered queue that asynchronously retrieves and stores trajectory pairs from a database.

    This class creates a queue that is continuously refilled in a separate thread. It retrieves
    trajectory pairs from a database using a DBManager, generates videos using a VideoExtractor,
    and stores them in the queue for later retrieval.
    """

    def __init__(self, db_manager: DBManager, video_extractor:VideoExtractor, n: int = 10, sleep_interval: int =5, daemon = True):
        """
        Initializes the BufferedQueueManager with a database manager, video extractor, queue size, and sleep interval.

        Args:
            db_manager (DBManager): An instance of DBManager for database operations.
            video_extractor (VideoExtractor): An instance of VideoExtractor to generate videos from trajectories.
            n (int): The maximum size of the buffered queue.
            sleep_interval (int): The interval between queue refills in seconds.
        """        
        self.buffered_queue = queue.Queue(maxsize=n)
        self.db_manager: DBManager = db_manager
        self.video_extractor: VideoExtractor = video_extractor

        self.sleep_interval = sleep_interval
        self.run = True
        self.refilling_thread = threading.Thread(target=self.refill_loop)
        self.refilling_thread.daemon = daemon
        self.refilling_thread.start()

    def _generate_videos_from_trajectory_pair(self, trajectory_pair: TrajectoryPair) -> Tuple[str, str]:
        """
        Generates video file paths from a given trajectory pair.

        Args:
            trajectory_pair (TrajectoryPair): A pair of trajectories for which to generate videos.

        Returns:
            Tuple[str, str]: A tuple containing the file paths of the generated videos.
        """
        frames1 = self.video_extractor.recreate_frames_from_trajectory(trajectory_pair.trajectory1)
        frames2 = self.video_extractor.recreate_frames_from_trajectory(trajectory_pair.trajectory2)

        video_file_path_1 = self.video_extractor.generate_video(frames1)
        video_file_path_2 = self.video_extractor.generate_video(frames2)

        return video_file_path_1, video_file_path_2
    
    def refill_loop(self):
        """
        Continuously refills the queue with new entries. Runs as a separate thread.
        """
        while self.run:
            if self.get_queue_size() < self.get_queue_max():
                new_entry, error = self.db_manager.get_next_entry()
                print(error)
                if new_entry is not None:
                    video1, video2 = self._generate_videos_from_trajectory_pair(new_entry)
                    print(f"New entry: {new_entry} with videos {video1} and {video2}")
                    self.buffered_queue.put((new_entry, video1, video2))
                else:
                    print("No entry yet available.")
                    time.sleep(self.sleep_interval)
            time.sleep(self.sleep_interval)       

    def get_next_entry(self) -> Tuple[Trajectory, str, str]:
        """
        Retrieves the next entry from the queue.

        Returns:
            Tuple(Trajectory, str, str): The next entry of the queue.
        """
        return self.buffered_queue.get(block=True)

    def get_queue_size(self) -> int:
        """
        Returns the current size of the queue.

        Returns:
            int: The number of items currently in the queue.
        """
        return self.buffered_queue.qsize()
    
    def get_queue_max(self) -> int:
        """
        Returns the maximum size of the queue.

        Returns:
            int: The maximum capacity of the queue.
        """
        return self.buffered_queue.maxsize
    
    def close_routine(self):
        self.run = False
        print("Closing queue")
        self.refilling_thread.join()  # Wait for the thread to finish
