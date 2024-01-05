import subprocess
import numpy as np
import os
import shutil

from pymongo import MongoClient, errors
from typing import Tuple, Optional
from src.DataHandling.trajectory_pair import Transition, Trajectory, TrajectoryPair

class DBManager:
    """
    Manages database operations for storing and retrieving trajectory pairs and their associated data.

    Attributes:
        client (MongoClient): The MongoDB client for database operations.
        db: The MongoDB database instance.
        collection: The MongoDB collection for storing video data.
        id_of_current_video: The ID of the current video being processed (for sequential access).

    Methods:
        add_entry: Adds a new trajectory pair entry to the database.
        fetch_entry: Retrieves a specific trajectory pair entry from the database.
        set_preference: Updates the preference value of a specific trajectory pair entry.
        skip_pair: Marks a specific trajectory pair entry as skipped.
        get_next_entry: Retrieves the next trajectory pair entry that has not been processed.
    """

    def __init__(self, debug=True):
        """
        Initializes the DBManager with a MongoDB client and sets up the database and collection.
        """
        self.debug = debug
        self.mongod_process = self._start_mongodb()
        self.client = MongoClient("localhost", 27017)
        self.db = self.client['database']
        self.collection = self.db.videos
        self.id_of_current_video = None

    def _start_mongodb(self):
        return subprocess.Popen(
        ["mongod", "--port", "27017", "--dbpath", "data/db"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    def add_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        """
        Adds a new entry to the database based on the provided trajectory pair.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair data to be stored.

        Returns:
            Tuple[Optional[str], Optional[str]]: The inserted ID if successful, and an error message if not.
        """
        try:
            entry = trajectory_pair.to_bson()

            inserted_id = self.collection.insert_one(entry).inserted_id
            return str(inserted_id), None
        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, str(e)

    def fetch_entry(self, trajectory1: Trajectory, trajectory2: Trajectory) -> Tuple[Optional[TrajectoryPair], Optional[str]]:
        """
        Fetches a trajectory pair entry from the database based on given trajectories.

        Args:
            trajectory1 (Trajectory): The first trajectory of the pair.
            trajectory2 (Trajectory): The second trajectory of the pair.

        Returns:
            Tuple[Optional[TrajectoryPair], Optional[str]]: The fetched TrajectoryPair or an error message.
        """
        try:
            query = {"trajectory1": trajectory1, "trajectory2": trajectory2}
            entry = self.collection.find_one(query)

            if entry:
                trajectory_pair = TrajectoryPair(
                    trajectory1=entry["trajectory1"],
                    trajectory2=entry["trajectory2"],
                    preference=entry["preference"],
                )
                if entry["skipped"]:
                    trajectory_pair.skip()
                return trajectory_pair, None
            else:
                return None, "No matching entry found."

        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, str(e)

    def set_preference(self, trajectory_pair: TrajectoryPair, preference: float) -> Tuple[Optional[str], Optional[str]]:
        """
        Updates the preference value of a specific trajectory pair entry.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair to update.
            preference (float): The new preference value.

        Returns:
            Tuple[Optional[str], Optional[str]]: Success message or error message.
        """
        query = {
            "trajectory1": trajectory_pair.trajectory1,
            "trajectory2": trajectory_pair.trajectory2
        }
        new_values = {"$set": {"preference": preference}}

        updated_result = self.collection.update_one(query, new_values)

        if updated_result.matched_count == 0:
            return None, "No entry found matching the criteria."
        elif updated_result.modified_count == 0:
            return None, "Entry found but no update was needed."
        else:
            return "Entry successfully updated.", None

    def skip_pair(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        """
        Marks a specific trajectory pair entry as skipped.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair to mark as skipped.

        Returns:
            Tuple[Optional[str], Optional[str]]: Success message or error message.
        """
        query = {
            "trajectory1": trajectory_pair.trajectory1,
            "trajectory2": trajectory_pair.trajectory2
        }
        new_values = {"$set": {"skipped": True}}

        updated_result = self.collection.update_one(query, new_values)

        if updated_result.matched_count == 0:
            return None, "No entry found matching the criteria."
        elif updated_result.modified_count == 0:
            return None, "Entry found but no update was needed."
        else:
            return "Entry successfully updated.", None

    def get_next_entry(self):
        try:
            base_query = {"preference": None, "skipped": False}

            # If current_id is set, modify the query to start after that ID
            if self.id_of_current_video is not None:
                base_query["_id"] = {"$gt": self.id_of_current_video}

            entry = self.collection.find_one(base_query, sort=[('_id', 1)])

            if entry:
                self.id_of_current_video = entry["_id"]
                trajectory1 = dict_to_trajectory(entry["trajectory1"])
                trajectory2 = dict_to_trajectory(entry["trajectory2"])

                trajectory_pair = TrajectoryPair(
                    trajectory1=trajectory1,
                    trajectory2=trajectory2,
                    preference=entry["preference"]
                )
                return trajectory_pair, None
            else:
                return None, "No more unprocessed entries found."
        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, str(e)

    def close_db(self):
        self.mongod_process.terminate()
        self.mongod_process.wait()
        if self.debug:
            db_directory = os.path.join(os.path.dirname(__file__),'..', '..', 'data', 'db')
            for filename in os.listdir(db_directory):
                file_path = os.path.join(db_directory, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        print("DB Closed")

def dict_to_transition(transition_dict: dict) -> Transition:
    return Transition(
        state=np.array(transition_dict['state'], dtype=transition_dict.get('state_dtype', 'float')),
        action=np.array(transition_dict['action'], dtype=transition_dict.get('action_dtype', 'float')),
        reward=transition_dict['reward'],
        terminated=transition_dict['terminated'],
        truncated=transition_dict['truncated'],
        next_state=np.array(transition_dict['next_state'], dtype=transition_dict.get('next_state_dtype', 'float'))
    )

def dict_to_trajectory(trajectory_dict: dict) -> Trajectory:
    transitions = [dict_to_transition(t) for t in trajectory_dict['transitions']]
    return Trajectory(
        initial_conditions=trajectory_dict['initial_conditions'], 
        transitions=transitions
    )
