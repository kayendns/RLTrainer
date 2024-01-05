import sys
import os
import numpy as np
import shutil
import yaml
import time
import traceback
import subprocess
import fancy_gym
import gymnasium as gym

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from src.DataHandling.database_manager import DBManager
from src.DataHandling.trajectory_pair import Transition, Trajectory, TrajectoryPair
from src.DataHandling.simulator import Simulator
from src.DataHandling.buffered_queue_manager import BufferedQueueManager
from src.DataHandling.video_extractor import VideoExtractor
from src.DataHandling.video_streamer import VideoStreamer


env = gym.make('fancy/AirHockey-3dof-hit-v0', render_mode="rgb_array", width=600, height=400)

video_extractor = VideoExtractor(env)
db_manager = DBManager()
simulator = Simulator(env)
video_streamer = VideoStreamer(db_manager=db_manager, simulator=simulator, max_entries=3)
time.sleep(5)
buffered_queue = BufferedQueueManager(db_manager, video_extractor, n=10)

current_entry = None

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    config_path = 'config.yaml'
    default_config = {'allowTies': 'off', 'allowSkipping': 'off', 'allowEditing': 'off'}

    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or default_config
    else:
        config = default_config

    config_status = {
        'allowTies': config.get('allowTies', 'off') == 'on',
        'allowSkipping': config.get('allowSkipping', 'off') == 'on',
        'allowEditing': config.get('allowEditing', 'off') == 'on'
    }

    return render_template('index.html', config=config_status)

@app.route('/next-stage-url')
def next_stage():
    config_path = 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config_status = {
        'allowTies': config.get('allowTies', 'off') == 'on',
        'allowSkipping': config.get('allowSkipping', 'off') == 'on',
        'allowEditing': config.get('allowEditing', 'off') == 'on'
    }
    return render_template('PreferenceSelection.html', config=config_status)

@app.route('/get_current_video_pair')
def get_next_video_pair():
    global current_entry
    try:
        current_entry = buffered_queue.get_next_entry()
        _, video_file_path_1, video_file_path_2 = current_entry
        print(current_entry)

        return jsonify({
            "video1": video_file_path_1, 
            "video2":video_file_path_2
            }), 201
    except IndexError:
        return jsonify({"message": "The queue is empty"}), 500

@app.route('/update_preference', methods=['POST'])
def update_preference():
    global current_entry

    preference = request.form.get('preference')

    trajectory_pair, _, _ = current_entry
    db_manager.set_preference(trajectory_pair, int(preference))
    return jsonify({"success": True, "message": "Preference updated successfully."}), 200

@app.route('/skip_video_pair')
def skip_pair():
    global current_entry
    try: 
        trajectory_pair, _, _ = current_entry
        db_manager.skip_pair(trajectory_pair)
        return jsonify({"success": True, "message": "Video pair skipped successfully."}), 200
    except Exception as e:
        print(f"Error skipping video pair: {e}")
        return jsonify({"success": False, "message": "Failed to skip video pair."}), 500

@app.route('/update-config', methods=['POST'])
def update_config():
    try:
        expected_keys = ['allowTies', 'allowSkipping', 'allowEditing'] # More to come
        config_data = {key: (request.form.get(key) or 'off') for key in expected_keys}
        config_path = 'config.yaml'

        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file) or {}
        else:
            config = {}

        for key in expected_keys:
            config[key] = 'on' if config_data[key] == 'on' else 'off'

        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        return {'message': 'Config updated'}, 200

    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True)
