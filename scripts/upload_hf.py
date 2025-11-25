#!/usr/bin/env python3
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Cargar dataset local
dataset = LeRobotDataset(
    repo_id="yaisa5ramriez/episodes_test_2actions_final",
    root="/workspace/data/episodes/final_actions",
)

dataset.push_to_hub(
    repo_id="yaisa5ramriez/episodes_test_2actions_final",
    private=False
)