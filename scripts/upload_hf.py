#!/usr/bin/env python3
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Cargar dataset local
dataset = LeRobotDataset(
    repo_id="yaisa5ramriez/episodes_test",
    root="/workspace/data/episodes/final23",
)

dataset.push_to_hub(
    repo_id="yaisa5ramriez/episodes_test",
    private=False
)