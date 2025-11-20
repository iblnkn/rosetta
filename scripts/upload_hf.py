#!/usr/bin/env python3
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Cargar dataset local
dataset = LeRobotDataset(
    repo_id="yaisa5ramriez/monday_new",
    root="/workspace/data/test_cams/closest_tes6",
)

dataset.push_to_hub(
    repo_id="yaisa5ramriez/monday_new",
    private=False
)