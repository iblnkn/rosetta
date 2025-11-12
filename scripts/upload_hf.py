#!/usr/bin/env python3
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Cargar dataset local
dataset = LeRobotDataset(
    repo_id="yaisa5ramriez/monday_test",
    root="/workspace/data/monday_trimmed_dir/outrosetta2",
)

dataset.push_to_hub(
    repo_id="yaisa5ramriez/monday_test",
    private=False
)