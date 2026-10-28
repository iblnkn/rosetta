# Train

Training is LeRobot's job. Rosetta doesn't add anything here.

```bash
lerobot-train \
    --dataset.repo_id=my_org/my_dataset \
    --dataset.root=datasets/lerobot/my_org/my_dataset \
    --policy.type=act \
    --output_dir=outputs/train/my_policy \
    --policy.device=cuda
```

Drop `--dataset.root` to train from a dataset on the Hugging Face Hub. The
policy runner loads the checkpoint at
`outputs/train/my_policy/checkpoints/last/pretrained_model`.

The [LeRobot training guide](https://huggingface.co/docs/lerobot/il_robots#train-a-policy)
covers policies, resuming, multi-GPU and pushing checkpoints to the Hub.
