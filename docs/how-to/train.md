# Train a policy

Train a policy on a ported dataset with LeRobot's `lerobot-train`. Rosetta
doesn't add anything here: a ported dataset is a plain LeRobot dataset.

You need a dataset from [Prepare a dataset](prepare-a-dataset.md) and LeRobot
installed with its `training` extra (see [Installation](../installation.md)).

## Train from a local dataset

```bash
lerobot-train \
    --dataset.repo_id=my_org/my_dataset \
    --dataset.root=datasets/lerobot/my_org/my_dataset \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/my_policy
```

- `--dataset.root` is the dataset directory itself, which is `rosetta_port`'s
  `--root` joined with `--repo-id`. Drop it to train from a dataset on the
  Hugging Face Hub.
- `--policy.type` is the architecture. The checkpoint records it, so
  deploying doesn't ask for it again.
- `--policy.push_to_hub=false` keeps the checkpoint local. LeRobot pushes by
  default, and without `--policy.repo_id` it stops with
  `ValueError: 'repo_id' argument missing. Please specify it to push the model to the hub.`
  To push, replace this flag with `--policy.repo_id=<hf_user>/<model>`.
- `--output_dir` must not exist yet, unless you resume with `--resume=true`
  and `--config_path` pointing at a checkpoint's `train_config.json`.

ACT and Diffusion need an `observation.images.*` key or
`observation.environment_state` in the dataset. On a dataset with only
`observation.state`, training stops with
`ValueError: You must provide at least one image or the environment state among the inputs.`

## Find the checkpoint

Each save writes a numbered directory under `outputs/train/my_policy/checkpoints/`,
and `last` links to the newest one. Deploy this path:

```text
outputs/train/my_policy/checkpoints/last/pretrained_model
```

`train_config.json` in that directory records the dataset the policy was
trained on.

The [LeRobot training guide](https://huggingface.co/docs/lerobot/il_robots#train-a-policy)
covers the policy options, resuming, Weights and Biases logging, and uploading
checkpoints to the Hub.
