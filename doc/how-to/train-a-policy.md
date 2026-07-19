# Train a Policy

This guide shows you how to train a policy on your ported dataset. Training is framework-native: Rosetta produces the dataset, LeRobot trains on it. Which policies accept which keys: [policy compatibility](../reference/lerobot-data-model.md#policy-feature-compatibility).

## ACT (recommended start)

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --output_dir=outputs/train/act_my_robot \
    --policy.device=cuda \
    --wandb.enable=true
```

## Fine-tune a VLA

Use [PEFT](https://huggingface.co/docs/peft/index)/[LoRA](https://huggingface.co/docs/peft/task_guides/lora_based_methods) for [efficient fine-tuning](https://huggingface.co/docs/lerobot/peft_training):

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=my-org/my-dataset \
    --policy.output_features=null \
    --policy.input_features=null \
    --steps=100000 \
    --batch_size=32 \
    --peft.method_type=LORA \
    --peft.r=64
```

## Multi-GPU

LeRobot supports [multi-GPU training](https://huggingface.co/docs/lerobot/multi_gpu_training) via [Accelerate](https://huggingface.co/docs/accelerate/index):

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    $(which lerobot-train) \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --batch_size=32
```

## Resume

```bash
lerobot-train \
    --config_path=outputs/train/my_run/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

## Upload to the Hub

```bash
huggingface-cli upload my-org/my-policy \
    outputs/train/my_run/checkpoints/last/pretrained_model
```
