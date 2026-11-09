

## Start with recording episodes with record.py
uv run python pick_place_lerobot/record.py --repo-id local/ur5e_pick_place --num-episodes 10


## Replay of recorded episodes
uv run lerobot-dataset-viz --repo-id local/ur5e_pick_place --episode-index 0


## Train


1. Provision the A100 GPU session
colab new -s lerobot-train --gpu A100

colab drivemount -s lerobot-train

3. Install LeRobot
colab install -s lerobot-train lerobot

4. Open the persistent console
colab console -s lerobot-train


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN="your_huggingface_token_here"


lerobot-train --dataset.repo_id="MattoTT/ur5e_pick_place" --policy.type=diffusion \
  --output_dir="/content/drive/MyDrive/lerobot_outputs/diffusion_ur5e" \
  --batch_size=64 --steps=30000 --num_workers=$(( $(nproc) - 2 )) \
  --policy.resize_shape="[240, 320]" --policy.crop_ratio=0.9 \
  --save_checkpoint=true --save_freq=5000 \
  --policy.push_to_hub=true --policy.repo_id="MattoTT/diffusion_ur5e_pick_place" --policy.private=true \
  --trackio.enable=true --trackio.space_id="MattoTT/ur5e-tracker" --trackio.private=true \
  --accelerator.mixed_precision=bf16
