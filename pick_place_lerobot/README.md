

## Start with recording episodes with record.py
uv run python pick_place_lerobot/record.py --repo-id local/ur5e_pick_place --num-episodes 300

Frames are rendered at 320x240 (IMAGE_HEIGHT/IMAGE_WIDTH in lerobot_dataset.py), the same size
eval_policy.py renders at, so training needs no resize and scripts/downscale_dataset.py is not needed.


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
  --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.crop_shape="[216, 288]" \
  --save_checkpoint=true --save_freq=5000 \
  --policy.push_to_hub=true --policy.repo_id="MattoTT/diffusion_ur5e_pick_place" --policy.private=true \
  --trackio.enable=true --trackio.space_id="MattoTT/ur5e-tracker" --trackio.private=true \
  --accelerator.mixed_precision=bf16
