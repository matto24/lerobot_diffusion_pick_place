

## Start with recording episodes with record.py
uv run python pick_place_lerobot/record.py --repo-id local/ur5e_pick_place_v2 --num-episodes 300

Frames are rendered at 320x240 (IMAGE_HEIGHT/IMAGE_WIDTH in lerobot_dataset.py), the same size
eval_policy.py renders at, so training needs no resize and scripts/downscale_dataset.py is not needed.


## Replay of recorded episodes
uv run lerobot-dataset-viz --repo-id local/ur5e_pick_place_v2 --episode-index 0


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

## Long run
lerobot-train --dataset.repo_id="MattoTT/ur5e_pick_place_v2" --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_ur5e_v2 \
  --batch_size=64 --steps=100000 --num_workers=22 \
  --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.crop_shape="[216, 288]" \
  --save_checkpoint=true --save_freq=20000 \
  --policy.push_to_hub=true --policy.repo_id="MattoTT/diffusion_ur5e_pick_place" --policy.private=true \
  --trackio.enable=true --trackio.space_id="MattoTT/ur5e-tracker" --trackio.private=true \
  --accelerator.mixed_precision=bf16


## Test run
lerobot-train --dataset.repo_id="MattoTT/ur5e_pick_place_v2" --policy.type=diffusion \
  --output_dir=outputs/train/test_run \
  --batch_size=64 --steps=200 --num_workers=22 \
  --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.crop_shape="[216, 288]" \
  --save_checkpoint=true --save_freq=20000 \
  --policy.push_to_hub=false --policy.repo_id="MattoTT/diffusion_ur5e_pick_place" --policy.private=true \
  --accelerator.mixed_precision=bf16

## Local command
lerobot-train --dataset.repo_id="MattoTT/ur5e_pick_place_v2" --policy.type=diffusion \
  --output_dir="outputs/train/diffusion_ur5e_v2" \
  --batch_size=64 --steps=100000 --num_workers=$(( $(nproc) - 2 )) \
  --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.crop_shape="[216, 288]" \
  --policy.device=cuda \
  --save_checkpoint=true --save_freq=10000 \
  --policy.push_to_hub=true --policy.repo_id="MattoTT/diffusion_ur5e_pick_place_v2" --policy.private=true \
  --accelerator.mixed_precision=bf16

