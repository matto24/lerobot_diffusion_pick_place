

## Start with recording episodes with record.py
uv run python pick_place_lerobot/record.py --repo-id local/ur5e_pick_place --num-episodes 10


## Replay of recorded episodes
uv run python -m lerobot.scripts.visualize_dataset --repo-id local/ur5e_pick_place --episode-index 0
