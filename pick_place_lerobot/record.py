import argparse
import concurrent.futures
import os
from lerobot_dataset import create_dataset
from expert_episode import generate_single_episode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record expert demonstrations for LeRobot dataset in headless MuJoCo."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/ur5e_pick_place",
        help="Repository ID for the dataset (default: 'local/ur5e_pick_place').",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to record (default: 50).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Number of parallel worker processes (default: CPU count - 2).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick and place the object",
        help="Language task instruction string (default: 'pick and place the object').",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Dataset folder (default: $HF_LEROBOT_HOME/<repo-id>).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    NUM_EPISODES = args.num_episodes
    REPO_ID = args.repo_id
    NUM_WORKERS = args.num_workers
    TASK = args.task

    dataset = create_dataset(repo_id=REPO_ID, root=args.root)

    print(
        f"Collecting {NUM_EPISODES} episodes for '{REPO_ID}' using {NUM_WORKERS} parallel workers..."
    )

    saved_episodes = 0
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS)
    try:
        # Each finished episode is ~200 MB of raw frames, and writing it to
        # the dataset is slower than generating it. Submitting everything
        # up front lets finished results pile up in this process until the
        # OOM killer takes it, so keep only a bounded number in flight.
        max_in_flight = NUM_WORKERS + 2
        next_episode = 0
        pending = set()

        def submit_more():
            global next_episode
            while next_episode < NUM_EPISODES and len(pending) < max_in_flight:
                pending.add(executor.submit(generate_single_episode, next_episode))
                next_episode += 1

        submit_more()
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                result = future.result()
                if result is None:
                    continue

                for frame in result["frames"]:
                    dataset.add_frame({**frame, "task": TASK})
                dataset.save_episode()

                saved_episodes += 1
                print(
                    f"Saved episode {saved_episodes}/{NUM_EPISODES} (from worker task {result['episode_id']})"
                )
            # Drop references so the frames can be freed.
            del done
            submit_more()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        # v3.0 buffers metadata and parquet writes; finalize flushes them so the
        # dataset is readable, also after Ctrl-C or a crash mid-recording.
        dataset.finalize()
