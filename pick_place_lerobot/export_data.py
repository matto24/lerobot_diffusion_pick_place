import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a locally recorded dataset to the Hugging Face Hub.")
    parser.add_argument(
        "--local-repo-id",
        default="local/ur5e_pick_place_v3",
        help="Repo id the dataset was recorded under (record.py --repo-id).",
    )
    parser.add_argument(
        "--hub-repo-id",
        default="MattoTT/ur5e_pick_place_v3",
        help="Destination dataset repo on the Hub. Use a new one per dataset version.",
    )
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.local_repo_id)
    # Tell LeRobot the destination repo on Hugging Face Hub
    ds.repo_id = args.hub_repo_id
    ds.push_to_hub(private=True)
    print(f"Pushed {args.local_repo_id} to {args.hub_repo_id} successfully!")
