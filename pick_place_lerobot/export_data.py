if __name__ == "__main__":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/ur5e_pick_place")
    # Tell LeRobot the destination repo on Hugging Face Hub
    ds.repo_id = "MattoTT/ur5e_pick_place"
    ds.push_to_hub(private=True)
    print("Pushed successfully!")
