"""Low-dimensional shard caching for the merged RoboTwin v3 datasets."""
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class RobotwinV3Dataset(LeRobotSingleDataset):
    def __init__(self, **kwargs):
        self._frame_files = {}
        super().__init__(**kwargs)

    def get_trajectory_data_lerobot_v3(self, trajectory_id):
        meta = self.trajectory_ids_to_metadata[trajectory_id]
        path = self.dataset_path / self.data_path_pattern.format(
            chunk_index=int(meta['data/chunk_index']), file_index=int(meta['data/file_index']))
        if path not in self._frame_files:
            frame = pd.read_parquet(path)
            self._frame_files[path] = (frame, frame.groupby('episode_index', sort=False).indices)
        frame, groups = self._frame_files[path]
        if trajectory_id not in groups:
            raise ValueError(f'Episode {trajectory_id} missing from {path}')
        return frame.iloc[groups[trajectory_id]].copy()
