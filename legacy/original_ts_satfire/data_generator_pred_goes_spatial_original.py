import numpy as np
import torch

from original_ts_satfire.data_generator_pred_torch_original import FireDataset


class FireDatasetWithGOESSpatial(FireDataset):
    def __init__(
        self,
        image_path,
        label_path,
        goes_spatial_path,
        ts_length=8,
        use_augmentations=False,
        n_channel=8,
        label_sel=0,
        target_is_single_day=False,
    ):
        super().__init__(
            image_path=image_path,
            label_path=label_path,
            ts_length=ts_length,
            use_augmentations=use_augmentations,
            n_channel=n_channel,
            label_sel=label_sel,
            target_is_single_day=target_is_single_day,
        )
        self.goes_spatial_path = goes_spatial_path
        self.num_goes_samples = np.load(self.goes_spatial_path, mmap_mode="r").shape[0]
        if self.num_samples != self.num_goes_samples:
            raise ValueError(
                f"GOES spatial samples ({self.num_goes_samples}) do not match VIIRS samples ({self.num_samples}) "
                f"for image_path={image_path} and goes_spatial_path={goes_spatial_path}"
            )

    def __getitem__(self, idx):
        x, y = self.load_data(idx)
        goes_spatial = self.load_goes(idx)
        if self.clean_invalid:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            goes_spatial = torch.nan_to_num(goes_spatial, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.normalizer(x)
        if self.use_augmentations:
            params = self.sample_augmentation_params()
            x, y = self.augment(x, y, params=params)
            goes_spatial = self.augment_spatial_features(goes_spatial, params)
        x = self.preprocess(x)
        if self.clean_invalid:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            goes_spatial = torch.nan_to_num(goes_spatial, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "data": x,
            "labels": y,
            "goes_spatial": goes_spatial,
        }

    def load_goes(self, indices):
        goes_spatial_chunk = np.load(self.goes_spatial_path, mmap_mode="r")[indices]
        return torch.squeeze(torch.from_numpy(goes_spatial_chunk.copy())).float()
