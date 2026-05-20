import numpy as np
import torch

from original_ts_satfire.data_generator_pred_torch_original import FireDataset


class FireDatasetWithGOESSubdaily(FireDataset):
    def __init__(
        self,
        image_path,
        label_path,
        goes_subdaily_path,
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
        self.goes_subdaily_path = goes_subdaily_path
        self.num_goes_samples = np.load(self.goes_subdaily_path, mmap_mode="r").shape[0]
        if self.num_samples != self.num_goes_samples:
            raise ValueError(
                f"GOES sub-daily samples ({self.num_goes_samples}) do not match VIIRS samples ({self.num_samples}) "
                f"for image_path={image_path} and goes_subdaily_path={goes_subdaily_path}"
            )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        goes_subdaily = self.load_goes(idx)
        if self.clean_invalid:
            goes_subdaily = torch.nan_to_num(goes_subdaily, nan=0.0, posinf=0.0, neginf=0.0)
        sample["goes_subdaily"] = goes_subdaily
        return sample

    def load_goes(self, indices):
        goes_subdaily_chunk = np.load(self.goes_subdaily_path, mmap_mode="r")[indices]
        return torch.squeeze(torch.from_numpy(goes_subdaily_chunk.copy())).float()
