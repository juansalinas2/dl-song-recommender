import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .indexer import TrackIndexer
from .loader import load_png_resized

class SongAugmentation:
    def __init__(
        self,
        pitch_shift_bins=2,
        time_scale_range=(0.98, 1.02),
        gain_range=(0.98, 1.02),
        noise_std=0.005,
        mask_prob=0.15,
        max_mask_width=8,
        one_second_dropout_prob=0.03,
        one_second_width=22,
        enabled=True,
    ):
        self.pitch_shift_bins = pitch_shift_bins
        self.time_scale_range = time_scale_range
        self.gain_range = gain_range
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.max_mask_width = max_mask_width
        self.one_second_dropout_prob = one_second_dropout_prob
        self.one_second_width = one_second_width
        self.enabled = enabled

    def _time_scale(self, x, scale):
        _, h, w = x.shape
        new_w = max(8, int(round(w * scale)))
        x_scaled = F.interpolate(x.unsqueeze(0), size=(h, new_w), mode="bilinear", align_corners=False)
        x_scaled = F.interpolate(x_scaled, size=(h, w), mode="bilinear", align_corners=False)
        return x_scaled.squeeze(0)

    def _pitch_shift(self, x, shift):
        if shift == 0:
            return x
        shifted = torch.zeros_like(x)
        if shift > 0:
            shifted[:, shift:, :] = x[:, :-shift, :]
        else:
            shifted[:, :shift, :] = x[:, -shift:, :]
        return shifted

    def _time_mask(self, x):
        if random.random() > self.mask_prob:
            return x
        _, _, w = x.shape
        mask_width = random.randint(1, self.max_mask_width)
        start = random.randint(0, max(0, w - mask_width))
        x = x.clone()
        x[:, :, start:start + mask_width] = 0.0
        return x

    def _long_time_dropout(self, x):
        if random.random() > self.one_second_dropout_prob:
            return x
        _, _, w = x.shape
        mask_width = min(self.one_second_width, w)
        start = random.randint(0, max(0, w - mask_width))
        x = x.clone()
        x[:, :, start:start + mask_width] = 0.0
        return x

    def __call__(self, mix, stems):
        if not self.enabled:
            return mix, stems

        shift = random.randint(-self.pitch_shift_bins, self.pitch_shift_bins)
        scale = random.uniform(*self.time_scale_range)
        gain = random.uniform(*self.gain_range)

        # Share the coarse transform across views, then add light per-view perturbations.
        def apply(x):
            x = self._pitch_shift(x, shift)
            x = self._time_scale(x, scale)
            x = x * gain
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x = self._time_mask(x)
            x = self._long_time_dropout(x)
            return x.clamp(0.0, 1.0)

        mix = apply(mix)
        stems = torch.stack([apply(stem) for stem in stems], dim=0)
        return mix, stems


class StemSongDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.transform = transform
        self.indexer = TrackIndexer(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def _load_spec(self, path) -> torch.Tensor:
        array = load_png_resized(str(path), image_size=self.image_size)
        return torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        track_id = row["spotify_id"]
        # TrackIndexer returns the full mix first, then the stems in the checked project order.
        spec_paths = self.indexer.get_spec_png_paths(track_id)

        mix = self._load_spec(spec_paths[0])
        stems = torch.stack([self._load_spec(path) for path in spec_paths[1:]], dim=0)

        if self.transform is not None:
            mix, stems = self.transform(mix, stems)

        return {"track_id": track_id, "mix": mix, "stems": stems}
