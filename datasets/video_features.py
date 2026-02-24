"""
Video Feature Dataset for CoOp prompt learning.

Loads pre-extracted .npy video features from class-organized directories.
Supports strict normal snippet filtering following ETRIFeatureDataset pattern.

Directory structure:
    {feature_dir}/
        {class_name}/
            {class_name}_{id}_x264.npy   # shape: [T, D]
            ...
        ...

Each .npy file contains per-frame features [T, D] for one video.
Frames are treated as individual samples via sliding windows.
"""

import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _time_to_seconds(time_str) -> float:
    """Convert time string (HH:MM:SS or MM:SS or seconds) to float seconds."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    time_str = str(time_str).strip()
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(time_str)


def _merge_events(events: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or adjacent event intervals."""
    if len(events) <= 1:
        return events
    sorted_events = sorted(events, key=lambda x: x[0])
    merged = [sorted_events[0]]
    for start, end in sorted_events[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class VideoFeatureDataset(Dataset):
    """Dataset for pre-extracted video features with strict normal snippet filtering.

    Args:
        feature_dir: Path to feature directory with class subdirectories.
        annotation_dir: Path to annotation CSV files (optional).
            Each CSV should have columns: file_name, start_time, end_time.
            If None, all frames in each class directory are labeled as that class.
        normal_class: Name of the normal class directory (case-insensitive).
            Used for strict normal filtering logic.
        unit_duration: Window size in seconds (number of frames per sample).
            Only used when use_video_level_pooling=False.
        overlap_ratio: Sliding window overlap ratio (0.0 to <1.0).
            Only used when use_video_level_pooling=False.
        strict_normal_sampling: If True, discard post-event untagged windows
            in non-normal class videos. Only used when use_video_level_pooling=False.
        use_video_level_pooling: If True, use mean pooling to aggregate each video
            [T, D] -> [D] as a single sample. If False, use sliding windows.
        max_files_per_class: Limit number of files per class (for balancing).
        verbose: Print dataset statistics.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        feature_dir: str,
        annotation_dir: Optional[str] = None,
        normal_class: str = "normal",
        unit_duration: int = 1,
        overlap_ratio: float = 0.0,
        strict_normal_sampling: bool = True,
        use_video_level_pooling: bool = False,
        max_files_per_class: Optional[int] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.feature_dir = feature_dir
        self.annotation_dir = annotation_dir
        self.normal_class = normal_class.lower()
        self.unit_duration = unit_duration
        self.overlap_ratio = overlap_ratio
        self.strict_normal_sampling = strict_normal_sampling
        self.use_video_level_pooling = use_video_level_pooling
        self.max_files_per_class = max_files_per_class
        self.verbose = verbose
        self.seed = seed

        self.samples: List[Dict] = []
        self.classnames: List[str] = []
        self.class_to_label: Dict[str, int] = {}
        self.annotations: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        self._discarded_post_event = 0
        self._total_windows = 0

        random.seed(seed)
        np.random.seed(seed)

        # Build dataset
        self._scan_classes()
        if annotation_dir is not None:
            self._load_annotations()
        self._build_samples()

        if verbose:
            self._print_stats()

    def _scan_classes(self):
        """Scan class directories and build class-to-label mapping."""
        class_dirs = sorted([
            d for d in os.listdir(self.feature_dir)
            if os.path.isdir(os.path.join(self.feature_dir, d))
        ])
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.feature_dir}")

        self.classnames = [d.replace("_", " ") for d in class_dirs]
        self.class_to_label = {d: i for i, d in enumerate(class_dirs)}

        if self.verbose:
            print(f"Found {len(class_dirs)} classes: {class_dirs}")

    def _load_annotations(self):
        """Load annotations from CSV files in annotation_dir."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for annotation loading: pip install pandas")

        if not os.path.isdir(self.annotation_dir):
            print(f"Warning: Annotation dir not found: {self.annotation_dir}")
            return

        for csv_file in sorted(os.listdir(self.annotation_dir)):
            if not csv_file.endswith(".csv"):
                continue
            csv_path = os.path.join(self.annotation_dir, csv_file)
            try:
                df = pd.read_csv(csv_path, on_bad_lines="skip")
            except Exception as e:
                print(f"Warning: Failed to read {csv_path}: {e}")
                continue

            for _, row in df.iterrows():
                file_name = str(row.get("file_name", row.get("filename", "")))
                file_stem = os.path.splitext(file_name)[0]
                if not file_stem:
                    continue

                start_time = _time_to_seconds(row.get("start_time", 0))
                end_time = _time_to_seconds(row.get("end_time", 0))

                if end_time > start_time:
                    self.annotations[file_stem].append((start_time, end_time))

        # Merge overlapping events
        for file_stem in self.annotations:
            self.annotations[file_stem] = _merge_events(self.annotations[file_stem])

        if self.verbose:
            print(f"Loaded annotations for {len(self.annotations)} files")

    def _build_samples(self):
        """Build sample list from feature files.

        If use_video_level_pooling=True, creates one sample per video (no sliding windows).
        Otherwise, creates sliding window samples.
        """
        if self.use_video_level_pooling:
            # Video-level pooling: one sample per .npy file
            for class_dir, label in self.class_to_label.items():
                class_path = os.path.join(self.feature_dir, class_dir)

                npy_files = sorted([
                    f for f in os.listdir(class_path)
                    if f.endswith(".npy")
                ])

                if self.max_files_per_class is not None and len(npy_files) > self.max_files_per_class:
                    npy_files = random.sample(npy_files, self.max_files_per_class)

                for npy_file in npy_files:
                    npy_path = os.path.join(class_path, npy_file)
                    file_stem = os.path.splitext(npy_file)[0]

                    self.samples.append({
                        "npy_path": npy_path,
                        "label": label,
                        "video_id": file_stem,
                        "pool_video": True,  # Flag to indicate video-level pooling
                    })
        else:
            # Sliding window approach (original)
            stride = max(1, int(self.unit_duration * (1.0 - self.overlap_ratio)))

            for class_dir, label in self.class_to_label.items():
                class_path = os.path.join(self.feature_dir, class_dir)
                is_normal_class = class_dir.lower() == self.normal_class

                npy_files = sorted([
                    f for f in os.listdir(class_path)
                    if f.endswith(".npy")
                ])

                if self.max_files_per_class is not None and len(npy_files) > self.max_files_per_class:
                    npy_files = random.sample(npy_files, self.max_files_per_class)

                for npy_file in npy_files:
                    npy_path = os.path.join(class_path, npy_file)
                    self._process_npy_feature(
                        npy_path, label, class_dir, is_normal_class, stride
                    )

    def _process_npy_feature(
        self,
        npy_path: str,
        label: int,
        class_dir: str,
        is_normal_class: bool,
        stride: int,
    ):
        """Process a single .npy feature file into sliding window samples."""
        file_stem = os.path.splitext(os.path.basename(npy_path))[0]

        # Load shape only (memory-mapped)
        feat = np.load(npy_path, mmap_mode="r")
        total_seconds = feat.shape[0]

        if total_seconds < self.unit_duration:
            return

        # Look up annotations
        events = self.annotations.get(file_stem, [])
        has_annotations = self.annotation_dir is not None and len(self.annotations) > 0

        # Get earliest event start for strict filtering
        if events:
            events = _merge_events(events)
            earliest_event_start = min(e[0] for e in events)
        else:
            earliest_event_start = float("inf")

        # Create sliding windows
        num_windows = max(0, (total_seconds - self.unit_duration) // stride + 1)

        for i in range(num_windows):
            start_sec = i * stride
            end_sec = start_sec + self.unit_duration
            self._total_windows += 1

            # Determine sample label
            if has_annotations and not is_normal_class and events:
                # Check if window overlaps with any event
                overlaps_event = False
                for event_start, event_end in events:
                    if start_sec < event_end and end_sec > event_start:
                        overlaps_event = True
                        break

                if not overlaps_event:
                    # Strict normal filtering: discard post-event normal windows
                    if self.strict_normal_sampling and end_sec > earliest_event_start:
                        self._discarded_post_event += 1
                        continue
                    # Non-overlapping windows before events keep the class label
                    # (they are pre-event context from the same action class)

            self.samples.append({
                "npy_path": npy_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
                "video_id": file_stem,
            })

    def _print_stats(self):
        """Print dataset statistics."""
        print(f"\n{'='*50}")
        print(f"VideoFeatureDataset Statistics")
        print(f"{'='*50}")
        print(f"Feature dir: {self.feature_dir}")

        if self.use_video_level_pooling:
            print(f"Mode: Video-level mean pooling (one sample per video)")
            print(f"Total videos: {len(self.samples)}")
        else:
            print(f"Unit duration: {self.unit_duration}s, Overlap: {self.overlap_ratio}")
            print(f"Strict normal sampling: {self.strict_normal_sampling}")
            print(f"Total windows created: {self._total_windows}")
            print(f"Discarded post-event windows: {self._discarded_post_event}")
            print(f"Final samples: {len(self.samples)}")

        # Per-class distribution
        label_counts = defaultdict(int)
        for s in self.samples:
            label_counts[s["label"]] += 1
        for label in sorted(label_counts.keys()):
            cname = self.classnames[label] if label < len(self.classnames) else f"label_{label}"
            print(f"  {cname}: {label_counts[label]} samples")
        print(f"{'='*50}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]

        feat = np.load(sample["npy_path"], mmap_mode="r")

        if sample.get("pool_video", False):
            # Video-level pooling: mean across temporal dimension [T, D] -> [D]
            feature_vector = np.mean(feat, axis=0)  # [D]
            feature_tensor = torch.from_numpy(feature_vector).float()
        else:
            # Sliding window: extract specific temporal segment [unit_duration, D]
            window = feat[sample["start_sec"]:sample["end_sec"]]  # [unit_duration, D]
            feature_tensor = torch.from_numpy(np.array(window)).float()

        return feature_tensor, sample["label"], sample["video_id"]

    def get_video_ids(self) -> List[str]:
        """Return list of video IDs for all samples (for video-level evaluation)."""
        return [s["video_id"] for s in self.samples]

    def get_labels(self) -> List[int]:
        """Return list of labels for all samples."""
        return [s["label"] for s in self.samples]
