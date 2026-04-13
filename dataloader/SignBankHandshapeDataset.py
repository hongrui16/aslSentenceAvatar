"""
ASL SignBank Handshape Video Dataset
====================================
Loads SignBank videos and maps them to classification labels.
Supports two tasks via `task` parameter:
    - "finger"    → "Dominant hand - Selected Fingers" (10 classes)
    - "handshape" → "Dominant hand - Flexion" (9 classes)

Supports two loading modes:
    - frames_dir provided → load preextracted .pt tensors (fast)
    - frames_dir=None     → decode from video on the fly (slow)
"""

import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


# ── Task definitions ──
TASK_CONFIG = {
    'finger': {
        'csv_col': 'Dominant hand - Selected Fingers',
        'classes': ['imrp', 'i', 'im', 'thumb', 'm', 'p', 'imr', 'r', 'mrp', 'ip'],
    },
    'handshape': {
        'csv_col': 'Dominant hand - Flexion',
        'classes': [
            '1 (fully open)', '5 (curved open)', '3 (flat open)',
            '6 (curved closed)', '7 (fully closed)', '2 (bent or closed)',
            '4 (flat closed)', 'Crossed', 'Stacked',
        ],
    },
}


def get_task_info(task):
    """Return (csv_col, class_list, class_to_idx, num_classes) for a task."""
    cfg = TASK_CONFIG[task]
    classes = cfg['classes']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return cfg['csv_col'], classes, class_to_idx, len(classes)


class SignBankHandshapeDataset(Dataset):
    """
    Args:
        csv_path:      path to asl_signbank_dictionary-export.csv
        video_dir:     path to video directory (used for annotation matching)
        task:          "finger" or "handshape"
        n_frames:      number of frames to sample per video (default 16)
        img_size:      resize frames to (img_size, img_size)
        split_indices: list of integer indices into the valid sample list
        augment:       whether to apply training augmentations
        frames_dir:    path to preextracted .pt frames (if None, decode from video)
    """

    def __init__(self, csv_path, video_dir, task='finger', n_frames=16,
                 img_size=224, split_indices=None, augment=False, frames_dir=None):
        super().__init__()
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.img_size = img_size
        self.task = task
        self.frames_dir = frames_dir

        self.csv_col, self.classes, self.class_to_idx, self.num_classes = \
            get_task_info(task)

        # Build list of (video_path, label_idx) for valid samples
        all_samples = self._load_annotations(csv_path, video_dir, task,
                                             frames_dir=frames_dir)

        if split_indices is not None:
            self.samples = [all_samples[i] for i in split_indices]
        else:
            self.samples = all_samples

        # Augmentation transform (applied on top of preextracted tensors too)
        self.augment = augment
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.aug_transform = None

        # Only needed for on-the-fly video decoding
        if frames_dir is None:
            if augment:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])

    @staticmethod
    def _load_annotations(csv_path, video_dir, task='finger', frames_dir=None):
        """Parse CSV → list of (file_path, label_idx), filtering for valid entries."""
        csv_col, classes, class_to_idx, _ = get_task_info(task)

        # Build lookup: stem → full path (from frames_dir or video_dir)
        lookup_dir = frames_dir if frames_dir else video_dir
        stem_to_path = {}
        for f in os.listdir(lookup_dir):
            stem = os.path.splitext(f)[0]
            stem_to_path[stem] = os.path.join(lookup_dir, f)

        samples = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row[csv_col].strip()
                if label not in class_to_idx:
                    continue
                gloss = row['Annotation ID Gloss'].strip()
                if gloss not in stem_to_path:
                    continue
                samples.append((stem_to_path[gloss], class_to_idx[label]))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        if self.frames_dir is not None:
            frames = self._load_pt(file_path)
        else:
            frames = self._load_video(file_path)

        return frames, label

    def _load_pt(self, path):
        """Load preextracted .pt tensor → (n_frames, C, H, W) float32."""
        frames = torch.load(path, map_location='cpu', weights_only=True).float()

        if self.augment and self.aug_transform is not None:
            # Denormalize → augment → re-normalize per frame
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
            augmented = []
            for i in range(frames.shape[0]):
                frame = frames[i] * std + mean  # denormalize to [0,1]
                frame = frame.clamp(0, 1)
                pil = transforms.ToPILImage()(frame)
                pil = self.aug_transform(pil)
                augmented.append(normalize(transforms.ToTensor()(pil)))
            frames = torch.stack(augmented)

        return frames

    def _load_video(self, path):
        """Read video, sample n_frames uniformly, apply transforms."""
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return torch.zeros(self.n_frames, 3, self.img_size, self.img_size)

        if total >= self.n_frames:
            indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        else:
            indices = list(range(total)) + [total - 1] * (self.n_frames - total)

        frames = []
        frame_cache = {}
        for fi in indices:
            if fi in frame_cache:
                frames.append(frame_cache[fi].clone())
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, self.img_size, self.img_size))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = self.transform(frame)
            frame_cache[fi] = frame_t
            frames.append(frame_t)

        cap.release()
        return torch.stack(frames)

    def get_class_counts(self):
        """Return dict of class_name → count."""
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {self.classes[k]: v for k, v in counts.items()}
