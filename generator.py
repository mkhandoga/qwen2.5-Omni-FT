import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SyntheticVideoDataset(Dataset):
    """
    A PyTorch Dataset that generates synthetic videos (sequences of frames) with distinct patterns for each class.
    Each video is associated with a single label (action class).
    """
    def __init__(self, num_videos=100, num_classes=5, frames_per_video=16, 
                 frame_height=64, frame_width=64, max_skip=2, seed=None):
        super().__init__()
        self.num_videos = num_videos
        self.num_classes = num_classes
        self.frames_per_video = frames_per_video
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.max_skip = max_skip

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Define class names
        self.classes = [f"class_{i}" for i in range(num_classes)]
        predefined_names = ["horizontal move", "vertical move", "stationary", "random move", "diagonal move"]
        for i, name in enumerate(predefined_names):
            if i < num_classes:
                self.classes[i] = name

        # Pre-generate all videos and labels
        self.videos = []
        self.labels = []
        for i in range(num_videos):
            label_idx = i % num_classes
            frames = self._generate_video_frames(label_idx)
            self.videos.append(frames)
            self.labels.append(label_idx)
    
    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        # Retrieve frames (list of PIL Images) and label
        frames = self.videos[idx]
        label = self.labels[idx]

        # Possibly skip frames or do horizontal flip
        frames = self._apply_augmentations(frames)
        return frames, label

    def _generate_video_frames(self, label_idx):
        """
        Generate a base sequence of frames for a given label_idx.
        Patterns: moving dot, blinking, random teleport, etc.
        """
        base_length = self.frames_per_video * max(1, self.max_skip)
        H, W = self.frame_height, self.frame_width
        frames = []

        # e.g. label=0 => horizontal movement, label=1 => vertical, etc.
        if label_idx == 0:
            frames = self._gen_horizontal_dot(H, W, base_length)
        elif label_idx == 1:
            frames = self._gen_vertical_dot(H, W, base_length)
        elif label_idx == 2:
            frames = self._gen_blinking_dot(H, W, base_length)
        elif label_idx == 3:
            frames = self._gen_random_teleport(H, W, base_length)
        elif label_idx == 4:
            frames = self._gen_bouncing_diag(H, W, base_length)
        else:
            frames = self._gen_random_teleport(H, W, base_length)
        return frames

    def _apply_augmentations(self, frames):
        """
        Applies random skip (temporal) plus random horizontal flip.
        Returns a list of exactly self.frames_per_video frames.
        """
        total_frames = len(frames)  # base_length
        target_len = self.frames_per_video
        skip = 1
        if self.max_skip is not None and self.max_skip > 1:
            skip = random.randint(1, self.max_skip)

        max_start = total_frames - skip * target_len
        start = 0
        if max_start > 0:
            start = random.randint(0, max_start)

        selected = [frames[start + i * skip] for i in range(target_len)]

        # Spatial augmentation: horizontal flip with 50% probability
        if random.random() < 0.5:
            flipped = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in selected]
            selected = flipped
        return selected

    # Below are your existing pattern-generation helpers for each label type.
    def _gen_horizontal_dot(self, H, W, length):
        frames = []
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        dx = random.choice([-1, 1])
        for t in range(length):
            arr = np.zeros((H, W, 3), dtype=np.uint8)
            arr[max(0, y-2):min(H, y+3), max(0, x-2):min(W, x+3)] = 255
            frames.append(Image.fromarray(arr))
            nx = x + dx
            if nx < 0 or nx >= W:
                dx = -dx
                nx = x + dx
                nx = max(0, min(W-1, nx))
            x = nx
        return frames

    def _gen_vertical_dot(self, H, W, length):
        frames = []
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        dy = random.choice([-1, 1])
        for t in range(length):
            arr = np.zeros((H, W, 3), dtype=np.uint8)
            arr[max(0, y-2):min(H, y+3), max(0, x-2):min(W, x+3)] = 255
            frames.append(Image.fromarray(arr))
            ny = y + dy
            if ny < 0 or ny >= H:
                dy = -dy
                ny = y + dy
                ny = max(0, min(H-1, ny))
            y = ny
        return frames

    def _gen_blinking_dot(self, H, W, length):
        frames = []
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        phase = random.choice([0, 1])
        for t in range(length):
            arr = np.zeros((H, W, 3), dtype=np.uint8)
            if ((t + phase) % 2 == 0):
                arr[max(0, y-2):min(H, y+3), max(0, x-2):min(W, x+3)] = 255
            frames.append(Image.fromarray(arr))
        return frames

    def _gen_random_teleport(self, H, W, length):
        frames = []
        for t in range(length):
            arr = np.zeros((H, W, 3), dtype=np.uint8)
            x = random.randint(0, W-1)
            y = random.randint(0, H-1)
            arr[max(0, y-2):min(H, y+3), max(0, x-2):min(W, x+3)] = 255
            frames.append(Image.fromarray(arr))
        return frames

    def _gen_bouncing_diag(self, H, W, length):
        frames = []
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        dx = random.choice([-1, 1])
        dy = random.choice([-1, 1])
        for t in range(length):
            arr = np.zeros((H, W, 3), dtype=np.uint8)
            arr[max(0, y-2):min(H, y+3), max(0, x-2):min(W, x+3)] = 255
            frames.append(Image.fromarray(arr))
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= W:
                dx = -dx
                nx = x + dx
                nx = max(0, min(W-1, nx))
            if ny < 0 or ny >= H:
                dy = -dy
                ny = y + dy
                ny = max(0, min(H-1, ny))
            x, y = nx, ny
        return frames
