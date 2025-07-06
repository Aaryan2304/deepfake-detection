# data_utils.py
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DFDCDataLoader:
    """Data loader for the DeepFake Detection Challenge (DFDC) dataset."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.train_video_dir = self.root_dir / 'train_sample_videos'
        self.metadata_path = self.train_video_dir / 'metadata.json'

    def extract_face_frames(self, video_path: str, max_frames: int = 10, face_size: int = 224) -> List[np.ndarray]:
        """Extracts and crops faces from video frames."""
        frames = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 3 == 0:  # Process every 3rd frame to get variety
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    # Get the largest face
                    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                    face_crop = frame[y:y+h, x:x+w]
                    face_crop_resized = cv2.resize(face_crop, (face_size, face_size))
                    frames.append(cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB))
            frame_count += 1

        cap.release()
        return frames

    def create_dataset_from_videos(self, output_dir: str, max_videos: int = None) -> Dict[str, List[str]]:
        """Creates a dataset of face images from the videos."""
        output_dir = Path(output_dir)
        real_dir = output_dir / 'real'
        fake_dir = output_dir / 'fake'
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        with open(self.metadata_path) as f:
            metadata = json.load(f)

        dataset_info = {'real': [], 'fake': []}
        video_files = list(metadata.keys())
        if max_videos:
            video_files = video_files[:max_videos]

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = str(self.train_video_dir / video_file)
            label = metadata[video_file]['label']

            try:
                face_frames = self.extract_face_frames(video_path)

                for i, face_frame in enumerate(face_frames):
                    filename = f"{Path(video_file).stem}_{i}.jpg"
                    if label == 'REAL':
                        filepath = real_dir / filename
                        dataset_info['real'].append(str(filepath))
                    else:
                        filepath = fake_dir / filename
                        dataset_info['fake'].append(str(filepath))

                    cv2.imwrite(str(filepath), cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logger.error(f"Error processing video {video_file}: {e}")

        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=4)

        logger.info(f"Dataset created with {len(dataset_info['real'])} real and {len(dataset_info['fake'])} fake images.")
        return dataset_info

class DeepfakeDatasetBuilder:
    """Builds train/val/test splits for the deepfake dataset."""

    def __init__(self, dataset_info: Dict[str, List[str]], test_size: float = 0.2, val_size: float = 0.15):
        self.dataset_info = dataset_info
        self.test_size = test_size
        self.val_size = val_size

    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Creates train, validation, and test splits."""
        real_paths = self.dataset_info['real']
        fake_paths = self.dataset_info['fake']

        # Create DataFrames
        df_real = pd.DataFrame({'path': real_paths, 'label': 0})
        df_fake = pd.DataFrame({'path': fake_paths, 'label': 1})

        # Balance the dataset before splitting
        min_samples = min(len(df_real), len(df_fake))
        df_real = df_real.sample(n=min_samples, random_state=42)
        df_fake = df_fake.sample(n=min_samples, random_state=42)

        df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and test
        train_df, test_df = train_test_split(df, test_size=self.test_size, stratify=df['label'], random_state=42)

        # Split train into train and validation
        train_df, val_df = train_test_split(train_df, test_size=self.val_size / (1 - self.test_size), stratify=train_df['label'], random_state=42)

        logger.info(f"Train split: {len(train_df)} samples")
        logger.info(f"Validation split: {len(val_df)} samples")
        logger.info(f"Test split: {len(test_df)} samples")

        return train_df, val_df, test_df

    def save_splits(self, output_dir: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Saves the data splits to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        logger.info(f"Data splits saved to {output_dir}")