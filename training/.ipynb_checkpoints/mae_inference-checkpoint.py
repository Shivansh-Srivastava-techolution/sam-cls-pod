import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)
import cv2
from typing import Optional, Dict, List
import os

class VideoClassifier:
    def __init__(self, model_path: str):
        """
        Initialize the video classifier with a pre-trained model.
        
        Args:
            model_path: Path to the pretrained VideoMAE model
        """
        # Load the image processor and model
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_path)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path)
        
        # Set up model parameters
        self.num_frames = self.model.config.num_frames
        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        
        # Set up image dimensions
        if "shortest_edge" in self.image_processor.size:
            self.height = self.width = self.image_processor.size["shortest_edge"]
        else:
            self.height = self.image_processor.size["height"]
            self.width = self.image_processor.size["width"]
        
        # Create transform pipeline matching original script
        self.transform = Compose([
            UniformTemporalSubsample(self.num_frames),
            Lambda(lambda x: x / 255.0),
            Normalize(self.mean, self.std),
            Resize((self.height, self.width))
        ])
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Preprocess video for inference.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Preprocessed video tensor
        """
        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Convert to tensor with shape (C, T, H, W)
        video = torch.tensor(np.array(frames))
        video = video.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        
        # Apply transforms
        video = self.transform(video)
        
        # Permute to final shape (T, C, H, W)
        video = video.permute(1, 0, 2, 3)
        
        return video

    @torch.no_grad()
    def get_top_label(self, video_path: str) -> str:
        """
        Get just the label with highest probability for a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            String label of the most probable class
        """
        # Preprocess video
        video = self.preprocess_video(video_path)
        
        # Prepare input
        inputs = {
            "pixel_values": video.unsqueeze(0).to(self.device)
        }
        
        # Run inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get the label with highest probability
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class_idx]
        
        return predicted_label
    
    
# # Initialize the classifier
# classifier = VideoClassifier("/home/jupyter/rack/video_classify/videomae-base-finetuned-ucf101-subset/checkpoint-568")

# # Classify a single video
# video_path = "/home/jupyter/rack/video_classify/data/test/invalid/67599e5fe783d0f3a558eedb.mp4"
# top_label = classifier.get_top_label(video_path)
# print(f"Predicted class: {top_label}")
