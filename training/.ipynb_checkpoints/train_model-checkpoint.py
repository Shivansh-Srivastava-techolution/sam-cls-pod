import json
import os
import shutil
import pandas as pd
import time
import yaml
import csv
import random
import pathlib
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import pytorchvideo.data
import torch
import numpy as np
import evaluate
import imageio
from PIL import Image
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from transformers import pipeline
from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback

# Load metric globally
metric = evaluate.load("accuracy")

class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_metric = -float('inf')
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "best_model"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "latest_checkpoint"), exist_ok=True)
    
    def _copy_checkpoint_files(self, src_dir, dest_dir, is_best_model=False):
        """Helper function to copy checkpoint files"""
        if os.path.exists(src_dir):
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)
            print(src_dir, dest_dir, is_best_model)
            
            if is_best_model:
                # For best_model, directly copy model checkpoint files
                # Look for model checkpoint files (.bin, .json)
                
                checkpoint_files = [f for f in os.listdir(src_dir)]
                print(checkpoint_files)
                
                if checkpoint_files:
                    for file in checkpoint_files:
                        src_file_path = os.path.join(src_dir, file)
                        dest_file_path = os.path.join(dest_dir, file)
                        print(f"Copying {src_file_path} to {dest_file_path}")
                        shutil.copy2(src_file_path, dest_file_path)
                else:
                    print("No model checkpoint files found in source directory.")
            else:
                # For latest_checkpoint, copy the entire directory structure
                shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Get the current checkpoint directory
        # Find the latest checkpoint in the output directory
        checkpoints = [d for d in os.listdir(self.save_path) if d.startswith('checkpoint-')]
        if checkpoints: 
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            current_checkpoint = os.path.join(self.save_path, latest_checkpoint)
        
            # Save latest checkpoint
            latest_path = os.path.join(self.save_path, "latest_checkpoint")
            self._copy_checkpoint_files(current_checkpoint, latest_path, is_best_model=False)

            # Check if this is the best model
            metric_value = metrics.get("eval_accuracy", 0)
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                best_path = os.path.join(self.save_path, "best_model")

                # Copy files to best_model without nested directories
                self._copy_checkpoint_files(current_checkpoint, best_path, is_best_model=True)

                # Save the metric value
                with open(os.path.join(best_path, "best_metric.txt"), "w") as f:
                    f.write(f"Best eval_accuracy: {self.best_metric}")
                print(f"\nNew best model saved with accuracy: {self.best_metric:.4f}")

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    """Collate function for data loading."""
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def delete_inner_folders_except_one(location, except_folder):
    """Delete all inner folders except the specified one."""
    for item in os.listdir(location):
        item_path = os.path.join(location, item)
        if os.path.isdir(item_path):
            if item != except_folder:
                shutil.rmtree(item_path)

def train_val_split(train_dataset_path):
    """Split data into training and validation sets."""
    train_ratio = 0.85
    val_ratio = 0.15
    classes = ["grab", "invalid"]

    # Create directories
    for split in ["train", "val"]:
        for class_name in classes:
            os.makedirs(os.path.join(train_dataset_path, split, class_name), exist_ok=True)

    # Split data
    for class_name in classes:
        class_dir = os.path.join(train_dataset_path, class_name)
        files = os.listdir(class_dir)
        
        random.shuffle(files)
        
        train_size = int(len(files) * train_ratio)
        val_size = int(len(files) * val_ratio)
        
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        
        # Move files
        for file in train_files:
            shutil.copy(os.path.join(train_dataset_path, class_name, file), 
                       os.path.join(train_dataset_path, "train", class_name, file))
            os.remove(os.path.join(train_dataset_path, class_name, file))
        for file in val_files:
            shutil.copy(os.path.join(train_dataset_path, class_name, file), 
                       os.path.join(train_dataset_path, "val", class_name, file))
            os.remove(os.path.join(train_dataset_path, class_name, file))

    print("Data successfully split into train and validation directories!")

def main(train_dataset_path, test_dataset_path, models_path, model_details, train_csv_path, test_csv_path):
    """Main training function."""
    num = random.randint(111, 9999999999999)
    dataset_root_path = pathlib.Path(train_dataset_path)
    test_dataset_path = pathlib.Path(test_dataset_path)
    
    train_val_split(train_dataset_path)
    
    # Count videos
    video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
    video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
    video_count_test = len(list(test_dataset_path.glob("*/*.mp4")))
    video_total = video_count_train + video_count_val + video_count_test
    print(f"Total videos: {video_total}")

    # Get video paths
    all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.mp4")) + 
        list(dataset_root_path.glob("val/*/*.mp4")) +
        list(test_dataset_path.glob("*/*.mp4"))
    )
    
    # Set up labels
    classes = ["grab", "invalid"]
    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Unique classes: {list(label2id.keys())}.")

    # Initialize model
    model_ckpt = "MCG-NJU/videomae-base"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Set up processing parameters
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    # Video parameters
    num_frames_to_sample = model.config.num_frames
    # {"epochs" : 10, "batch_size" : 2, "fps" : 30, "sample_rate" : 4}
    sample_rate = model_details['hyperParameter']['sample_rate']
    fps = model_details['hyperParameter']['fps']
    num_epochs = model_details['hyperParameter']['epochs']
    batch_size = model_details['hyperParameter']['batch_size']
    
    clip_duration = num_frames_to_sample * sample_rate / fps

    # Define transforms
    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(num_frames_to_sample),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(resize_to),
                RandomHorizontalFlip(p=0.5),
            ]),
        ),
    ])

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(num_frames_to_sample),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                Resize(resize_to),
            ]),
        ),
    ])

    # Create datasets
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_dataset = pytorchvideo.data.Ucf101(
        data_path=test_dataset_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    print(f"Dataset sizes - Train: {train_dataset.num_videos}, Val: {val_dataset.num_videos}, Test: {test_dataset.num_videos}")

    # Training setup
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"models/{model_name}-{num}"
    

    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_only_model=True,
        greater_is_better=True,
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        report_to=[], # This disables all integrations including wandb
    )

    # Initialize callback
    checkpoint_callback = CustomCheckpointCallback(save_path=new_model_name)

    # Create trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[checkpoint_callback],
    )
    
    # Train
    train_results = trainer.train()
    print("************************************ model trained successfully **********************************************")
    
    return new_model_name
