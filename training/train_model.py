# import json
# import os
# import shutil
# import pandas as pd
# import time
# import yaml
# import csv
# import random
# import pathlib

# # import pytorchvideo.data
# import torch
# import numpy as np

# # class CustomCheckpointCallback(TrainerCallback):
# #     def __init__(self, save_path):
# #         self.save_path = save_path
# #         self.best_metric = -float('inf')
# #         os.makedirs(save_path, exist_ok=True)
# #         os.makedirs(os.path.join(save_path, "best_model"), exist_ok=True)
# #         os.makedirs(os.path.join(save_path, "latest_checkpoint"), exist_ok=True)
    
# #     def _copy_checkpoint_files(self, src_dir, dest_dir, is_best_model=False):
# #         """Helper function to copy checkpoint files"""
# #         if os.path.exists(src_dir):
# #             if os.path.exists(dest_dir):
# #                 shutil.rmtree(dest_dir)
# #             os.makedirs(dest_dir, exist_ok=True)
# #             print(src_dir, dest_dir, is_best_model)
            
# #             if is_best_model:
# #                 # For best_model, directly copy model checkpoint files
# #                 # Look for model checkpoint files (.bin, .json)
                
# #                 checkpoint_files = [f for f in os.listdir(src_dir)]
# #                 print(checkpoint_files)
                
# #                 if checkpoint_files:
# #                     for file in checkpoint_files:
# #                         src_file_path = os.path.join(src_dir, file)
# #                         dest_file_path = os.path.join(dest_dir, file)
# #                         print(f"Copying {src_file_path} to {dest_file_path}")
# #                         shutil.copy2(src_file_path, dest_file_path)
# #                 else:
# #                     print("No model checkpoint files found in source directory.")
# #             else:
# #                 # For latest_checkpoint, copy the entire directory structure
# #                 shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
    
# #     def on_evaluate(self, args, state, control, metrics, **kwargs):
# #         # Get the current checkpoint directory
# #         # Find the latest checkpoint in the output directory
# #         checkpoints = [d for d in os.listdir(self.save_path) if d.startswith('checkpoint-')]
# #         if checkpoints: 
# #             latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
# #             current_checkpoint = os.path.join(self.save_path, latest_checkpoint)
        
# #             # Save latest checkpoint
# #             latest_path = os.path.join(self.save_path, "latest_checkpoint")
# #             self._copy_checkpoint_files(current_checkpoint, latest_path, is_best_model=False)

# #             # Check if this is the best model
# #             metric_value = metrics.get("eval_accuracy", 0)
# #             if metric_value > self.best_metric:
# #                 self.best_metric = metric_value
# #                 best_path = os.path.join(self.save_path, "best_model")

# #                 # Copy files to best_model without nested directories
# #                 self._copy_checkpoint_files(current_checkpoint, best_path, is_best_model=True)

# #                 # Save the metric value
# #                 with open(os.path.join(best_path, "best_metric.txt"), "w") as f:
# #                     f.write(f"Best eval_accuracy: {self.best_metric}")
# #                 print(f"\nNew best model saved with accuracy: {self.best_metric:.4f}")

# # def compute_metrics(eval_pred):
# #     """Compute metrics for evaluation."""
# #     predictions = np.argmax(eval_pred.predictions, axis=1)
# #     return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# # def collate_fn(examples):
# #     """Collate function for data loading."""
# #     pixel_values = torch.stack(
# #         [example["video"].permute(1, 0, 2, 3) for example in examples]
# #     )
# #     labels = torch.tensor([example["label"] for example in examples])
# #     return {"pixel_values": pixel_values, "labels": labels}

# # def delete_inner_folders_except_one(location, except_folder):
# #     """Delete all inner folders except the specified one."""
# #     for item in os.listdir(location):
# #         item_path = os.path.join(location, item)
# #         if os.path.isdir(item_path):
# #             if item != except_folder:
# #                 shutil.rmtree(item_path)

# # def train_val_split(train_dataset_path):
# #     """Split data into training and validation sets."""
# #     train_ratio = 0.85
# #     val_ratio = 0.15
# #     classes = ["grab", "invalid"]

# #     # Create directories
# #     for split in ["train", "val"]:
# #         for class_name in classes:
# #             os.makedirs(os.path.join(train_dataset_path, split, class_name), exist_ok=True)

# #     # Split data
# #     for class_name in classes:
# #         class_dir = os.path.join(train_dataset_path, class_name)
# #         files = os.listdir(class_dir)
        
# #         random.shuffle(files)
        
# #         train_size = int(len(files) * train_ratio)
# #         val_size = int(len(files) * val_ratio)
        
# #         train_files = files[:train_size]
# #         val_files = files[train_size:train_size + val_size]
        
# #         # Move files
# #         for file in train_files:
# #             shutil.copy(os.path.join(train_dataset_path, class_name, file), 
# #                        os.path.join(train_dataset_path, "train", class_name, file))
# #             os.remove(os.path.join(train_dataset_path, class_name, file))
# #         for file in val_files:
# #             shutil.copy(os.path.join(train_dataset_path, class_name, file), 
# #                        os.path.join(train_dataset_path, "val", class_name, file))
# #             os.remove(os.path.join(train_dataset_path, class_name, file))

# #     print("Data successfully split into train and validation directories!")

# # def main(train_dataset_path, test_dataset_path, models_path, model_details, train_csv_path, test_csv_path):
# #     """Main training function."""
# #     num = random.randint(111, 9999999999999)
# #     dataset_root_path = pathlib.Path(train_dataset_path)
# #     test_dataset_path = pathlib.Path(test_dataset_path)
    
# #     train_val_split(train_dataset_path)
    
# #     # Count videos
# #     video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
# #     video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
# #     video_count_test = len(list(test_dataset_path.glob("*/*.mp4")))
# #     video_total = video_count_train + video_count_val + video_count_test
# #     print(f"Total videos: {video_total}")

# #     # Get video paths
# #     all_video_file_paths = (
# #         list(dataset_root_path.glob("train/*/*.mp4")) + 
# #         list(dataset_root_path.glob("val/*/*.mp4")) +
# #         list(test_dataset_path.glob("*/*.mp4"))
# #     )
    
# #     # Set up labels
# #     classes = ["grab", "invalid"]
# #     class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
# #     label2id = {label: i for i, label in enumerate(class_labels)}
# #     id2label = {i: label for label, i in label2id.items()}

# #     print(f"Unique classes: {list(label2id.keys())}.")

# #     # Initialize model
# #     model_ckpt = "MCG-NJU/videomae-base"
# #     image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
# #     model = VideoMAEForVideoClassification.from_pretrained(
# #         model_ckpt,
# #         label2id=label2id,
# #         id2label=id2label,
# #         ignore_mismatched_sizes=True,
# #     )

# #     # Set up processing parameters
# #     mean = image_processor.image_mean
# #     std = image_processor.image_std
# #     if "shortest_edge" in image_processor.size:
# #         height = width = image_processor.size["shortest_edge"]
# #     else:
# #         height = image_processor.size["height"]
# #         width = image_processor.size["width"]
# #     resize_to = (height, width)

# #     # Video parameters
# #     num_frames_to_sample = model.config.num_frames
# #     # {"epochs" : 10, "batch_size" : 2, "fps" : 30, "sample_rate" : 4}
# #     sample_rate = model_details['hyperParameter']['sample_rate']
# #     fps = model_details['hyperParameter']['fps']
# #     num_epochs = model_details['hyperParameter']['epochs']
# #     batch_size = model_details['hyperParameter']['batch_size']
    
# #     clip_duration = num_frames_to_sample * sample_rate / fps

# #     # Define transforms
# #     train_transform = Compose([
# #         ApplyTransformToKey(
# #             key="video",
# #             transform=Compose([
# #                 UniformTemporalSubsample(num_frames_to_sample),
# #                 Lambda(lambda x: x / 255.0),
# #                 Normalize(mean, std),
# #                 RandomShortSideScale(min_size=256, max_size=320),
# #                 RandomCrop(resize_to),
# #                 RandomHorizontalFlip(p=0.5),
# #             ]),
# #         ),
# #     ])

# #     val_transform = Compose([
# #         ApplyTransformToKey(
# #             key="video",
# #             transform=Compose([
# #                 UniformTemporalSubsample(num_frames_to_sample),
# #                 Lambda(lambda x: x / 255.0),
# #                 Normalize(mean, std),
# #                 Resize(resize_to),
# #             ]),
# #         ),
# #     ])

# #     # Create datasets
# #     train_dataset = pytorchvideo.data.Ucf101(
# #         data_path=os.path.join(dataset_root_path, "train"),
# #         clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
# #         decode_audio=False,
# #         transform=train_transform,
# #     )

# #     val_dataset = pytorchvideo.data.Ucf101(
# #         data_path=os.path.join(dataset_root_path, "val"),
# #         clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
# #         decode_audio=False,
# #         transform=val_transform,
# #     )

# #     test_dataset = pytorchvideo.data.Ucf101(
# #         data_path=test_dataset_path,
# #         clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
# #         decode_audio=False,
# #         transform=val_transform,
# #     )

# #     print(f"Dataset sizes - Train: {train_dataset.num_videos}, Val: {val_dataset.num_videos}, Test: {test_dataset.num_videos}")

# #     # Training setup
# #     model_name = model_ckpt.split("/")[-1]
# #     new_model_name = f"models/{model_name}-{num}"
    

# #     args = TrainingArguments(
# #         new_model_name,
# #         remove_unused_columns=False,
# #         evaluation_strategy="epoch",
# #         save_strategy="epoch",
# #         save_total_limit=1,
# #         learning_rate=5e-5,
# #         per_device_train_batch_size=batch_size,
# #         per_device_eval_batch_size=batch_size,
# #         warmup_ratio=0.1,
# #         logging_steps=10,
# #         load_best_model_at_end=True,
# #         metric_for_best_model="accuracy",
# #         save_only_model=True,
# #         greater_is_better=True,
# #         push_to_hub=False,
# #         max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
# #         report_to=[], # This disables all integrations including wandb
# #     )

# #     # Initialize callback
# #     checkpoint_callback = CustomCheckpointCallback(save_path=new_model_name)

# #     # Create trainer
# #     trainer = Trainer(
# #         model,
# #         args,
# #         train_dataset=train_dataset,
# #         eval_dataset=val_dataset,
# #         processing_class=image_processor,
# #         compute_metrics=compute_metrics,
# #         data_collator=collate_fn,
# #         callbacks=[checkpoint_callback],
# #     )
    
# #     # Train
# #     train_results = trainer.train()
# #     print("************************************ model trained successfully **********************************************")
    
# #     return new_model_name

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

# Configuration
TRAINING_DATA_DIR = "videos/split_data"
MODEL_SAVE_PATH = "cnn1d_model.pth"
EARLY_STOPPING_PATIENCE = 15

# --------------------------------------------------------------------------------
# 1) Custom Dataset
# --------------------------------------------------------------------------------
class MotionDataset(Dataset):
    """
    Expects:
      data   -> list of sequences, each sequence is shape [seq_len, 9]
      labels -> list of integer labels, each label corresponds to one sequence
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert each sequence and label into torch tensors
        # shape: (seq_len, 9)
        seq = torch.tensor(self.data[idx], dtype=torch.float32)
        lab = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, lab

# --------------------------------------------------------------------------------
# 2) Collate function (to handle variable sequence lengths)
# --------------------------------------------------------------------------------
def collate_fn(batch):
    """
    batch: list of (sequence_tensor, label_tensor).

    We'll:
      1) Find the max sequence length within this batch
      2) Pad all sequences to that length along the time dimension
      3) Stack everything into a single batch
    """
    data, labels = zip(*batch)  # data is tuple of Tensors, labels is tuple of Tensors
    
    # Find maximum sequence length in the batch
    max_len = max(seq.size(0) for seq in data)  # seq.size(0) is the time dimension
    
    # Number of features = data[0].size(1) if there's at least 1 sample in the batch
    num_features = data[0].size(1)
    
    # Prepare a padded tensor: shape (batch_size, max_len, num_features)
    padded_data = torch.zeros(len(data), max_len, num_features, dtype=torch.float32)
    
    # Copy each sequence into the padded_data
    for i, seq in enumerate(data):
        seq_len = seq.size(0)
        padded_data[i, :seq_len, :] = seq
    
    labels = torch.stack(labels)  # shape (batch_size,)
    return padded_data, labels

# --------------------------------------------------------------------------------
# 3) 1D CNN Model
# --------------------------------------------------------------------------------
class CNN1DModel(nn.Module):
    """
    A simple 1D CNN that processes sequences of shape:
      (batch, seq_len, num_features)
    
    We'll reshape to (batch, num_features, seq_len) so that 'num_features' acts like "channels",
    and 'seq_len' is the time dimension for the 1D convolution.
    
    Architecture Overview:
      - Conv1d -> ReLU
      - Conv1d -> ReLU
      - AdaptiveMaxPool1d(1) to reduce the time dimension to 1
      - Fully-connected layer to output classification
    """
    def __init__(self, num_features=9, num_classes=2):
        super(CNN1DModel, self).__init__()
        
        # We treat 'num_features' as the number of input channels
        # (conv across the time axis).
        # Feel free to adjust kernel_size, channels, and layers.
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # We reduce the time dimension to a single value per channel
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        # Permute to (batch, num_features, seq_len) so we can apply Conv1d
        x = x.permute(0, 2, 1)  # shape: (batch, 9, seq_len)
        
        x = self.conv1(x)       # shape: (batch, 64, seq_len)
        x = nn.functional.relu(x)
        
        x = self.conv2(x)       # shape: (batch, 128, seq_len)
        x = nn.functional.relu(x)
        
        x = self.pool(x)        # shape: (batch, 128, 1)
        
        # Flatten out the last dimension
        x = x.squeeze(-1)       # shape: (batch, 128)
        
        out = self.fc(x)        # shape: (batch, num_classes)
        return out

# --------------------------------------------------------------------------------
# 4) Data Loading
# --------------------------------------------------------------------------------
def load_training_data(train_dataset_path, test_dataset_path, class_map):
    """
    Reads .json files from provided training and testing folder paths.
    Expects JSON with structure: { "features": <2D list>, "class": <str> }
    
    Returns:
      train_data   -> [list of sequences], each sequence shape [seq_len, 9]
      train_labels -> [list of integer labels]
      test_data    -> [list of sequences], each sequence shape [seq_len, 9]
      test_labels  -> [list of integer labels]
    """
    def load_data_from_folder(folder_path):
        data = []
        labels = []
        
        for class_name, class_label in class_map.items():
            class_dir = os.path.join(folder_path, class_name, "json_data")
            if not os.path.exists(class_dir):
                continue
            
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".json"):
                    file_path = os.path.join(class_dir, file_name)
                    with open(file_path, "r") as f:
                        content = json.load(f)
                        seq_features = content["features"]  # shape: [seq_len, 9]
                        
                        # Append to dataset
                        data.append(seq_features)
                        labels.append(class_label)
        
        return data, labels
    
    train_data, train_labels = load_data_from_folder(train_dataset_path)
    test_data, test_labels = load_data_from_folder(test_dataset_path)
    
    return train_data, train_labels, test_data, test_labels

# --------------------------------------------------------------------------------
# 5) Train & Evaluate Routines
# --------------------------------------------------------------------------------
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for inputs, targets in train_loader:
        # inputs: (batch, seq_len, 9)
        # targets: (batch,)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)            # shape: (batch, num_classes)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

# --------------------------------------------------------------------------------
# 6) Main Training Loop
# --------------------------------------------------------------------------------
def main(train_dataset_path, test_dataset_path, model_details, model_save_path):
 
    epochs = model_details['hyperParameter']['epochs']
    patience = model_details['hyperParameter']['patience']
    batch_size = model_details['hyperParameter']['batch_size']
    learning_rate = model_details['hyperParameter']['learning_rate']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clss = os.listdir(train_dataset_path)
    cls_map = {v: k for k, v in enumerate(clss)}
    clss_map = {k: v for k, v in enumerate(clss)}

    train_data, train_labels, test_data, test_labels = load_training_data(train_dataset_path, test_dataset_path, cls_map)

    # Create Dataset & DataLoader
    train_dataset = MotionDataset(train_data, train_labels)
    test_dataset = MotionDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize 1D CNN Model
    num_features = 9
    num_classes = len(clss)
    model = CNN1DModel(num_features=num_features, num_classes=num_classes).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    num_epochs = epochs
    best_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        train_accuracy = evaluate_model(train_loader, model, device)
        test_accuracy  = evaluate_model(test_loader,  model, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Test Acc: {test_accuracy:.4f}")
        
        # Early stopping
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    print(f"Model saved to {model_save_path}")

    return clss_map

if __name__ == "__main__":
    main()
