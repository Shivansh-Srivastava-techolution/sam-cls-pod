import os
import sys

current_directory = os.getcwd()

print(sys.path)

import pandas as pd
import json

import torch
import torch.nn as nn

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
    """
    This must match the architecture used during training.
    For example:
      - 2 convolutional layers
      - Adaptive pooling
      - 1 linear output
    Adjust channels, kernel_size, etc. as per your training script.
    """
    def __init__(self, num_features=9, num_classes=2):
        super(CNN1DModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.fc   = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, num_features=9)
        We permute to: (batch_size, num_features=9, seq_len) for Conv1d.
        """
        x = x.permute(0, 2, 1)  # => (batch, 9, seq_len)
        x = torch.relu(self.conv1(x))    # => (batch, 64, seq_len)
        x = torch.relu(self.conv2(x))    # => (batch, 128, seq_len)
        x = self.pool(x)                 # => (batch, 128, 1)
        x = x.squeeze(-1)                # => (batch, 128)
        out = self.fc(x)                 # => (batch, num_classes)
        return out
    
def inference(json_path, cnn_model, device, class_map):

    # Get feature vectors
    with open(json_path, "r") as f:
        content = json.load(f)
        features = content["features"]

    # Convert to (1, seq_len, 9) for CNN
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward pass
    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(input_tensor)  # => shape [1, 2]
        predicted_idx = torch.argmax(outputs, dim=1).item()

    # Map to label
    print("cls_amo")
    print(class_map)
    print("Inference: ", predicted_idx, class_map.get(predicted_idx, "unknown"))
    return class_map.get(predicted_idx, "unknown")

def main(test_csv_path, dataset_path, output_file, class_map, model_save_path=None):
    """
    dataset_path: the path where testing data is stored
    models_path: the path where the weights of the trained model is stored
    output_file: the path where the predictions of the model will be stored

    The main objective of this function is to test the model using the weights present at the path "models_path" and on the data present in the "dataset_path".
    This function should store it's prediction into the file "output_file", in a csv format. You can look at example_results.csv to gain more clarity about it.
    This function should return a dictionary containing overall accuracy and also accuracy for specific labels. You can look at example_acc_dict.json to gain more clarity about it.

    """

    dtype_dict = {'label': str, 'parentLabel': str} 
    df = pd.read_csv(test_csv_path, dtype=dtype_dict)
    
    resource_ids = df['_id']
    gcs_filenames = df['GCStorage_file_path']
    actual_labels = df['label']
    
    columns = ['resource_id', 'trim_id', 'filename', 'label', 'parentLabel', 'predicted', 'confidence_score']+['startTime', 'endTime']
    output_df = pd.DataFrame(
        columns=columns)

    confusion_matrix = {}
    for label in actual_labels:
        confusion_matrix[str(label)] = {}
        for labeld in actual_labels:
            confusion_matrix[str(label)][str(labeld)] = 0
    
    # for actual_label in actual_labels:
    #     confusion_matrix[actual_label] = {}
    #     for actual_labeld in actual_labels:
    #         confusion_matrix[actual_label][actual_labeld] = 0

    records_inf = {}
    count = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(os.listdir(dataset_path))

    # Create and load the CNN1DModel
    cnn_model = CNN1DModel(num_features=9, num_classes=num_classes)
    cnn_model.load_state_dict(torch.load(model_save_path, map_location=device))
    cnn_model.to(device)
 

    # records_inf = []
    null_false_positive_count = 0
    for resource_id, gcs_filename, actual_label in zip(resource_ids, gcs_filenames, actual_labels):
        count+=1
        actual_label = str(actual_label)
        # Load an image file to test, resizing it to 150x150 pixels (as required by this model)

        # If the hyperparameter is coming during training
        json_file = os.path.basename(gcs_filename).replace('.mp4', '.json')
        json_path = os.path.join(dataset_path, actual_label, "json_data", json_file)

        if not os.path.exists(json_path):
            print(json_path, "NOT FOUND")
            predicted_label = 'invalid'
            score = 100

        else:
            predicted_label = inference(json_path, cnn_model, device, class_map) 
            score = 100
        
        if predicted_label == 'unknown':
            predicted_label = predicted_label
            score = 0
            
        # Add the data to the output dataframe
        try:
            meta_data = json.dumps({"raw_text": predicted_label})
        except:
            meta_data = json.dumps({"raw_text":''})
            
        append_df = {'resource_id': resource_id,
                                      'trim_id': '',
                                      'filename': os.path.basename(gcs_filename),
                                      'label': actual_label,
                                      'parentLabel': actual_label,
                                      'predicted': predicted_label,
                                      'confidence_score': round(score, 2),
                                      
                                      # f'label_{actual_label}': round(score, 2) if actual_label == 'No_Items' else 0,
                                      'startTime': 0.0,
                                      'endTime': 1.0, 
                                      'metadata':meta_data,
                                      f'label_{actual_label}': round(score, 2) if actual_label == predicted_label else 0
                                      }
        
        print("Final Label : ", predicted_label)
        print("Final Score : ", score)
        # for z in target_labels:
        #     append_df[f'label_{z}'] = round(score*100, 2) if z==label else 0
        records_inf[actual_label] = predicted_label
        output_df = output_df._append(append_df, ignore_index=True)


        # Updating the confusion matrix
        print("="*30)
        print("Actual_label: ", actual_label)
        print("Predicted_label: ", predicted_label)
        try:
        # if predicted_label == actual_label:
            confusion_matrix[actual_label][predicted_label] = confusion_matrix[actual_label][predicted_label] + 1
            # print("updated value: ",confusion_matrix[actual_label][predicted_label])
            print("****************** 1st case in confusion matrix ***************************")
            
#         elif confusion_matrix[actual_label][predicted_label] == 1:
#             confusion_matrix[actual_label][predicted_label] += 1
#             print("****************** 3rd case in confusion matrix ***************************")
            
#         else:
#             confusion_matrix[actual_label][predicted_label] = 1
#             print("****************** 2nd case in confusion matrix ***************************")
        except:
            print("it came here")
            null_false_positive_count += 1
            # print("****************** 2nd case in confusion matrix ***************************")
            # confusion_matrix[str(actual_label)][str(predicted_label)] = 1
            
    total_not_detected = null_false_positive_count / len(df) 
    total_not_detected = round(total_not_detected,2) * 100
    # Calculate the accuracy
    df_rec = pd.DataFrame(records_inf.items(), columns=['actual_label', 'predicted'])
    
    acc = {'Total':round((df_rec[df_rec['actual_label']==df_rec['predicted']].shape[0]/df_rec.shape[0]) * 100, 2)}

    
    name_counts = output_df['label'].value_counts().reset_index()
    name_counts.columns = ['label', 'Count']

    # Merge count with original DataFrame
    df1 = output_df.merge(name_counts, on='label')

    # Group by 'Name' and aggregate with sum
    grouped_df = df1.groupby('label').agg({'confidence_score': ['sum',"mean"]}).reset_index()
    grouped_df.columns = ['label', 'Total_confidence', "Average_confidence"]
    
    for index in range(len(grouped_df)):
        acc[grouped_df["label"][index]] = grouped_df["Average_confidence"][index]

    
    
    

    for col in output_df.columns.tolist():
        if col.find("label_") >= 0:
            print("it came to fill nul values")
            output_df[[col]] = output_df.loc[:,[col]].fillna(value=0)

    print("Output Path : ", output_file)
    
    output_df.to_csv(output_file)

    acc["not_detected_percentage"] = total_not_detected
    return acc, confusion_matrix


if "__main__" == __name__:
    test_csv_path = "/home/jupyter/background_rem/pod_updated/add_rem_sku/testing_pod/Keras-Applications-Pod/data/test_6575c131f5e454c3bb04ca22/defaultDataSetCollection_65753d843b58d84c89eaeded_resources.csv"
    dataset_path = "/home/jupyter/background_rem/pod_updated/add_rem_sku/testing_pod/Keras-Applications-Pod/data/test_6575c131f5e454c3bb04ca22/test data"
    models_path = "/home/anandhakrishnan/Projects/American-Ordinance/ao-fedramp/training-pods/Yolo-V58-Pod/runs/detect/train/weights/best.pt"
    output_file = "temp123.csv"
    statistics_file = "temp123.json"
    model_details = {}
    # main(test_csv_path, test_json_path, dataset_path, models_path, output_file, statistics_file, hyperparameters)
    main(test_csv_path, dataset_path, models_path, output_file, model_details)
