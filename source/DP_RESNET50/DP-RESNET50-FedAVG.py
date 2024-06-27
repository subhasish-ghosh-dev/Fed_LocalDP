import flwr as fl
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from collections import OrderedDict

from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
from typing import List

import pydicom
import rarfile
import zipfile
import pydicom
from PIL import Image
import math, json, os, sys

from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification.pytorch import PyTorchClassifier
from torchvision import datasets, transforms, models
import os

zip_path = 'rsna-pneumonia-detection-challenge.zip'
extract_path = 'rsna-pneumonia-detection-challenge'
#with zipfile.ZipFile(zip_path, 'r') as zf:
#    zf.extractall(extract_path)
train_images_path = os.path.join(extract_path, 'stage_2_train_images')
labels_csv_path = os.path.join(extract_path, 'stage_2_train_labels.csv')
labels_df = pd.read_csv(labels_csv_path)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(img):  
    pixel_array = np.array(img)
    return pixel_array
def conv_dcm_img(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    pixel_array = dicom.pixel_array
    image = Image.fromarray(pixel_array)
    return image
feature_vectors = []
labels = []
for index, row in labels_df.iterrows():
    dicom_id = row['patientId']
    label = row['Target']
    dicom_path = os.path.join(train_images_path, f"{dicom_id}.dcm")    
    if os.path.exists(dicom_path): 
        image=transform(conv_dcm_img(dicom_path))
        feature_vector = extract_features(image)
        feature_vectors.append(feature_vector)
        labels.append(label)
        if index%100==0:
            print(index)
            if index>4000:
                break

X = np.array(feature_vectors)
y = np.array(labels)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
#print(X_scaled.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)
print(f'Training data shape: {X_train.shape}')
print(f'Testing data shape: {X_test.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Testing labels shape: {y_test.shape}')

device = torch.device("cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)  
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)   

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class GNResNet(nn.Module):
    def __init__(self, original_resnet, num_groups=32):
        super(GNResNet, self).__init__()
        self.model = original_resnet
        self.num_groups = num_groups
        self._replace_batchnorm_with_groupnorm(self.model)

    def _replace_batchnorm_with_groupnorm(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                setattr(module, name, nn.GroupNorm(self.num_groups, num_channels))
            elif isinstance(child, nn.Sequential) or isinstance(child, nn.Module):
                self._replace_batchnorm_with_groupnorm(child)

    def forward(self, x):
        return self.model(x)

device = torch.device("cpu")
original_resnet = models.resnet50(pretrained=True)
num_classes = 2
original_resnet.fc = nn.Linear(original_resnet.fc.in_features, num_classes)
model = GNResNet(original_resnet, num_groups=32)
model = model.to(device)

def test(net, test_loader):
    correct, total, loss = 0, 0, 0
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            outputs = net(data)
            loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')    
    return loss, accuracy

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    set_parameters(model, parameters)  # Update model with the latest parameters
    loss, accuracy = test(model, test_loader)
    print(f"Server Round: {server_round} loss: {loss}  accuracy: {accuracy}")
    f = open("server_D-DP-RESNET50.txt", "a")
    f.write(f"SERVER ROUND:{server_round} LOSS:{loss} ACCURACY:{accuracy:.8f}\n")
    f.close()
    return loss, {"accuracy": accuracy}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  
    min_fit_clients=5,  
    min_available_clients=5,  
    evaluate_fn=evaluate,  # Pass the evaluation function
)
fl.server.start_server(server_address="172.16.10.220:8080", config=fl.server.ServerConfig(num_rounds=50), strategy=strategy)
