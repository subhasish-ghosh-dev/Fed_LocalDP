import os
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
            #if index>10000:
            #    break

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)  
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)   

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()
#model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1
train_losses = []
val_losses = []

privacy_engine = PrivacyEngine()
# Set privacy parameters
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=0.1,
    max_grad_norm=1.0,
  )

checkpoint_dir = 'DP-CNN_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
f2 = open("DP-CNN-3.csv", "a")
f2.write(f"Epoch,Loss,Accuracy,Val_Loss,Val_Accuracy\n")

def train(model, train_loader, epochs=35):
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        model = model.to(device)
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            print(f"Loss:{loss}")
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += target.size(0)
            correct += (torch.max(outputs.data, 1)[1] == target).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss:.8f}, accuracy {epoch_acc}")
        f = open("train_xray_35_1.txt", "a")
        f.write(f"Epoch: {epoch+1} Loss: {epoch_loss:.8f}, Accuracy: {epoch_acc:.8f}\n")
        f.close()
        test_correct, test_total, test_loss = 0, 0, 0

        with torch.no_grad():
            for test_data, test_target in test_loader:
                test_data = test_data.to(device)
                test_target = test_target.to(device)
                test_outputs = model(test_data)
                test_loss += criterion(test_outputs, test_target).item()
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_total += test_target.size(0)
                test_correct += (test_predicted == test_target).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / test_total
        print(f'Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        f1 = open("test_xray_35_1.txt", "a")
        f1.write(f"Loss: {test_loss:.8f}, Accuracy: {test_accuracy:.8f}\n")
        f1.close()
        f2.write(f"{epoch+1},{epoch_loss:.8f},{epoch_acc:.8f},{test_loss:.4f},{test_accuracy:.8f}\n")
    f2.close()
    return model
   
def test(net, test_loader):
    correct, total, loss = 0, 0, 0
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            outputs = net(data)
            loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')    
    return loss, accuracy

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

def _attackMIA(X_train, X_test, y_train, y_test, model):
  loss_fn = nn.CrossEntropyLoss()
  attack_train_ratio = 0.5
  attack_train_size = int(len(X_train) * attack_train_ratio)
  attack_test_size = int(len(X_test) * attack_train_ratio)

  mlp_art_model = PyTorchClassifier(model=model, loss=loss_fn, input_shape=(3,224,224), nb_classes=2)
  mlp_attack_bb = MembershipInferenceBlackBox(mlp_art_model, attack_model_type='rf')
  mlp_attack_bb.fit(X_train[:attack_train_size].astype(np.float32),y_train[:attack_train_size].astype(np.float32),
                    X_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size].astype(np.float32))
  mlp_inferred_train_bb = mlp_attack_bb.infer(X_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:])
  mlp_inferred_test_bb = mlp_attack_bb.infer(X_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])

  mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
  mlp_test_acc_bb = 1-(np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
  mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + 
                                                                                                                len(mlp_inferred_test_bb))

  print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
  print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
  print(f"Attack Accuracy {mlp_acc_bb:.4f}")

  precision, recall = calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)),
                              np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb)))))
  y_train_pred = np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb))
  y_train_true = np.concatenate((np.ones_like(mlp_inferred_train_bb), np.zeros_like(mlp_inferred_test_bb)))
  #print(classification_report(y_pred=y_train_pred, y_true=y_train_true))
  return mlp_train_acc_bb, mlp_test_acc_bb, mlp_acc_bb, precision, recall

trained_model = train(model, train_loader, epochs=35)
# Save model checkpoint
checkpoint_path = os.path.join(checkpoint_dir, f'epoch_35_1.pth')
torch.save(trained_model, checkpoint_path)
print(f"Model checkpoint saved at {checkpoint_path}")
#checkpoint_path = os.path.join(checkpoint_dir, f'epoch_14_01.pth')
#CNN_model = CNN()
#CNN_model=torch.load(checkpoint_path)
#CNN_model.eval()
#loss, accuracy = test(CNN_model, test_loader)
#_attackMIA(X_train, X_test, y_train, y_test, CNN_model)