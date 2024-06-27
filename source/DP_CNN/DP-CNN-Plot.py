import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'DP-CNN.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Extract relevant columns
epochs = data['Epoch']
train_loss = data['Loss']
train_accuracy = data['Accuracy']
val_loss = data['Val_Loss']
val_accuracy = data['Val_Accuracy']

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot loss values
ax1.plot(epochs, train_loss, label='Train Loss')
ax1.plot(epochs, val_loss, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss over Epochs')
ax1.legend()
ax1.grid(True)

# Plot accuracy values
ax2.plot(epochs, train_accuracy, label='Train Accuracy')
ax2.plot(epochs, val_accuracy, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy over Epochs')
ax2.legend()
ax2.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
