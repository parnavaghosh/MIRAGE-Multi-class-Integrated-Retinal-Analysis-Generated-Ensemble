
import zipfile
import os

def unzip_dataset(zip_path, extract_to):

    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Directory '{extract_to}' already exists. ")

# Example usage:
zip_path = "/content/MIRAGE.zip"
extract_dir = "/content/MIRAGE_extracted"
unzip_dataset(zip_path, extract_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import time
import copy
import zipfile
import shutil

# --- 1. Data Loading and Augmentation ---

def get_data_loaders(data_dir, batch_size=32):

    # Define transformations
    # For training:
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),        # 50% chance to flip
    transforms.RandomRotation(13),            # Rotate
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # For validation and test:
    val_test_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),        # 50% chance to flip
    transforms.RandomRotation(13),            # Rotate
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), val_test_transforms),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), val_test_transforms)
    }

    # Create data loaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'valid', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Number of classes: {len(class_names)}")
    print("Class names:", class_names)
    print("Dataset sizes:", dataset_sizes)


    return dataloaders, dataset_sizes, class_names

#  2. Model Setup ---

def setup_model(num_classes, freeze_features=True):

    # Load pre-trained VGG16 with the default weights
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze convolutional layers
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Feature extraction layers are frozen.")


    # Replace the classifier
    # VGG16's classifier has 4096 input
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        # nn.Dropout(0.5), # Add dropout for regularization
        nn.Linear(512, num_classes)
    )

    return model

#  3. Training Loop ---

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Tracks history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize  in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Stat
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # Adjust learning rate based on validation loss
                scheduler.step(epoch_loss)


            # Deep copy the model if it's the best so far
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

#  4. Plotting and Evaluation

def plot_loss_curves(history):
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model, dataloader, device, class_names):
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f'\nTest Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

#  5. Main method

if __name__ == '__main__':
    # Hyperparameters and Setup
    DATA_DIR = '/content/MIRAGE_extracted' # Folder to extract data into
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4 # L2 regularization

    # Proceed with Training
    # Final check to ensure the data directory is usable
    if not os.path.isdir(os.path.join(DATA_DIR, 'train')):
         print(f"ERROR: The directory '{DATA_DIR}' does not contain a 'train' sub-folder.")
         print("Please ensure your zip file extracts into train/validation/test folders.")
         exit()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    dataloaders, dataset_sizes, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)

    #  Model Setup
    model = setup_model(num_classes=num_classes, freeze_features=True)
    model = model.to(device)

    # Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    #  Training
    trained_model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS)

    #  Visualization and Evaluation
    plot_loss_curves(history)
    print("\n--- Evaluating on Test Set ---")
    evaluate_model(trained_model, dataloaders['test'], device, class_names)

import os
from torchvision import datasets

# Path to your training data directory
train_data_dir = '/content/MIRAGE_extracted/train'  # Change this to your actual path

# Load dataset
dataset = datasets.ImageFolder(train_data_dir)

# Get mapping: class name ➝ index
class_to_idx = dataset.class_to_idx
print("Class-to-Index Mapping:")
for class_name, idx in class_to_idx.items():
    print(f"{idx}: {class_name}")

# Optional: reverse mapping (index ➝ class name)
idx_to_class = {v: k for k, v in class_to_idx.items()}

print("\nIndex-to-Class Mapping:")
for idx, class_name in idx_to_class.items():
    print(f"{idx}: {class_name}")

# Optional: Save mapping to JSON for use in deployment
import json

with open("class_mapping.json", "w") as f:
    json.dump(idx_to_class, f, indent=4)

print("\nMapping saved to class_mapping.json")